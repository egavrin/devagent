/**
 * LSP Client — manages a language server as a child process.
 * Communicates via JSON-RPC over stdio.
 * Uses string method names for JSON-RPC calls to avoid version mismatches
 * between vscode-jsonrpc and vscode-languageserver-protocol.
 * Fail fast: if the server crashes, report the error — don't silently degrade.
 */

import { spawn } from "node:child_process";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  createMessageConnection,
  StreamMessageReader,
  StreamMessageWriter,
} from "vscode-jsonrpc/node.js";

import {
  callHierarchyItemToResult,
  flattenSymbols,
  formatHoverContents,
  normalizeLocations,
  severityToString,
  workspaceSymbolToResult,
  type CallHierarchyResult,
  type DiagnosticResult,
  type LocationResult,
  type SymbolResult,
  type WorkspaceSymbolResult,
} from "./client-format.js";
import type { ChildProcess } from "node:child_process";
import type { MessageConnection } from "vscode-jsonrpc/node.js";
import type {
  InitializeResult,
  Diagnostic,
  Location,
  SymbolInformation,
  DocumentSymbol,
  PublishDiagnosticsParams,
} from "vscode-languageserver-protocol";


// ─── Types ──────────────────────────────────────────────────

interface LSPClientOptions {
  readonly command: string;
  readonly args: ReadonlyArray<string>;
  readonly rootPath: string;
  readonly languageId: string;
  /** Timeout for LSP requests (initialize, definition, etc.) in ms. Default: 10000. */
  readonly timeout?: number;
  /** How long to wait for diagnostics (pushed asynchronously) in ms. Default: 3000.
   *  Servers like rust-analyzer delegate to `cargo check` which can take 15-30s. */
  readonly diagnosticTimeout?: number;
}

interface OpenDocumentResult {
  readonly uri: string;
  readonly sentVersionNotification: boolean;
}

interface LSPClientStopOptions {
  readonly deadlineMs?: number;
}

class LSPTimeoutError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "LSPTimeoutError";
  }
}

// ─── LRU Map ────────────────────────────────────────────────

/**
 * Simple LRU cache built on a plain Map (which preserves insertion order in JS).
 * On get/set the entry is moved to the end (most-recently-used).
 * When the capacity is exceeded the oldest entry (first key) is evicted.
 */
class LRUMap<K, V> {
  private readonly map = new Map<K, V>();
  private readonly capacity: number;

  constructor(capacity: number) {
    if (capacity < 1) throw new Error("LRUMap capacity must be >= 1");
    this.capacity = capacity;
  }

  get(key: K): V | undefined {
    const value = this.map.get(key);
    if (value === undefined) return undefined;
    // Move to end (most-recently-used)
    this.map.delete(key);
    this.map.set(key, value);
    return value;
  }

  has(key: K): boolean {
    return this.map.has(key);
  }

  set(key: K, value: V): void {
    // If already present, delete first so re-insert moves to end
    if (this.map.has(key)) {
      this.map.delete(key);
    } else if (this.map.size >= this.capacity) {
      // Evict least-recently-used (first key)
      const oldest = this.map.keys().next().value;
      if (oldest !== undefined) {
        this.map.delete(oldest);
      }
    }
    this.map.set(key, value);
  }

  delete(key: K): boolean {
    return this.map.delete(key);
  }

  keys(): MapIterator<K> {
    return this.map.keys();
  }

  clear(): void {
    this.map.clear();
  }

  get size(): number {
    return this.map.size;
  }
}

// ─── LSP Client ─────────────────────────────────────────────

/** Maximum number of entries in the diagnostics LRU cache. */
const DIAGNOSTICS_LRU_CAP = 100;
/** Maximum number of entries in the open-documents LRU cache. */
const OPEN_DOCUMENTS_LRU_CAP = 100;

function getErrorCode(error: unknown): string {
  if (typeof error === "object" && error !== null && "code" in error) {
    return String((error as { code?: unknown }).code ?? "");
  }
  return "";
}

function getErrorMessage(error: unknown): string {
  if (error instanceof Error) return error.message;
  if (typeof error === "object" && error !== null && "message" in error) {
    return String((error as { message?: unknown }).message ?? "");
  }
  return String(error);
}

function isIgnorableTransportError(error: unknown): boolean {
  const code = getErrorCode(error);
  if (code === "ERR_STREAM_DESTROYED" || code === "EPIPE") {
    return true;
  }

  const message = getErrorMessage(error).toLowerCase();
  return message.includes("stream was destroyed") ||
    message.includes("write after end") ||
    message.includes("broken pipe") ||
    message.includes("connection is closed") ||
    message.includes("connection got disposed") ||
    message.includes("connection is disposed");
}

function isPresent<T>(value: T | null | undefined): value is T {
  return value !== null && value !== undefined;
}

async function sendLSPNotification(
  connection: MessageConnection,
  method: string,
  params: unknown,
  options?: { ignoreTransportErrors?: boolean; timeoutMs?: number; onTimeout?: () => void },
): Promise<void> {
  try {
    await withPromiseTimeout(
      connection.sendNotification(method, params),
      options?.timeoutMs,
      `LSP notification ${method} timed out`,
    );
  } catch (error) {
    if (options?.ignoreTransportErrors && isIgnorableTransportError(error)) {
      return;
    }
    if (error instanceof LSPTimeoutError) {
      options?.onTimeout?.();
    }
    throw error;
  }
}

function createLSPConnection(processRef: ChildProcess): MessageConnection {
  if (!processRef.stdin || !processRef.stdout) {
    throw new Error("Failed to create language server stdio streams");
  }
  return createMessageConnection(
    new StreamMessageReader(processRef.stdout),
    new StreamMessageWriter(processRef.stdin),
  );
}

async function withPromiseTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number | undefined,
  message: string,
): Promise<T> {
  if (!timeoutMs || timeoutMs <= 0) return promise;
  let timer: ReturnType<typeof setTimeout>;
  try {
    return await Promise.race([
      promise,
      new Promise<T>((_, reject) => {
        timer = setTimeout(() => reject(new LSPTimeoutError(message)), timeoutMs);
        unrefTimer(timer);
      }),
    ]);
  } finally {
    clearTimeout(timer!);
  }
}

function unrefTimer(timer: ReturnType<typeof setTimeout>): void {
  (timer as { unref?: () => void }).unref?.();
}

export class LSPClient {
  private process: ChildProcess | null = null;
  private connection: MessageConnection | null = null;
  private readonly command: string;
  private readonly args: ReadonlyArray<string>;
  private readonly rootPath: string;
  private readonly languageId: string;
  private readonly timeout: number;
  private readonly diagnosticTimeout: number;
  private initialized = false;
  private diagnosticsStore = new LRUMap<string, Diagnostic[]>(DIAGNOSTICS_LRU_CAP);
  private openDocuments = new LRUMap<string, { version: number; content: string }>(OPEN_DOCUMENTS_LRU_CAP);

  constructor(options: LSPClientOptions) {
    this.command = options.command;
    this.args = options.args;
    this.rootPath = options.rootPath;
    this.languageId = options.languageId;
    this.timeout = options.timeout ?? 10_000;
    this.diagnosticTimeout = options.diagnosticTimeout ?? 3_000;
  }

  async start(): Promise<void> {
    if (this.initialized) return;

    this.process = this.spawnLanguageServer();
    this.registerProcessExitHandler();
    this.connection = createLSPConnection(this.process);
    this.registerDiagnosticsHandler();
    this.connection.listen();
    await this.initializeConnection();
  }

  private async initializeConnection(): Promise<void> {
    try {
      await this.withTimeout<InitializeResult>(
        this.connection!.sendRequest("initialize", {
          processId: process.pid,
          rootUri: `file://${this.rootPath}`,
          capabilities: {
            workspace: {
              symbol: { dynamicRegistration: false },
            },
            textDocument: {
              synchronization: { dynamicRegistration: false, didSave: true },
              publishDiagnostics: { relatedInformation: true },
              hover: {
                dynamicRegistration: false,
                contentFormat: ["markdown", "plaintext"],
              },
              definition: { dynamicRegistration: false },
              references: { dynamicRegistration: false },
              documentSymbol: { dynamicRegistration: false },
              implementation: { dynamicRegistration: false },
              callHierarchy: { dynamicRegistration: false },
            },
          },
          workspaceFolders: [
            { uri: `file://${this.rootPath}`, name: "workspace" },
          ],
        }),
      );

      await sendLSPNotification(this.connection!, "initialized", {}, { timeoutMs: this.timeout });
      this.initialized = true;
    } catch (err) {
      this.connection?.dispose();
      this.connection = null;
      this.process?.kill();
      this.process = null;
      this.initialized = false;
      this.openDocuments.clear();
      throw err;
    }
  }

  private spawnLanguageServer(): ChildProcess {
    const processRef = spawn(this.command, [...this.args], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: this.rootPath,
    });
    if (!processRef.stdin || !processRef.stdout) {
      processRef.kill();
      throw new Error("Failed to create language server stdio streams");
    }
    return processRef;
  }

  private registerProcessExitHandler(): void {
    this.process?.on("exit", () => {
      this.initialized = false;
      this.openDocuments.clear();
      this.diagnosticsStore.clear();
      this.connection?.dispose();
      this.connection = null;
      this.process = null;
    });
  }

  private registerDiagnosticsHandler(): void {
    this.connection?.onNotification(
      "textDocument/publishDiagnostics",
      (params: PublishDiagnosticsParams) => {
        this.diagnosticsStore.set(params.uri, params.diagnostics);
      },
    );
  }

  async stop(options?: LSPClientStopOptions): Promise<void> {
    if (options?.deadlineMs && options.deadlineMs > 0) {
      try {
        await withPromiseTimeout(
          this.stopGracefully(),
          options.deadlineMs,
          `LSP stop timed out after ${options.deadlineMs}ms`,
        );
      } catch {
        // Stop is best-effort during shutdown/restart. Force-dispose below.
      }
      this.forceDispose();
      return;
    }

    await this.stopGracefully();
  }

  private async stopGracefully(): Promise<void> {
    const connection = this.connection;
    const processRef = this.process;
    if (!connection && !processRef) return;

    try {
      if (connection && this.initialized) {
        await this.closeAllOpenDocuments(connection);
        await this.withTimeout(connection.sendRequest("shutdown"));
        await sendLSPNotification(connection, "exit", {}, { ignoreTransportErrors: true, timeoutMs: this.timeout });
      }
    } catch {
      // Server might already be dead
    }

    this.forceDispose(connection, processRef);
  }

  isRunning(): boolean {
    return this.initialized;
  }

  async getDiagnostics(filePath: string, languageId?: string): Promise<DiagnosticResult> {
    this.ensureRunning();
    const { uri, sentVersionNotification } = await this.openOrUpdateDocument(filePath, languageId);

    if (!sentVersionNotification && this.diagnosticsStore.has(uri)) {
      return this.formatDiagnosticResult(filePath, this.diagnosticsStore.get(uri) ?? []);
    }

    // Poll for diagnostics — servers push them asynchronously.
    // Exit early if diagnostics arrive (saves time for fast servers like clangd).
    // Slow servers (rust-analyzer → cargo check) may need 15-30s; use diagnosticTimeout.
    const maxWait = this.diagnosticTimeout;
    const pollInterval = 100;
    const start = Date.now();
    while (Date.now() - start < maxWait) {
      await new Promise((r) => setTimeout(r, pollInterval));
      if (this.diagnosticsStore.has(uri)) break;
    }
    const diagnostics = this.diagnosticsStore.get(uri) ?? [];

    return this.formatDiagnosticResult(filePath, diagnostics);
  }

  private formatDiagnosticResult(filePath: string, diagnostics: ReadonlyArray<Diagnostic>): DiagnosticResult {
    return {
      file: filePath,
      diagnostics: diagnostics.map((d) => ({
        line: d.range.start.line + 1,
        character: d.range.start.character + 1,
        message: d.message,
        severity: severityToString(d.severity),
      })),
    };
  }

  async syncDocument(
    filePath: string,
    languageId?: string,
    options?: { readonly didSave?: boolean },
  ): Promise<void> {
    this.ensureRunning();
    const { uri } = await this.openOrUpdateDocument(filePath, languageId);
    if (options?.didSave) {
      await this.sendNotification("textDocument/didSave", {
        textDocument: { uri },
      });
    }
  }

  async getDefinition(
    filePath: string,
    line: number,
    character: number,
    languageId?: string,
  ): Promise<ReadonlyArray<LocationResult>> {
    this.ensureRunning();
    const { uri } = await this.openOrUpdateDocument(filePath, languageId);

    const result = await this.withTimeout(
      this.connection!.sendRequest("textDocument/definition", {
        textDocument: { uri },
        position: { line: line - 1, character: character - 1 },
      }),
    );

    if (!result) return [];

    const locations = Array.isArray(result)
      ? (result as Location[])
      : [result as Location];
    return locations.map((loc) => ({
      file: loc.uri.replace(`file://${this.rootPath}/`, ""),
      line: loc.range.start.line + 1,
      character: loc.range.start.character + 1,
    }));
  }

  async getReferences(
    filePath: string,
    line: number,
    character: number,
    languageId?: string,
  ): Promise<ReadonlyArray<LocationResult>> {
    this.ensureRunning();
    const { uri } = await this.openOrUpdateDocument(filePath, languageId);

    const result = await this.withTimeout(
      this.connection!.sendRequest("textDocument/references", {
        textDocument: { uri },
        position: { line: line - 1, character: character - 1 },
        context: { includeDeclaration: true },
      }),
    );

    if (!result) return [];
    return (result as Location[]).map((loc) => ({
      file: loc.uri.replace(`file://${this.rootPath}/`, ""),
      line: loc.range.start.line + 1,
      character: loc.range.start.character + 1,
    }));
  }

  async getHover(
    filePath: string,
    line: number,
    character: number,
    languageId?: string,
  ): Promise<string | null> {
    this.ensureRunning();
    const { uri } = await this.openOrUpdateDocument(filePath, languageId);

    const result = await this.withTimeout(
      this.connection!.sendRequest("textDocument/hover", {
        textDocument: { uri },
        position: { line: line - 1, character: character - 1 },
      }),
    );

    return formatHoverContents(result);
  }

  async getImplementation(
    filePath: string,
    line: number,
    character: number,
    languageId?: string,
  ): Promise<ReadonlyArray<LocationResult>> {
    this.ensureRunning();
    const { uri } = await this.openOrUpdateDocument(filePath, languageId);

    const result = await this.withTimeout(
      this.connection!.sendRequest("textDocument/implementation", {
        textDocument: { uri },
        position: { line: line - 1, character: character - 1 },
      }),
    );

    return normalizeLocations(result, this.rootPath);
  }

  async getWorkspaceSymbols(query = ""): Promise<ReadonlyArray<WorkspaceSymbolResult>> {
    this.ensureRunning();
    const result = await this.withTimeout(
      this.connection!.sendRequest("workspace/symbol", { query }),
    );
    if (!Array.isArray(result)) return [];
    return result.map((symbol) => workspaceSymbolToResult(symbol, this.rootPath)).filter(isPresent);
  }

  async getIncomingCalls(
    filePath: string,
    line: number,
    character: number,
    languageId?: string,
  ): Promise<ReadonlyArray<CallHierarchyResult>> {
    return this.getCallHierarchy(filePath, line, character, "incoming", languageId);
  }

  async getOutgoingCalls(
    filePath: string,
    line: number,
    character: number,
    languageId?: string,
  ): Promise<ReadonlyArray<CallHierarchyResult>> {
    return this.getCallHierarchy(filePath, line, character, "outgoing", languageId);
  }

  async getSymbols(filePath: string, languageId?: string): Promise<ReadonlyArray<SymbolResult>> {
    this.ensureRunning();
    const { uri } = await this.openOrUpdateDocument(filePath, languageId);

    const result = await this.withTimeout(
      this.connection!.sendRequest("textDocument/documentSymbol", {
        textDocument: { uri },
      }),
    );

    if (!result) return [];
    return flattenSymbols(result as Array<SymbolInformation | DocumentSymbol>);
  }

  private async getCallHierarchy(
    filePath: string,
    line: number,
    character: number,
    direction: "incoming" | "outgoing",
    languageId?: string,
  ): Promise<ReadonlyArray<CallHierarchyResult>> {
    this.ensureRunning();
    const { uri } = await this.openOrUpdateDocument(filePath, languageId);
    const position = { line: line - 1, character: character - 1 };
    const prepared = await this.withTimeout(
      this.connection!.sendRequest("textDocument/prepareCallHierarchy", {
        textDocument: { uri },
        position,
      }),
    );
    if (!Array.isArray(prepared) || prepared.length === 0) return [];

    const method = direction === "incoming"
      ? "callHierarchy/incomingCalls"
      : "callHierarchy/outgoingCalls";
    const results: CallHierarchyResult[] = [];
    for (const item of prepared) {
      const calls = await this.withTimeout(
        this.connection!.sendRequest(method, { item }),
      );
      if (!Array.isArray(calls)) continue;
      for (const call of calls) {
        const hierarchyItem = direction === "incoming"
          ? (call as { from?: unknown }).from
          : (call as { to?: unknown }).to;
        const formatted = callHierarchyItemToResult(hierarchyItem, this.rootPath);
        if (formatted) results.push(formatted);
      }
    }
    return results;
  }

  // ─── Private ────────────────────────────────────────────────

  private ensureRunning(): void {
    if (!this.initialized || !this.connection) {
      throw new Error("LSP client not initialized. Call start() first.");
    }
  }

  private async openOrUpdateDocument(
    filePath: string,
    languageId?: string,
  ): Promise<OpenDocumentResult> {
    this.ensureRunning();
    const absPath = resolve(this.rootPath, filePath);
    const uri = `file://${absPath}`;
    const content = await readFile(absPath, "utf-8");
    const existing = this.openDocuments.get(uri);

    if (!existing) {
      this.diagnosticsStore.delete(uri);
      await this.sendNotification("textDocument/didOpen", {
        textDocument: {
          uri,
          languageId: languageId ?? this.languageId,
          version: 1,
          text: content,
        },
      });
      this.openDocuments.set(uri, { version: 1, content });
      return { uri, sentVersionNotification: true };
    }

    if (existing.content !== content) {
      const nextVersion = existing.version + 1;
      this.diagnosticsStore.delete(uri);
      await this.sendNotification("textDocument/didChange", {
        textDocument: { uri, version: nextVersion },
        contentChanges: [{ text: content }],
      });
      this.openDocuments.set(uri, { version: nextVersion, content });
      return { uri, sentVersionNotification: true };
    }

    return { uri, sentVersionNotification: false };
  }

  private async closeAllOpenDocuments(connection: MessageConnection): Promise<void> {
    for (const uri of this.openDocuments.keys()) {
      await sendLSPNotification(connection, "textDocument/didClose", {
        textDocument: { uri },
      }, {
        ignoreTransportErrors: true,
        timeoutMs: this.timeout,
        onTimeout: () => this.markUnhealthy(),
      });
    }
    this.openDocuments.clear();
  }

  private async sendNotification(method: string, params: unknown): Promise<void> {
    await sendLSPNotification(this.connection!, method, params, {
      timeoutMs: this.timeout,
      onTimeout: () => this.markUnhealthy(),
    });
  }

  private async withTimeout<T>(promise: Promise<T>): Promise<T> {
    let timer: ReturnType<typeof setTimeout>;
    try {
      return await Promise.race([
        promise,
        new Promise<T>((_, reject) => {
          timer = setTimeout(
            () => reject(new LSPTimeoutError("LSP request timed out")),
            this.timeout,
          );
          unrefTimer(timer);
        }),
      ]);
    } catch (error) {
      if (error instanceof LSPTimeoutError) {
        this.markUnhealthy();
      }
      throw error;
    } finally {
      clearTimeout(timer!);
    }
  }

  private markUnhealthy(): void {
    this.forceDispose();
  }

  private forceDispose(
    connection: MessageConnection | null = this.connection,
    processRef: ChildProcess | null = this.process,
  ): void {
    (connection as { dispose?: () => void } | null)?.dispose?.();
    processRef?.kill();
    this.connection = null;
    this.process = null;
    this.initialized = false;
    this.openDocuments.clear();
    this.diagnosticsStore.clear();
  }
}
