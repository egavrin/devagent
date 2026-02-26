/**
 * LSP Client — manages a language server as a child process.
 * Communicates via JSON-RPC over stdio.
 * Uses string method names for JSON-RPC calls to avoid version mismatches
 * between vscode-jsonrpc and vscode-languageserver-protocol.
 * Fail fast: if the server crashes, report the error — don't silently degrade.
 */

import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import {
  createMessageConnection,
  StreamMessageReader,
  StreamMessageWriter,
} from "vscode-jsonrpc/node.js";
import type { MessageConnection } from "vscode-jsonrpc/node.js";
import type {
  InitializeResult,
  Diagnostic,
  Location,
  SymbolInformation,
  DocumentSymbol,
  PublishDiagnosticsParams,
} from "vscode-languageserver-protocol";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

// ─── Types ──────────────────────────────────────────────────

export interface LSPClientOptions {
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

export interface DiagnosticResult {
  readonly file: string;
  readonly diagnostics: ReadonlyArray<{
    readonly line: number;
    readonly character: number;
    readonly message: string;
    readonly severity: string;
  }>;
}

export interface SymbolResult {
  readonly name: string;
  readonly kind: string;
  readonly line: number;
  readonly character: number;
  readonly containerName?: string;
}

export interface LocationResult {
  readonly file: string;
  readonly line: number;
  readonly character: number;
}

// ─── LSP Client ─────────────────────────────────────────────

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
  private diagnosticsStore = new Map<string, Diagnostic[]>();
  private openDocuments = new Map<string, { version: number; content: string }>();

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

    this.process = spawn(this.command, [...this.args], {
      stdio: ["pipe", "pipe", "pipe"],
      cwd: this.rootPath,
    });

    if (!this.process.stdin || !this.process.stdout) {
      this.process.kill();
      this.process = null;
      throw new Error("Failed to create language server stdio streams");
    }

    // Detect server crash and mark client as dead so callers can restart.
    this.process.on("exit", () => {
      this.initialized = false;
      this.openDocuments.clear();
      this.connection?.dispose();
      this.connection = null;
      this.process = null;
    });

    this.connection = createMessageConnection(
      new StreamMessageReader(this.process.stdout),
      new StreamMessageWriter(this.process.stdin),
    );

    // Listen for published diagnostics
    this.connection.onNotification(
      "textDocument/publishDiagnostics",
      (params: PublishDiagnosticsParams) => {
        this.diagnosticsStore.set(params.uri, params.diagnostics);
      },
    );

    this.connection.listen();

    try {
      // Initialize
      await this.withTimeout<InitializeResult>(
        this.connection.sendRequest("initialize", {
          processId: process.pid,
          rootUri: `file://${this.rootPath}`,
          capabilities: {
            textDocument: {
              synchronization: { dynamicRegistration: false },
              publishDiagnostics: { relatedInformation: true },
              definition: { dynamicRegistration: false },
              references: { dynamicRegistration: false },
              documentSymbol: { dynamicRegistration: false },
            },
          },
          workspaceFolders: [
            { uri: `file://${this.rootPath}`, name: "workspace" },
          ],
        }),
      );

      this.connection.sendNotification("initialized", {});
      this.initialized = true;
    } catch (err) {
      this.connection.dispose();
      this.connection = null;
      this.process?.kill();
      this.process = null;
      this.initialized = false;
      this.openDocuments.clear();
      throw err;
    }
  }

  async stop(): Promise<void> {
    const connection = this.connection;
    const processRef = this.process;
    if (!connection && !processRef) return;

    try {
      if (connection && this.initialized) {
        this.closeAllOpenDocuments();
        await this.withTimeout(connection.sendRequest("shutdown"));
        connection.sendNotification("exit");
      }
    } catch {
      // Server might already be dead
    }

    connection?.dispose();
    processRef?.kill();
    this.connection = null;
    this.process = null;
    this.initialized = false;
    this.openDocuments.clear();
  }

  isRunning(): boolean {
    return this.initialized;
  }

  async getDiagnostics(filePath: string, languageId?: string): Promise<DiagnosticResult> {
    this.ensureRunning();
    const { uri } = await this.openOrUpdateDocument(filePath, languageId);

    // Clear stale diagnostics from previous runs for this URI.
    this.diagnosticsStore.delete(uri);

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

  // ─── Private ────────────────────────────────────────────────

  private ensureRunning(): void {
    if (!this.initialized || !this.connection) {
      throw new Error("LSP client not initialized. Call start() first.");
    }
  }

  private async openOrUpdateDocument(
    filePath: string,
    languageId?: string,
  ): Promise<{ uri: string }> {
    this.ensureRunning();
    const absPath = resolve(this.rootPath, filePath);
    const uri = `file://${absPath}`;
    const content = await readFile(absPath, "utf-8");
    const existing = this.openDocuments.get(uri);

    if (!existing) {
      this.connection!.sendNotification("textDocument/didOpen", {
        textDocument: {
          uri,
          languageId: languageId ?? this.languageId,
          version: 1,
          text: content,
        },
      });
      this.openDocuments.set(uri, { version: 1, content });
      return { uri };
    }

    if (existing.content !== content) {
      const nextVersion = existing.version + 1;
      this.connection!.sendNotification("textDocument/didChange", {
        textDocument: { uri, version: nextVersion },
        contentChanges: [{ text: content }],
      });
      this.openDocuments.set(uri, { version: nextVersion, content });
    }

    return { uri };
  }

  private closeAllOpenDocuments(): void {
    if (!this.connection) return;
    for (const uri of this.openDocuments.keys()) {
      this.connection.sendNotification("textDocument/didClose", {
        textDocument: { uri },
      });
    }
    this.openDocuments.clear();
  }

  private async withTimeout<T>(promise: Promise<T>): Promise<T> {
    let timer: ReturnType<typeof setTimeout>;
    try {
      return await Promise.race([
        promise,
        new Promise<T>((_, reject) => {
          timer = setTimeout(
            () => reject(new Error("LSP request timed out")),
            this.timeout,
          );
        }),
      ]);
    } finally {
      clearTimeout(timer!);
    }
  }
}

// ─── Helpers ────────────────────────────────────────────────

function severityToString(severity: number | undefined): string {
  switch (severity) {
    case 1: return "error";
    case 2: return "warning";
    case 3: return "info";
    case 4: return "hint";
    default: return "unknown";
  }
}

function flattenSymbols(
  symbols: Array<SymbolInformation | DocumentSymbol>,
  containerName?: string,
): SymbolResult[] {
  const results: SymbolResult[] = [];
  for (const sym of symbols) {
    if ("range" in sym) {
      results.push({
        name: sym.name,
        kind: symbolKindToString(sym.kind),
        line: sym.range.start.line + 1,
        character: sym.range.start.character + 1,
        containerName,
      });
      if (sym.children) {
        results.push(...flattenSymbols(sym.children, sym.name));
      }
    } else {
      results.push({
        name: sym.name,
        kind: symbolKindToString(sym.kind),
        line: sym.location.range.start.line + 1,
        character: sym.location.range.start.character + 1,
        containerName: sym.containerName,
      });
    }
  }
  return results;
}

function symbolKindToString(kind: number): string {
  const kinds: Record<number, string> = {
    1: "file", 2: "module", 3: "namespace", 4: "package",
    5: "class", 6: "method", 7: "property", 8: "field",
    9: "constructor", 10: "enum", 11: "interface", 12: "function",
    13: "variable", 14: "constant", 15: "string", 16: "number",
    17: "boolean", 18: "array", 19: "object", 20: "key",
    21: "null", 22: "enum_member", 23: "struct", 24: "event",
    25: "operator", 26: "type_parameter",
  };
  return kinds[kind] ?? `kind_${kind}`;
}
