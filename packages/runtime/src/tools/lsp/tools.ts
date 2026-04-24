/**
 * LSP-powered code intelligence tool.
 *
 * DevAgent exposes one generic readonly `lsp` tool instead of separate
 * operation-specific tools. This keeps the prompt surface compact while still
 * exposing precise language-server operations.
 */

import type { LSPClient } from "./client.js";
import { extractErrorMessage } from "../../core/errors.js";
import type { ToolErrorGuidance, ToolResult, ToolSpec } from "../../core/types.js";

// ─── Types ──────────────────────────────────────────────────

/** Resolves the correct LSP client for a file path (e.g. by extension). */
export type LSPClientResolver = (filePath: string) => {
  client: LSPClient;
  languageId: string;
} | null;

export type LSPClientProvider = () => ReadonlyArray<LSPClient>;

type LSPClientMatch = NonNullable<ReturnType<LSPClientResolver>>;

type WorkspaceSymbolResult = Awaited<ReturnType<LSPClient["getWorkspaceSymbols"]>>[number];

interface WorkspaceSymbolAttempt {
  readonly symbols: ReadonlyArray<WorkspaceSymbolResult>;
  readonly successCount: number;
  readonly failures: ReadonlyArray<string>;
}

type LSPOperation =
  | "diagnostics"
  | "definitions"
  | "references"
  | "symbols"
  | "hover"
  | "implementations"
  | "workspace_symbols"
  | "incoming_calls"
  | "outgoing_calls";

const LSP_OPERATIONS: ReadonlyArray<LSPOperation> = [
  "diagnostics",
  "definitions",
  "references",
  "symbols",
  "hover",
  "implementations",
  "workspace_symbols",
  "incoming_calls",
  "outgoing_calls",
];

const MAX_LSP_RESULTS = 100;
const LSP_OPERATION_TIMEOUT_MS = 30_000;
const WORKSPACE_SYMBOL_RETRY_DELAYS_MS = [100, 250] as const;

const LSP_PARAM_SCHEMA = {
  type: "object",
  properties: {
    operation: {
      type: "string",
      enum: LSP_OPERATIONS,
      description: "LSP operation to run",
    },
    path: {
      type: "string",
      description: "File path relative to repo root. Required for file and position operations.",
    },
    line: {
      type: "number",
      description: "Line number (1-based). Required for position operations.",
    },
    character: {
      type: "number",
      description: "Column number (1-based). Required for position operations.",
    },
    query: {
      type: "string",
      description: "Workspace symbol search query. Optional for workspace_symbols.",
    },
  },
  required: ["operation"],
  additionalProperties: false,
};

const LSP_RESULT_SCHEMA = {
  type: "object",
  properties: {
    result: { type: "string" },
    count: { type: "number" },
  },
};

const LSP_GUIDANCE: ToolErrorGuidance = {
  common:
    "LSP operation failed. Verify the file path, position, and that a language server is configured for this file type.",
  patterns: [
    {
      match: "No LSP server",
      hint: "No LSP server is available for this file type. Check LSP configuration or install the relevant server.",
    },
    {
      match: "not running",
      hint: "The LSP client is not running. The language server may need to be restarted.",
    },
    {
      match: "timed out",
      hint: "The LSP request timed out. Try a narrower operation or restart the language server.",
    },
    {
      match: "requires",
      hint: "The operation is missing required input. Use path for file operations, path/line/character for position operations, and query for workspace_symbols when useful.",
    },
  ],
};

// ─── Tool Factory ───────────────────────────────────────────

/**
 * Create a single-client LSP tool. Kept for programmatic callers that bind one
 * language server directly.
 */
export function createLSPTools(client: LSPClient): ReadonlyArray<ToolSpec> {
  return [
    createLSPTool({
      resolver: () => ({ client, languageId: "" }),
      getClients: () => [client],
    }),
  ];
}

/**
 * Create a routed LSP tool that selects clients by file path.
 */
export function createRoutingLSPTools(
  resolver: LSPClientResolver,
  getClients: LSPClientProvider = () => [],
): ReadonlyArray<ToolSpec> {
  return [createLSPTool({ resolver, getClients })];
}

function createLSPTool(options: {
  readonly resolver: LSPClientResolver;
  readonly getClients: LSPClientProvider;
}): ToolSpec {
  return {
    name: "lsp",
    description:
      "Run Language Server Protocol code intelligence. Operations: diagnostics, symbols, definitions, references, hover, implementations, workspace_symbols, incoming_calls, outgoing_calls.",
    category: "readonly",
    paramSchema: LSP_PARAM_SCHEMA,
    resultSchema: LSP_RESULT_SCHEMA,
    errorGuidance: LSP_GUIDANCE,
    handler: async (params) => runLSPOperation(options, params),
  };
}

async function runLSPOperation(
  options: { readonly resolver: LSPClientResolver; readonly getClients: LSPClientProvider },
  params: Record<string, unknown>,
): Promise<ToolResult> {
  try {
    const operation = parseOperation(params["operation"]);
    const outputPromise = operation === "workspace_symbols"
      ? runWorkspaceSymbols(options.getClients, params)
      : runFileOperation(options.resolver, operation, params);
    const output = await withOperationTimeout(outputPromise);
    return { success: true, output, error: null, artifacts: [] };
  } catch (error) {
    return {
      success: false,
      output: "",
      error: extractErrorMessage(error),
      artifacts: [],
    };
  }
}

async function withOperationTimeout(output: Promise<string>): Promise<string> {
  let timer: ReturnType<typeof setTimeout>;
  try {
    return await Promise.race([
      output,
      new Promise<string>((_, reject) => {
        timer = setTimeout(
          () => reject(new Error(`LSP operation timed out after ${LSP_OPERATION_TIMEOUT_MS}ms`)),
          LSP_OPERATION_TIMEOUT_MS,
        );
        (timer as { unref?: () => void }).unref?.();
      }),
    ]);
  } finally {
    clearTimeout(timer!);
  }
}

function parseOperation(value: unknown): LSPOperation {
  if (typeof value !== "string" || !LSP_OPERATIONS.includes(value as LSPOperation)) {
    throw new Error(`Invalid lsp operation: ${String(value)}`);
  }
  return value as LSPOperation;
}

async function runWorkspaceSymbols(
  getClients: LSPClientProvider,
  params: Record<string, unknown>,
): Promise<string> {
  const query = typeof params["query"] === "string" ? params["query"] : "";
  const clients = getClients().filter((client) => client.isRunning());
  if (clients.length === 0) return "No LSP clients are running.";

  const symbols = await getWorkspaceSymbolsWithColdIndexRetry(clients, query);
  if (symbols.length === 0) return query
    ? `No workspace symbols found for query "${query}".`
    : "No workspace symbols found.";
  return `${symbols.length} workspace symbol(s):\n${formatWorkspaceSymbols(symbols)}`;
}

async function getWorkspaceSymbolsWithColdIndexRetry(
  clients: ReadonlyArray<LSPClient>,
  query: string,
): Promise<Awaited<ReturnType<LSPClient["getWorkspaceSymbols"]>>> {
  let lastAllFailedMessages: ReadonlyArray<string> = [];
  let lastSuccessfulSymbols: ReadonlyArray<WorkspaceSymbolResult> | null = null;
  const delays = [0, ...WORKSPACE_SYMBOL_RETRY_DELAYS_MS] as const;

  for (const [attemptIndex, delayMs] of delays.entries()) {
    if (delayMs > 0) await sleep(delayMs);

    const attempt = await requestWorkspaceSymbols(clients, query);
    if (attempt.successCount > 0) {
      lastSuccessfulSymbols = attempt.symbols;
      if (attempt.symbols.length > 0 || attemptIndex === delays.length - 1) {
        return attempt.symbols.slice(0, MAX_LSP_RESULTS);
      }
      continue;
    }

    lastAllFailedMessages = attempt.failures;
  }

  if (lastSuccessfulSymbols) {
    return lastSuccessfulSymbols.slice(0, MAX_LSP_RESULTS);
  }

  throw new Error(
    lastAllFailedMessages.length > 0
      ? `workspace_symbols failed for all LSP clients: ${lastAllFailedMessages.join("; ")}`
      : "workspace_symbols failed for all LSP clients",
  );
}

async function requestWorkspaceSymbols(
  clients: ReadonlyArray<LSPClient>,
  query: string,
): Promise<WorkspaceSymbolAttempt> {
  const settled = await Promise.allSettled(
    clients.map((client) => client.getWorkspaceSymbols(query)),
  );
  const symbols: WorkspaceSymbolResult[] = [];
  const failures: string[] = [];
  let successCount = 0;

  for (const result of settled) {
    if (result.status === "fulfilled") {
      successCount++;
      symbols.push(...result.value);
    } else {
      failures.push(extractErrorMessage(result.reason));
    }
  }

  return { symbols, successCount, failures };
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function runFileOperation(
  resolver: LSPClientResolver,
  operation: Exclude<LSPOperation, "workspace_symbols">,
  params: Record<string, unknown>,
): Promise<string> {
  const path = requireString(params["path"], `${operation} requires path`);
  const match = resolver(path);
  if (!match) throw new Error(`No LSP server available for ${path}`);
  if (!match.client.isRunning()) throw new Error(`LSP client not running for ${path}`);

  if (operation === "diagnostics") {
    return formatDiagnostics(path, await match.client.getDiagnostics(path, match.languageId));
  }
  if (operation === "symbols") {
    return formatDocumentSymbols(path, await match.client.getSymbols(path, match.languageId));
  }
  return runPositionOperation(match, operation, path, params);
}

async function runPositionOperation(
  match: LSPClientMatch,
  operation: Exclude<LSPOperation, "workspace_symbols" | "diagnostics" | "symbols">,
  path: string,
  params: Record<string, unknown>,
): Promise<string> {
  const position = requirePosition(params, operation);
  return runResolvedPositionOperation(match, operation, path, position);
}

async function runResolvedPositionOperation(
  match: LSPClientMatch,
  operation: Exclude<LSPOperation, "workspace_symbols" | "diagnostics" | "symbols">,
  path: string,
  position: { readonly line: number; readonly character: number },
): Promise<string> {
  switch (operation) {
    case "definitions":
      return formatLocations("Definition(s)", await match.client.getDefinition(path, position.line, position.character, match.languageId));
    case "references":
      return formatReferences(await match.client.getReferences(path, position.line, position.character, match.languageId));
    case "hover":
      return await match.client.getHover(path, position.line, position.character, match.languageId)
        ?? "No hover information found at this position.";
    case "implementations":
      return formatLocations("Implementation(s)", await match.client.getImplementation(path, position.line, position.character, match.languageId));
    case "incoming_calls":
      return formatCallHierarchy("Incoming call(s)", await match.client.getIncomingCalls(path, position.line, position.character, match.languageId));
    case "outgoing_calls":
      return formatCallHierarchy("Outgoing call(s)", await match.client.getOutgoingCalls(path, position.line, position.character, match.languageId));
  }
}

function formatReferences(references: Awaited<ReturnType<LSPClient["getReferences"]>>): string {
  return references.length === 0
    ? "No references found at this position."
    : `${references.length} reference(s):\n${formatLocationLines(references)}`;
}

function requireString(value: unknown, message: string): string {
  if (typeof value !== "string" || value.length === 0) throw new Error(message);
  return value;
}

function requirePosition(
  params: Record<string, unknown>,
  operation: LSPOperation,
): { readonly line: number; readonly character: number } {
  if (typeof params["line"] !== "number" || typeof params["character"] !== "number") {
    throw new Error(`${operation} requires path, line, and character`);
  }
  return {
    line: params["line"],
    character: params["character"],
  };
}

function formatDiagnostics(
  path: string,
  result: Awaited<ReturnType<LSPClient["getDiagnostics"]>>,
): string {
  if (result.diagnostics.length === 0) return `No diagnostics for ${path}. File is clean.`;
  const lines = result.diagnostics
    .map((d) => `${path}:${d.line}:${d.character}: [${d.severity}] ${d.message}`)
    .join("\n");
  return `${result.diagnostics.length} diagnostic(s) in ${path}:\n${lines}`;
}

function formatDocumentSymbols(
  path: string,
  symbols: Awaited<ReturnType<LSPClient["getSymbols"]>>,
): string {
  if (symbols.length === 0) return `No symbols found in ${path}.`;
  return `${symbols.length} symbol(s) in ${path}:\n${symbols.slice(0, MAX_LSP_RESULTS).map((symbol) => {
    const container = symbol.containerName ? ` (in ${symbol.containerName})` : "";
    return `${symbol.kind} ${symbol.name}${container} - ${path}:${symbol.line}:${symbol.character}`;
  }).join("\n")}`;
}

function formatLocations(
  label: string,
  locations: ReadonlyArray<{ readonly file: string; readonly line: number; readonly character: number }>,
): string {
  if (locations.length === 0) return `No ${label.toLowerCase()} found at this position.`;
  return `${label}:\n${formatLocationLines(locations)}`;
}

function formatLocationLines(
  locations: ReadonlyArray<{ readonly file: string; readonly line: number; readonly character: number }>,
): string {
  return locations
    .slice(0, MAX_LSP_RESULTS)
    .map((location) => `${location.file}:${location.line}:${location.character}`)
    .join("\n");
}

function formatWorkspaceSymbols(
  symbols: ReadonlyArray<{
    readonly kind: string;
    readonly name: string;
    readonly file: string;
    readonly line: number;
    readonly character: number;
    readonly containerName?: string;
  }>,
): string {
  return symbols.map((symbol) => {
    const container = symbol.containerName ? ` (in ${symbol.containerName})` : "";
    return `${symbol.kind} ${symbol.name}${container} - ${symbol.file}:${symbol.line}:${symbol.character}`;
  }).join("\n");
}

function formatCallHierarchy(
  label: string,
  calls: ReadonlyArray<{
    readonly kind: string;
    readonly name: string;
    readonly file: string;
    readonly line: number;
    readonly character: number;
    readonly detail?: string;
  }>,
): string {
  if (calls.length === 0) return `No ${label.toLowerCase()} found at this position.`;
  const lines = calls.slice(0, MAX_LSP_RESULTS).map((call) => {
    const detail = call.detail ? ` ${call.detail}` : "";
    return `${call.kind} ${call.name}${detail} - ${call.file}:${call.line}:${call.character}`;
  });
  return `${label}:\n${lines.join("\n")}`;
}
