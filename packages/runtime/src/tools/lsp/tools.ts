/**
 * LSP-powered code analysis tools.
 * These tools use a running LSP client for real compiler intelligence.
 * All are read-only (category: "readonly").
 */

import type { LSPClient } from "./client.js";
import { extractErrorMessage } from "../../core/errors.js";
import type { ToolSpec, ToolErrorGuidance } from "../../core/types.js";

// ─── Types ──────────────────────────────────────────────────

/** Resolves the correct LSP client for a file path (e.g. by extension). */
export type LSPClientResolver = (filePath: string) => {
  client: LSPClient;
  languageId: string;
} | null;

/** Common LSP error patterns shared by all single-client tools. */
const LSP_COMMON_PATTERNS: ReadonlyArray<{ readonly match: string; readonly hint: string }> = [
  {
    match: "not running",
    hint: "LSP client is not running. The language server may not be installed or may have crashed — check its process.",
  },
  {
    match: "timed out",
    hint: "LSP request timed out. The language server may be overloaded — try a smaller file or restart the server.",
  },
];

const PATH_ONLY_PARAM_SCHEMA = {
  type: "object",
  properties: {
    path: { type: "string", description: "File path relative to repo root" },
  },
  required: ["path"],
};

const POSITION_PARAM_SCHEMA = {
  type: "object",
  properties: {
    path: { type: "string", description: "File path relative to repo root" },
    line: { type: "number", description: "Line number (1-based)" },
    character: { type: "number", description: "Column number (1-based)" },
  },
  required: ["path", "line", "character"],
};

const DIAGNOSTICS_RESULT_SCHEMA = {
  type: "object",
  properties: {
    diagnostics: { type: "string" },
    count: { type: "number" },
  },
};

const LOCATIONS_RESULT_SCHEMA = {
  type: "object",
  properties: {
    locations: { type: "string" },
  },
};

const REFERENCES_RESULT_SCHEMA = {
  type: "object",
  properties: {
    references: { type: "string" },
    count: { type: "number" },
  },
};

const SYMBOLS_RESULT_SCHEMA = {
  type: "object",
  properties: {
    symbols: { type: "string" },
    count: { type: "number" },
  },
};

// ─── Tool Factory ───────────────────────────────────────────

/**
 * Create LSP-backed analysis tools bound to a running LSP client.
 * Returns an array of ToolSpec that can be registered in the ToolRegistry.
 */
export function createLSPTools(client: LSPClient): ReadonlyArray<ToolSpec> {
  return [
    createDiagnosticsTool(client),
    createDefinitionTool(client),
    createReferencesTool(client),
    createSymbolsTool(client),
  ];
}

/**
 * Create LSP-backed analysis tools that route to the correct LSP client
 * based on file path. Use this for multi-server setups where different
 * language servers handle different file types.
 */
export function createRoutingLSPTools(
  resolver: LSPClientResolver,
): ReadonlyArray<ToolSpec> {
  return [
    createRoutingDiagnosticsTool(resolver),
    createRoutingDefinitionTool(resolver),
    createRoutingReferencesTool(resolver),
    createRoutingSymbolsTool(resolver),
    createRoutingDefinitionByNameTool(resolver),
    createRoutingReferencesByNameTool(resolver),
  ];
}

function resolveOrFail(
  resolver: LSPClientResolver,
  filePath: string,
): { client: LSPClient; languageId: string } {
  const match = resolver(filePath);
  if (!match) {
    throw new Error(`No LSP server available for ${filePath}`);
  }
  if (!match.client.isRunning()) {
    throw new Error(`LSP client not running for ${filePath}`);
  }
  return match;
}

// ─── diagnostics ────────────────────────────────────────────
function createDiagnosticsTool(client: LSPClient): ToolSpec {
  return {
    name: "diagnostics",
    description:
      "Get compiler diagnostics (errors, warnings) for a file. Uses the language server for real compiler analysis.",
    category: "readonly",
    paramSchema: PATH_ONLY_PARAM_SCHEMA,
    resultSchema: DIAGNOSTICS_RESULT_SCHEMA,
    errorGuidance: {
      common: "Diagnostics failed. Ensure the LSP server is running and the file path is correct.",
      patterns: LSP_COMMON_PATTERNS,
    },
    handler: async (params) => {
      if (!client.isRunning()) return lspFailure("LSP client not running. Language server may not be available.");

      const filePath = params["path"] as string;

      try {
        const result = await client.getDiagnostics(filePath);

        if (result.diagnostics.length === 0) {
          return {
            success: true,
            output: `No diagnostics for ${filePath}. File is clean.`,
            error: null,
            artifacts: [],
          };
        }

        const output = result.diagnostics
          .map((d) => `${filePath}:${d.line}:${d.character}: [${d.severity}] ${d.message}`)
          .join("\n");

        return {
          success: true,
          output: `${result.diagnostics.length} diagnostic(s) in ${filePath}:\n${output}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        return lspFailure(`Diagnostics failed: ${extractErrorMessage(err)}`);
      }
    },
  };
}

// ─── definitions ────────────────────────────────────────────
function createDefinitionTool(client: LSPClient): ToolSpec {
  return {
    name: "definitions",
    description:
      "Go to definition of a symbol at a specific position. Returns the file and location where the symbol is defined.",
    category: "readonly",
    paramSchema: POSITION_PARAM_SCHEMA,
    resultSchema: LOCATIONS_RESULT_SCHEMA,
    errorGuidance: {
      common: "Definition lookup failed. Verify the file path and position — use read_file to confirm the symbol location.",
      patterns: LSP_COMMON_PATTERNS,
    },
    handler: async (params) => {
      if (!client.isRunning()) return lspFailure("LSP client not running.");

      const filePath = params["path"] as string;
      const line = params["line"] as number;
      const character = params["character"] as number;

      try {
        const locations = await client.getDefinition(filePath, line, character);

        if (locations.length === 0) {
          return {
            success: true,
            output: "No definition found at this position.",
            error: null,
            artifacts: [],
          };
        }

        return {
          success: true,
          output: `Definition(s):\n${formatLocations(locations)}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        return lspFailure(`Definition lookup failed: ${extractErrorMessage(err)}`);
      }
    },
  };
}

// ─── references ─────────────────────────────────────────────
function createReferencesTool(client: LSPClient): ToolSpec {
  return {
    name: "references",
    description:
      "Find all references to a symbol at a specific position. Returns all locations where the symbol is used.",
    category: "readonly",
    paramSchema: POSITION_PARAM_SCHEMA,
    resultSchema: REFERENCES_RESULT_SCHEMA,
    errorGuidance: {
      common: "References lookup failed. Verify the file path and position — use read_file to confirm the symbol location.",
      patterns: LSP_COMMON_PATTERNS,
    },
    handler: async (params) => {
      if (!client.isRunning()) return lspFailure("LSP client not running.");

      const filePath = params["path"] as string;
      const line = params["line"] as number;
      const character = params["character"] as number;

      try {
        const refs = await client.getReferences(filePath, line, character);

        if (refs.length === 0) {
          return {
            success: true,
            output: "No references found at this position.",
            error: null,
            artifacts: [],
          };
        }

        return {
          success: true,
          output: `${refs.length} reference(s):\n${formatLocations(refs)}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        return lspFailure(`References lookup failed: ${extractErrorMessage(err)}`);
      }
    },
  };
}

// ─── symbols (single-client) ─────────────────────────────────
function createSymbolsTool(client: LSPClient): ToolSpec {
  return {
    name: "symbols",
    description:
      "List all symbols (functions, classes, variables, etc.) in a file. Provides a structural overview of the file.",
    category: "readonly",
    paramSchema: PATH_ONLY_PARAM_SCHEMA,
    resultSchema: SYMBOLS_RESULT_SCHEMA,
    errorGuidance: {
      common: "Symbol listing failed. Verify the file path is correct — use find_files to discover available files.",
      patterns: LSP_COMMON_PATTERNS,
    },
    handler: async (params) => {
      if (!client.isRunning()) return lspFailure("LSP client not running.");

      const filePath = params["path"] as string;

      try {
        const symbols = await client.getSymbols(filePath);

        if (symbols.length === 0) {
          return {
            success: true,
            output: `No symbols found in ${filePath}.`,
            error: null,
            artifacts: [],
          };
        }

        return {
          success: true,
          output: `${symbols.length} symbol(s) in ${filePath}:\n${formatSymbols(symbols, filePath)}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        return lspFailure(`Symbol listing failed: ${extractErrorMessage(err)}`);
      }
    },
  };
}

function lspFailure(error: string): {
  readonly success: false;
  readonly output: "";
  readonly error: string;
  readonly artifacts: [];
} {
  return { success: false, output: "", error, artifacts: [] };
}

function formatLocations(locations: ReadonlyArray<{ readonly file: string; readonly line: number; readonly character: number }>): string {
  return locations.map((loc) => `${loc.file}:${loc.line}:${loc.character}`).join("\n");
}

function formatSymbols(
  symbols: ReadonlyArray<{ readonly kind: string; readonly name: string; readonly containerName?: string; readonly line: number }>,
  filePath: string,
): string {
  return symbols.map((symbol) => {
    const container = symbol.containerName ? ` (in ${symbol.containerName})` : "";
    return `${symbol.kind} ${symbol.name}${container} — ${filePath}:${symbol.line}`;
  }).join("\n");
}

// ─── Routing tools (multi-server) ────────────────────────────

function routingHandler(
  resolver: LSPClientResolver,
  filePath: string,
  fn: (client: LSPClient, filePath: string, languageId: string) => Promise<string>,
): Promise<{ success: boolean; output: string; error: string | null; artifacts: string[] }> {
  try {
    const { client, languageId } = resolveOrFail(resolver, filePath);
    return fn(client, filePath, languageId).then(
      (output) => ({ success: true, output, error: null, artifacts: [] }),
      (err) => ({
        success: false,
        output: "",
        error: extractErrorMessage(err),
        artifacts: [],
      }),
    );
  } catch (err) {
    const message = extractErrorMessage(err);
    return Promise.resolve({ success: false, output: "", error: message, artifacts: [] });
  }
}

/** Error guidance for position-based routing LSP tools. */
const LSP_ROUTING_GUIDANCE: ToolErrorGuidance = {
  common:
    "LSP operation failed. Verify the file path is correct and a language server is configured for this file type.",
  patterns: [
    {
      match: "No LSP server",
      hint: "No LSP server available for this file type. Check that a language server is configured for this extension.",
    },
    {
      match: "not running",
      hint: "LSP client is not running. The language server may need to be restarted.",
    },
    {
      match: "timed out",
      hint: "LSP request timed out. The language server may be overloaded — try a smaller file or restart the server.",
    },
  ],
};

/** Error guidance for name-based LSP tools (extends routing guidance with symbol lookup). */
const LSP_NAME_LOOKUP_GUIDANCE: ToolErrorGuidance = {
  common:
    "LSP name-based lookup failed. Verify the file path and symbol name are correct.",
  patterns: [
    {
      match: "No LSP server",
      hint: "No LSP server available for this file type. Check that a language server is configured for this extension.",
    },
    {
      match: "not running",
      hint: "LSP client is not running. The language server may need to be restarted.",
    },
    {
      match: "not found in",
      hint: "Symbol not found in this file. Use the symbols tool to list available symbols, then retry with the correct name.",
    },
    {
      match: "timed out",
      hint: "LSP request timed out. The language server may be overloaded — try a smaller file or restart the server.",
    },
  ],
};

function createRoutingDiagnosticsTool(resolver: LSPClientResolver): ToolSpec {
  return {
    name: "diagnostics",
    description:
      "Get real compiler diagnostics (errors, warnings) for a source file. Preferred over reading a file or running grep when you need to check whether code compiles or has type errors — returns the same errors the compiler would report, not text matches.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to repo root" },
      },
      required: ["path"],
    },
    resultSchema: {
      type: "object",
      properties: { diagnostics: { type: "string" }, count: { type: "number" } },
    },
    errorGuidance: LSP_ROUTING_GUIDANCE,
    handler: async (params) => {
      const filePath = params["path"] as string;
      return routingHandler(resolver, filePath, async (client, fp, langId) => {
        const result = await client.getDiagnostics(fp, langId);
        if (result.diagnostics.length === 0) return `No diagnostics for ${fp}. File is clean.`;
        const lines = result.diagnostics
          .map((d) => `${fp}:${d.line}:${d.character}: [${d.severity}] ${d.message}`)
          .join("\n");
        return `${result.diagnostics.length} diagnostic(s) in ${fp}:\n${lines}`;
      });
    },
  };
}

function createRoutingDefinitionTool(resolver: LSPClientResolver): ToolSpec {
  return {
    name: "definitions",
    description:
      "Jump to the definition of a symbol at a given position. Preferred over grep/search_files when you know a symbol name and need its source — resolves through imports, re-exports, and type aliases accurately.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to repo root" },
        line: { type: "number", description: "Line number (1-based)" },
        character: { type: "number", description: "Column number (1-based)" },
      },
      required: ["path", "line", "character"],
    },
    resultSchema: {
      type: "object",
      properties: { locations: { type: "string" } },
    },
    errorGuidance: LSP_ROUTING_GUIDANCE,
    handler: async (params) => {
      const filePath = params["path"] as string;
      return routingHandler(resolver, filePath, async (client, fp, langId) => {
        const locations = await client.getDefinition(fp, params["line"] as number, params["character"] as number, langId);
        if (locations.length === 0) return "No definition found at this position.";
        return `Definition(s):\n${locations.map((l) => `${l.file}:${l.line}:${l.character}`).join("\n")}`;
      });
    },
  };
}

function createRoutingReferencesTool(resolver: LSPClientResolver): ToolSpec {
  return {
    name: "references",
    description:
      "Find every reference to a symbol across the codebase. Preferred over grep/search_files for finding usages of a function, class, or variable — type-aware, so it won't return false-positive text matches and will catch indirect usages through aliases.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to repo root" },
        line: { type: "number", description: "Line number (1-based)" },
        character: { type: "number", description: "Column number (1-based)" },
      },
      required: ["path", "line", "character"],
    },
    resultSchema: {
      type: "object",
      properties: { references: { type: "string" }, count: { type: "number" } },
    },
    errorGuidance: LSP_ROUTING_GUIDANCE,
    handler: async (params) => {
      const filePath = params["path"] as string;
      return routingHandler(resolver, filePath, async (client, fp, langId) => {
        const refs = await client.getReferences(fp, params["line"] as number, params["character"] as number, langId);
        if (refs.length === 0) return "No references found at this position.";
        return `${refs.length} reference(s):\n${refs.map((l) => `${l.file}:${l.line}:${l.character}`).join("\n")}`;
      });
    },
  };
}

function createRoutingSymbolsTool(resolver: LSPClientResolver): ToolSpec {
  return {
    name: "symbols",
    description:
      "List all symbols (functions, classes, interfaces, variables) in a file with their types and locations. Preferred over reading the whole file when you need a structural overview — faster and gives precise line numbers for each declaration.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to repo root" },
      },
      required: ["path"],
    },
    resultSchema: {
      type: "object",
      properties: { symbols: { type: "string" }, count: { type: "number" } },
    },
    errorGuidance: LSP_ROUTING_GUIDANCE,
    handler: async (params) => {
      const filePath = params["path"] as string;
      return routingHandler(resolver, filePath, async (client, fp, langId) => {
        const symbols = await client.getSymbols(fp, langId);
        if (symbols.length === 0) return `No symbols found in ${fp}.`;
        const lines = symbols
          .map((s) => {
            const container = s.containerName ? ` (in ${s.containerName})` : "";
            return `${s.kind} ${s.name}${container} — ${fp}:${s.line}`;
          })
          .join("\n");
        return `${symbols.length} symbol(s) in ${fp}:\n${lines}`;
      });
    },
  };
}

// ─── Name-based wrappers (composite tools) ──────────────────

function createRoutingDefinitionByNameTool(resolver: LSPClientResolver): ToolSpec {
  return {
    name: "definition_by_name",
    description:
      "Jump to where a symbol is defined, by name. Accepts a symbol name and a file where it appears — resolves through imports and re-exports. Preferred over grep when you need the exact source location of a class, function, or variable.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to repo root where the symbol appears" },
        symbol_name: { type: "string", description: "Name of the symbol to look up (e.g. class, function, variable name)" },
      },
      required: ["path", "symbol_name"],
    },
    resultSchema: {
      type: "object",
      properties: { locations: { type: "string" } },
    },
    errorGuidance: LSP_NAME_LOOKUP_GUIDANCE,
    handler: async (params) => {
      const filePath = params["path"] as string;
      const symbolName = params["symbol_name"] as string;
      return routingHandler(resolver, filePath, async (client, fp, langId) => {
        const symbols = await client.getSymbols(fp, langId);
        const match = symbols.find((s) => s.name === symbolName);
        if (!match) {
          throw new Error(`Symbol "${symbolName}" not found in ${fp}`);
        }
        const locations = await client.getDefinition(fp, match.line, match.character, langId);
        if (locations.length === 0) return `No definition found for "${symbolName}".`;
        return `Definition(s):\n${locations.map((l) => `${l.file}:${l.line}:${l.character}`).join("\n")}`;
      });
    },
  };
}

function createRoutingReferencesByNameTool(resolver: LSPClientResolver): ToolSpec {
  return {
    name: "references_by_name",
    description:
      "Find every reference to a symbol across the codebase, by name. Accepts a symbol name and a file where it is declared — type-aware, catches indirect usages through aliases. Preferred over grep/search_files for finding all callers of a function or all uses of a class.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        path: { type: "string", description: "File path relative to repo root where the symbol is declared" },
        symbol_name: { type: "string", description: "Name of the symbol to find references for" },
      },
      required: ["path", "symbol_name"],
    },
    resultSchema: {
      type: "object",
      properties: { references: { type: "string" }, count: { type: "number" } },
    },
    errorGuidance: LSP_NAME_LOOKUP_GUIDANCE,
    handler: async (params) => {
      const filePath = params["path"] as string;
      const symbolName = params["symbol_name"] as string;
      return routingHandler(resolver, filePath, async (client, fp, langId) => {
        const symbols = await client.getSymbols(fp, langId);
        const match = symbols.find((s) => s.name === symbolName);
        if (!match) {
          throw new Error(`Symbol "${symbolName}" not found in ${fp}`);
        }
        const refs = await client.getReferences(fp, match.line, match.character, langId);
        if (refs.length === 0) return `No references found for "${symbolName}".`;
        return `${refs.length} reference(s):\n${refs.map((l) => `${l.file}:${l.line}:${l.character}`).join("\n")}`;
      });
    },
  };
}
