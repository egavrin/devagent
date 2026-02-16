/**
 * LSP-powered code analysis tools.
 * These tools use a running LSP client for real compiler intelligence.
 * All are read-only (category: "readonly").
 */

import type { ToolSpec } from "@devagent/core";
import type { LSPClient } from "./client.js";

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

// ─── diagnostics ────────────────────────────────────────────

function createDiagnosticsTool(client: LSPClient): ToolSpec {
  return {
    name: "diagnostics",
    description:
      "Get compiler diagnostics (errors, warnings) for a file. Uses the language server for real compiler analysis.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "File path relative to repo root",
        },
      },
      required: ["path"],
    },
    resultSchema: {
      type: "object",
      properties: {
        diagnostics: { type: "string" },
        count: { type: "number" },
      },
    },
    handler: async (params) => {
      if (!client.isRunning()) {
        return {
          success: false,
          output: "",
          error: "LSP client not running. Language server may not be available.",
          artifacts: [],
        };
      }

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
          .map(
            (d) =>
              `${filePath}:${d.line}:${d.character}: [${d.severity}] ${d.message}`,
          )
          .join("\n");

        return {
          success: true,
          output: `${result.diagnostics.length} diagnostic(s) in ${filePath}:\n${output}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          success: false,
          output: "",
          error: `Diagnostics failed: ${message}`,
          artifacts: [],
        };
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
    paramSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "File path relative to repo root",
        },
        line: {
          type: "number",
          description: "Line number (1-based)",
        },
        character: {
          type: "number",
          description: "Column number (1-based)",
        },
      },
      required: ["path", "line", "character"],
    },
    resultSchema: {
      type: "object",
      properties: {
        locations: { type: "string" },
      },
    },
    handler: async (params) => {
      if (!client.isRunning()) {
        return {
          success: false,
          output: "",
          error: "LSP client not running.",
          artifacts: [],
        };
      }

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

        const output = locations
          .map((loc) => `${loc.file}:${loc.line}:${loc.character}`)
          .join("\n");

        return {
          success: true,
          output: `Definition(s):\n${output}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          success: false,
          output: "",
          error: `Definition lookup failed: ${message}`,
          artifacts: [],
        };
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
    paramSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "File path relative to repo root",
        },
        line: {
          type: "number",
          description: "Line number (1-based)",
        },
        character: {
          type: "number",
          description: "Column number (1-based)",
        },
      },
      required: ["path", "line", "character"],
    },
    resultSchema: {
      type: "object",
      properties: {
        references: { type: "string" },
        count: { type: "number" },
      },
    },
    handler: async (params) => {
      if (!client.isRunning()) {
        return {
          success: false,
          output: "",
          error: "LSP client not running.",
          artifacts: [],
        };
      }

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

        const output = refs
          .map((loc) => `${loc.file}:${loc.line}:${loc.character}`)
          .join("\n");

        return {
          success: true,
          output: `${refs.length} reference(s):\n${output}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          success: false,
          output: "",
          error: `References lookup failed: ${message}`,
          artifacts: [],
        };
      }
    },
  };
}

// ─── symbols ────────────────────────────────────────────────

function createSymbolsTool(client: LSPClient): ToolSpec {
  return {
    name: "symbols",
    description:
      "List all symbols (functions, classes, variables, etc.) in a file. Provides a structural overview of the file.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        path: {
          type: "string",
          description: "File path relative to repo root",
        },
      },
      required: ["path"],
    },
    resultSchema: {
      type: "object",
      properties: {
        symbols: { type: "string" },
        count: { type: "number" },
      },
    },
    handler: async (params) => {
      if (!client.isRunning()) {
        return {
          success: false,
          output: "",
          error: "LSP client not running.",
          artifacts: [],
        };
      }

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

        const output = symbols
          .map((s) => {
            const container = s.containerName ? ` (in ${s.containerName})` : "";
            return `${s.kind} ${s.name}${container} — ${filePath}:${s.line}`;
          })
          .join("\n");

        return {
          success: true,
          output: `${symbols.length} symbol(s) in ${filePath}:\n${output}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        return {
          success: false,
          output: "",
          error: `Symbol listing failed: ${message}`,
          artifacts: [],
        };
      }
    },
  };
}
