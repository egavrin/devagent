/**
 * tool_search — Discovers and resolves deferred tools by keyword.
 * Category: state (no side effects beyond resolving tool schemas).
 *
 * When tools are registered as deferred, only stubs (name + description)
 * appear in the prompt. This tool searches deferred tools by keyword
 * and resolves matches so they become available in the next LLM call.
 */

import type { ToolSpec } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";

/**
 * Create the tool_search tool bound to a specific registry.
 * Must be called per-session since it needs the registry instance.
 */
export function createToolSearchTool(registry: ToolRegistry): ToolSpec {
  return {
    name: "tool_search",
    description:
      "Search for and activate deferred tools by keyword. " +
      "Returns matching tool names and descriptions. Matched tools become available for use in the next response. " +
      "Use when you need a tool not currently in your tool list (e.g., git operations, LSP diagnostics, symbols).",
    category: "state",
    paramSchema: {
      type: "object",
      properties: {
        query: {
          type: "string",
          description: "Search keywords (e.g., 'git diff', 'diagnostics', 'symbols')",
        },
        max_results: {
          type: "number",
          description: "Maximum number of tools to return (default: 5)",
        },
      },
      required: ["query"],
    },
    resultSchema: {
      type: "object",
      properties: {
        tools: { type: "array" },
        count: { type: "number" },
      },
    },
    handler: async (params) => {
      const query = params["query"] as string;
      const maxResults = (params["max_results"] as number | undefined) ?? 5;

      if (!query?.trim()) {
        return {
          success: false,
          output: "",
          error: "Query is required",
          artifacts: [],
        };
      }

      const results = registry.search(query, maxResults);

      if (results.length === 0) {
        // If no deferred tools match, check if the tool is already loaded
        const loaded = registry.getLoaded();
        const directMatch = loaded.find(
          (t) => t.name.toLowerCase().includes(query.toLowerCase()),
        );
        if (directMatch) {
          return {
            success: true,
            output: `Tool "${directMatch.name}" is already available. Use it directly.`,
            error: null,
            artifacts: [],
          };
        }
        return {
          success: true,
          output: "No matching tools found.",
          error: null,
          artifacts: [],
        };
      }

      const lines = results.map((t) => `- ${t.name} [${t.category}]: ${t.description}`);
      return {
        success: true,
        output: `Resolved ${results.length} tool(s) — now available for use:\n${lines.join("\n")}`,
        error: null,
        artifacts: [],
      };
    },
  };
}
