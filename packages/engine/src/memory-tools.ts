/**
 * LLM-callable memory tools — explicit hot-path for cross-session learning.
 * The agent can autonomously decide to remember lessons, recall past knowledge,
 * list all memories for auditing, or delete outdated entries.
 *
 * Four tools:
 * - memory_store: persist a lesson, pattern, or decision
 * - memory_recall: search past memories by query/category
 * - memory_list: list all memories with IDs and metadata
 * - memory_delete: delete a specific memory by ID
 */

import type { ToolSpec, MemoryCategory, MemoryStore } from "@devagent/core";

export interface MemoryToolOptions {
  /** Minimum relevance for recall search. Default: 0.3. */
  readonly recallMinRelevance?: number;
  /** Maximum results for recall search. Default: 10. */
  readonly recallLimit?: number;
}

export function createMemoryTools(
  memoryStore: MemoryStore,
  options?: MemoryToolOptions,
): ToolSpec[] {
  const recallMinRelevance = options?.recallMinRelevance ?? 0.3;
  const recallLimit = options?.recallLimit ?? 10;

  return [
    {
      name: "memory_store",
      description:
        "Store a lesson, pattern, or decision for future sessions. " +
        "Use this when you learn something important that should persist " +
        "(e.g., project conventions, user preferences, past mistakes to avoid).",
      category: "workflow",
      paramSchema: {
        type: "object",
        properties: {
          category: {
            type: "string",
            enum: [
              "pattern",
              "decision",
              "mistake",
              "preference",
              "context",
            ],
            description:
              "Memory category: pattern (code/workflow patterns), " +
              "decision (architectural choices), mistake (things that failed), " +
              "preference (user preferences), context (project context)",
          },
          key: {
            type: "string",
            description:
              "Short unique identifier for this memory (e.g., 'test-framework', 'import-style')",
          },
          content: {
            type: "string",
            description: "What was learned — concise and actionable",
          },
        },
        required: ["category", "key", "content"],
      },
      resultSchema: { type: "object" },
      handler: async (params, _context) => {
        const category = params["category"] as MemoryCategory;
        const key = params["key"] as string;
        const content = params["content"] as string;

        const id = memoryStore.store(category, key, content);
        return {
          success: true,
          output: `Memory stored: [${category}] ${key} (id: ${id})`,
          error: null,
          artifacts: [],
        };
      },
    },
    {
      name: "memory_recall",
      description:
        "Search past lessons, patterns, and decisions from previous sessions. " +
        "Use this to check if there are relevant learnings before starting a task.",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: {
          query: {
            type: "string",
            description: "Search query (keyword match against key and content)",
          },
          category: {
            type: "string",
            enum: [
              "pattern",
              "decision",
              "mistake",
              "preference",
              "context",
            ],
            description: "Filter by category (optional)",
          },
        },
        required: [],
      },
      resultSchema: { type: "object" },
      handler: async (params, _context) => {
        const query = params["query"] as string | undefined;
        const category = params["category"] as MemoryCategory | undefined;

        const results = memoryStore.search({
          query,
          category,
          minRelevance: recallMinRelevance,
          limit: recallLimit,
        });

        if (results.length === 0) {
          return {
            success: true,
            output: "No relevant memories found.",
            error: null,
            artifacts: [],
          };
        }

        const formatted = results
          .map(
            (m) =>
              `[${m.category}] ${m.key}: ${m.content} (relevance: ${m.relevance.toFixed(2)})`,
          )
          .join("\n");
        return {
          success: true,
          output: formatted,
          error: null,
          artifacts: [],
        };
      },
    },
    {
      name: "memory_list",
      description:
        "List all memories with their IDs, categories, and relevance scores. " +
        "Use this to audit stored memories, find outdated ones to delete, " +
        "or understand what knowledge persists across sessions.",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: {
          category: {
            type: "string",
            enum: [
              "pattern",
              "decision",
              "mistake",
              "preference",
              "context",
            ],
            description: "Filter by category (optional)",
          },
          limit: {
            type: "number",
            description: "Maximum number of results (default: 50)",
          },
        },
        required: [],
      },
      resultSchema: { type: "object" },
      handler: async (params, _context) => {
        const category = params["category"] as MemoryCategory | undefined;
        const limit = (params["limit"] as number | undefined) ?? 50;

        const results = memoryStore.search({ category, limit });

        if (results.length === 0) {
          return {
            success: true,
            output: "No memories stored.",
            error: null,
            artifacts: [],
          };
        }

        const summary = memoryStore.summary();
        const totalCount = Object.values(summary).reduce((a, b) => a + b, 0);
        const breakdown = Object.entries(summary)
          .filter(([, v]) => v > 0)
          .map(([k, v]) => `${k}: ${v}`)
          .join(", ");
        const header = `Total: ${totalCount} memories (${breakdown})`;

        const formatted = results
          .map(
            (m) =>
              `[${m.category}] ${m.key} (id: ${m.id.substring(0, 8)}…, relevance: ${m.relevance.toFixed(2)}, accessed: ${m.accessCount}x): ${m.content}`,
          )
          .join("\n");

        return {
          success: true,
          output: `${header}\n\n${formatted}`,
          error: null,
          artifacts: [],
        };
      },
    },
    {
      name: "memory_delete",
      description:
        "Delete a specific memory by its ID. Use this to remove outdated, " +
        "incorrect, or superseded memories. Use memory_list first to find IDs.",
      category: "workflow",
      paramSchema: {
        type: "object",
        properties: {
          id: {
            type: "string",
            description: "The memory ID to delete (use memory_list to find IDs)",
          },
        },
        required: ["id"],
      },
      resultSchema: { type: "object" },
      handler: async (params, _context) => {
        const id = params["id"] as string;
        const deleted = memoryStore.delete(id);
        return {
          success: deleted,
          output: deleted
            ? `Memory ${id} deleted.`
            : `Memory ${id} not found.`,
          error: deleted ? null : `No memory with ID ${id}`,
          artifacts: [],
        };
      },
    },
  ];
}
