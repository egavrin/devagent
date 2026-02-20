/**
 * LLM-callable memory tools — explicit hot-path for cross-session learning.
 * The agent can autonomously decide to remember lessons or recall past knowledge.
 *
 * Two tools:
 * - memory_store: persist a lesson, pattern, or decision
 * - memory_recall: search past memories by query/category
 */

import type { ToolSpec, MemoryCategory, MemoryStore } from "@devagent/core";

export function createMemoryTools(memoryStore: MemoryStore): ToolSpec[] {
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
          minRelevance: 0.3,
          limit: 10,
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
  ];
}
