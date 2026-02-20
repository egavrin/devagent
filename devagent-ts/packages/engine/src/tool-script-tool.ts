/**
 * Factory: creates the `execute_tool_script` ToolSpec.
 *
 * This tool allows the LLM to batch multiple readonly tool calls into a single
 * round-trip. Steps run sequentially with inter-step references, returning
 * all results aggregated. Only readonly tools are allowed.
 */

import type { ToolSpec, ToolContext } from "@devagent/core";
import type { EventBus } from "@devagent/core";
import type { ToolRegistry } from "@devagent/tools";
import { ToolScriptEngine } from "./tool-script.js";
import type { ToolScriptStep } from "./tool-script.js";

export interface ToolScriptToolContext {
  readonly registry: ToolRegistry;
  readonly bus: EventBus;
}

export function createToolScriptTool(ctx: ToolScriptToolContext): ToolSpec {
  return {
    name: "execute_tool_script",
    description:
      "Execute multiple readonly tools in a single batch. " +
      "Steps run sequentially; reference previous step outputs with $stepId " +
      "(full output) or $stepId.lines[N] (specific line, 0-indexed). " +
      "Only readonly tools are allowed (find_files, read_file, search_files, git_status, etc.).",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        steps: {
          type: "string",
          description:
            'JSON array of steps: [{"id":"find","tool":"find_files","args":{"pattern":"**/*.ts"}}, ' +
            '{"id":"read1","tool":"read_file","args":{"path":"$find.lines[0]"}}]',
        },
      },
      required: ["steps"],
    },
    resultSchema: { type: "object" },
    handler: async (
      params: Record<string, unknown>,
      toolContext: ToolContext,
    ) => {
      const stepsJson = params["steps"] as string;

      // Parse JSON
      let steps: ToolScriptStep[];
      try {
        steps = JSON.parse(stepsJson) as ToolScriptStep[];
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        return {
          success: false,
          output: "",
          error: `Invalid JSON in steps parameter: ${msg}`,
          artifacts: [],
        };
      }

      // Validate it's an array
      if (!Array.isArray(steps)) {
        return {
          success: false,
          output: "",
          error: "Steps must be a JSON array",
          artifacts: [],
        };
      }

      // Execute via engine
      const engine = new ToolScriptEngine({
        registry: ctx.registry,
        context: toolContext,
        bus: ctx.bus,
      });

      const result = await engine.execute({ steps });

      // Check if validation failed
      if (
        result.steps.length === 1 &&
        result.steps[0]!.id === "__validation__"
      ) {
        return {
          success: false,
          output: "",
          error: result.steps[0]!.error ?? "Validation failed",
          artifacts: [],
        };
      }

      // Format output
      const sections: string[] = [];
      let succeededCount = 0;

      for (const step of result.steps) {
        if (step.success) {
          succeededCount++;
          sections.push(
            `=== Step ${step.id} (${step.tool}) [${step.durationMs}ms] ===\n${step.output}`,
          );
        } else {
          sections.push(
            `=== Step ${step.id} (${step.tool}) [FAILED] ===\nError: ${step.error}`,
          );
        }
      }

      const summary = `\n[Script completed: ${succeededCount}/${result.steps.length} steps succeeded in ${result.totalDurationMs}ms]`;
      if (result.truncated) {
        sections.push("[Output truncated — limit exceeded]");
      }
      sections.push(summary);

      return {
        success: succeededCount > 0,
        output: sections.join("\n\n"),
        error: succeededCount === 0 ? "All steps failed" : null,
        artifacts: [],
      };
    },
  };
}
