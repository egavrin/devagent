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
import { ToolScriptEngine, parseToolScriptStepsArg } from "./tool-script.js";

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
      "Only readonly tools are allowed (find_files, read_file, search_files, git_status, etc.). " +
      "Tool names must be canonical (for example read_file, not functions.read_file).",
    category: "readonly",
    errorGuidance: {
      common: "Break the script into individual tool calls. Check that tool names are bare canonical names (e.g. read_file, not functions.read_file) and args are valid JSON strings.",
      patterns: [
        { match: "forward reference", hint: "Step references must refer to earlier steps. Reorder your steps so referenced steps come first." },
        { match: "Unknown tool", hint: "Check the tool name — only readonly tools are allowed (read_file, search_files, find_files, git_status, git_diff)." },
        { match: "steps failed", hint: "Some steps failed. Check the [FAILED] steps in the output. Consider using individual tool calls for the failed operations." },
        { match: "Invalid steps", hint: "The steps parameter must be a JSON array of {id, tool, args} objects. Each args value must be a JSON string, not an object." },
      ],
    },
    paramSchema: {
      type: "object",
      properties: {
        steps: {
          type: "array",
          description:
            "Array of tool steps to execute sequentially. " +
            "Reference previous step outputs with $stepId (full output) or $stepId.lines[N] (specific line, 0-indexed). " +
            "Use canonical tool names only (no functions./function./tools. prefixes).",
          items: {
            type: "object",
            properties: {
              id: {
                type: "string",
                description: "Unique identifier for this step, used for inter-step references.",
              },
              tool: {
                type: "string",
                description: "Canonical tool name (e.g. read_file, find_files, search_files).",
              },
              args: {
                type: "string",
                description:
                  'JSON-encoded tool parameters, e.g. {"path":"src/foo.ts"} or {} for no arguments.',
              },
            },
            required: ["id", "tool", "args"],
            additionalProperties: false,
          },
        },
      },
      required: ["steps"],
    },
    resultSchema: { type: "object" },
    handler: async (
      params: Record<string, unknown>,
      toolContext: ToolContext,
    ) => {
      const steps = parseToolScriptStepsArg(params["steps"]);
      if (!steps) {
        return {
          success: false,
          output: "",
          error: "Invalid steps parameter: must be an array of {id, tool, args} objects (or a JSON string encoding one)",
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

      const allSucceeded = succeededCount === result.steps.length;
      return {
        success: allSucceeded,
        output: sections.join("\n\n"),
        error: allSucceeded
          ? null
          : succeededCount === 0
            ? "All steps failed"
            : `${result.steps.length - succeededCount}/${result.steps.length} steps failed`,
        artifacts: [],
      };
    },
  };
}
