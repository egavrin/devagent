/**
 * Factory: creates the `execute_tool_script` ToolSpec.
 */

import {
  formatToolScriptDescription,
  formatToolScriptSchemaDescription,
} from "./readonly-batching-guidance.js";
import { ToolScriptEngine } from "./tool-script.js";
import type { EventBus, ToolContext, ToolSpec } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";

export interface ToolScriptToolContext {
  readonly registry: ToolRegistry;
  readonly bus: EventBus;
}

const TOOL_SCRIPT_PARAM_SCHEMA = {
  type: "object",
  properties: {
    script: {
      type: "string",
      description: formatToolScriptSchemaDescription(),
    },
    timeout_ms: {
      type: "number",
      description: "Optional script timeout in milliseconds. Defaults to 30000.",
    },
    max_output_chars: {
      type: "number",
      description: "Optional maximum final stdout characters. Defaults to 16384.",
    },
  },
  required: ["script"],
  additionalProperties: false,
};

export function createToolScriptTool(ctx: ToolScriptToolContext): ToolSpec {
  return {
    name: "execute_tool_script",
    description: formatToolScriptDescription(),
    category: "readonly",
    errorGuidance: {
      common:
        "Fix the TypeScript script or use direct readonly tool calls to debug. Check result.success, inspect result.output, call print(...), and print only synthesized findings, counts, paths, or summaries.",
      patterns: [
        { match: "script parameter", hint: "Pass a non-empty `script` string. The old `steps` array DSL is no longer supported." },
        { match: "not available", hint: "Only readonly tools are exposed through `tools`; use direct tools for workflow, external, or mutating operations." },
        { match: "timed out", hint: "Narrow the script, reduce tool calls, or switch to direct readonly calls for debugging instead of retrying the same script." },
        { match: "stdout exceeded", hint: "Print a shorter synthesized summary instead of raw tool output, raw file dumps, or broad regex line-hit dumps." },
        { match: "result.content", hint: "ToolResult uses `output`, not `content`; inspect `result.output`." },
        { match: "maximum of", hint: "Reduce the script or split it into smaller batches; the inner tool-call cap was reached." },
        { match: "No output", hint: "Call `print(...)` with the synthesized final answer." },
      ],
    },
    paramSchema: TOOL_SCRIPT_PARAM_SCHEMA,
    resultSchema: { type: "object" },
    handler: async (params, toolContext) => runToolScriptTool(ctx, params, toolContext),
  };
}

async function runToolScriptTool(
  ctx: ToolScriptToolContext,
  params: Record<string, unknown>,
  toolContext: ToolContext,
) {
  if (params["steps"] !== undefined) {
    return {
      success: false,
      output: "",
      error: "Invalid execute_tool_script input: `steps` is no longer supported. Pass a TypeScript `script` string instead.",
      artifacts: [],
    };
  }
  if (typeof params["script"] !== "string") {
    return {
      success: false,
      output: "",
      error: "Invalid script parameter: execute_tool_script requires a non-empty TypeScript script string.",
      artifacts: [],
    };
  }

  return new ToolScriptEngine({
    registry: ctx.registry,
    context: toolContext,
    bus: ctx.bus,
  }).execute({
    script: params["script"],
    timeoutMs: numberParam(params["timeout_ms"]),
    maxOutputChars: numberParam(params["max_output_chars"]),
  });
}

function numberParam(value: unknown): number | undefined {
  return typeof value === "number" && Number.isFinite(value) ? value : undefined;
}
