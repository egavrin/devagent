/**
 * Factory: creates the `execute_tool_script` ToolSpec.
 */

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
      description:
        "TypeScript code to run in a restricted child process. " +
        "Use this for narrowed audits that need 3+ readonly calls, especially known-path multi-file read_file batches. " +
        "Use await tools.read_file({path}), tools.search_files({...}), tools.find_files({...}), tools.git_status({}), or tools.git_diff({...}). " +
        "Call print(...) with only the final synthesized answer; raw intermediate tool outputs are not returned to the model.",
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
    description:
      "Execute a TypeScript program that calls multiple readonly tools locally and returns only final stdout. " +
      "Default to this as the first inspection tool for narrowed tasks needing 3+ readonly calls, including known-path multi-file audits, grouped read_file checks, implementation/schema/test comparisons, prompt-consistency checks, and security-leakage verification. " +
      "Use direct readonly tools instead for one-off lookups, broad unknown-scope reconnaissance, or debugging a failed script. " +
      "Only readonly tools are exposed through the tools object; imports, shell, filesystem, network, process, and recursive execute_tool_script calls are unavailable.",
    category: "readonly",
    errorGuidance: {
      common:
        "Fix the TypeScript script or use direct readonly tool calls. Print only synthesized findings, counts, paths, or summaries.",
      patterns: [
        { match: "script parameter", hint: "Pass a non-empty `script` string. The old `steps` array DSL is no longer supported." },
        { match: "not available", hint: "Only readonly tools are exposed through `tools`; use direct tools for workflow, external, or mutating operations." },
        { match: "timed out", hint: "Narrow the script, reduce tool calls, or switch to direct tool calls for debugging." },
        { match: "stdout exceeded", hint: "Print a shorter synthesized summary instead of raw tool output." },
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
