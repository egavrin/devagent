/**
 * Programmatic readonly tool script runner.
 *
 * Executes generated TypeScript in a child process with a restricted vm
 * context. Scripts can call readonly tools through `tools.*`, process results
 * locally, and return only their final stdout to the main model context.
 */

import { type ChildProcess, spawn } from "node:child_process";
import { transpileModule, DiagnosticCategory, ModuleKind, ScriptTarget } from "typescript";

import { syncLSPAfterToolResult } from "./task-loop-lsp-sync.js";
import type { AgentType, EventBus, ToolContext, ToolResult } from "../core/index.js";
import { extractErrorMessage, extractToolFileChangePreviewSummary } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";

export interface ToolScript {
  readonly script: string;
  readonly timeoutMs?: number;
  readonly maxOutputChars?: number;
}

export interface ToolScriptTelemetry {
  readonly toolCallCount: number;
  readonly innerOutputChars: number;
  readonly finalOutputChars: number;
  readonly durationMs: number;
  readonly timedOut: boolean;
  readonly truncated: boolean;
}

export interface ToolScriptResult extends ToolResult {
  readonly totalDurationMs: number;
  readonly timedOut: boolean;
  readonly truncated: boolean;
  readonly toolCallCount: number;
  readonly metadata: {
    readonly toolScript: ToolScriptTelemetry;
  };
}

export interface ToolScriptEngineOptions {
  readonly registry: ToolRegistry;
  readonly context: ToolContext;
  readonly bus: EventBus;
  readonly defaultTimeoutMs?: number;
  readonly defaultMaxOutputChars?: number;
  readonly maxToolCalls?: number;
}

type ParentToChildMessage =
  | {
    readonly type: "run";
    readonly code: string;
    readonly allowedToolNames: readonly string[];
    readonly timeoutMs: number;
    readonly maxOutputChars: number;
    readonly maxToolCalls: number;
  }
  | {
    readonly type: "tool_result";
    readonly id: string;
    readonly result: ToolResult;
  };

type ChildToParentMessage =
  | {
    readonly type: "tool_call";
    readonly id: string;
    readonly name: string;
    readonly args: Record<string, unknown>;
  }
  | {
    readonly type: "done";
    readonly output: string;
    readonly toolCallCount: number;
  }
  | {
    readonly type: "error";
    readonly error: string;
    readonly output: string;
    readonly toolCallCount: number;
  };

interface RunChildOptions {
  readonly code: string;
  readonly timeoutMs: number;
  readonly maxOutputChars: number;
  readonly startedAt: number;
}

interface RunChildState {
  stderr: string;
  parentToolCallCount: number;
  innerOutputChars: number;
}

interface ChildMessageContext {
  readonly child: ChildProcess;
  readonly options: Pick<RunChildOptions, "startedAt" | "maxOutputChars">;
  readonly settle: (result: ToolScriptResult) => void;
  readonly nextParentToolCallCount: () => number;
  readonly addInnerOutputChars: (chars: number) => void;
  readonly getInnerOutputCharsTotal: () => number;
}

interface FailureResultOptions {
  readonly toolCallCount?: number;
  readonly output?: string;
  readonly durationMs?: number;
  readonly timedOut?: boolean;
  readonly truncated?: boolean;
  readonly innerOutputChars?: number;
}

const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_MAX_OUTPUT_CHARS = 16 * 1024;
const DEFAULT_MAX_TOOL_CALLS = 20;
const BLOCKED_TOOLS = new Set(["execute_tool_script"]);
const IMPORT_EXPORT_PATTERN = /(^|\n)\s*(import|export)\b|\bimport\s*\(/;
let fallbackScriptRunCounter = 0;

export class ToolScriptEngine {
  private readonly registry: ToolRegistry;
  private readonly context: ToolContext;
  private readonly bus: EventBus;
  private readonly defaultTimeoutMs: number;
  private readonly defaultMaxOutputChars: number;
  private readonly maxToolCalls: number;
  private readonly scriptCallIdPrefix: string;

  constructor(options: ToolScriptEngineOptions) {
    this.registry = options.registry;
    this.context = options.context;
    this.bus = options.bus;
    this.defaultTimeoutMs = options.defaultTimeoutMs ?? DEFAULT_TIMEOUT_MS;
    this.defaultMaxOutputChars = options.defaultMaxOutputChars ?? DEFAULT_MAX_OUTPUT_CHARS;
    this.maxToolCalls = options.maxToolCalls ?? DEFAULT_MAX_TOOL_CALLS;
    this.scriptCallIdPrefix = options.context.callId ?? `script_run_${++fallbackScriptRunCounter}`;
  }

  async execute(script: ToolScript): Promise<ToolScriptResult> {
    const startedAt = Date.now();
    const validationError = validateScript(script.script);
    if (validationError) return failureResult(validationError, startedAt);

    const compiled = transpileScript(script.script);
    if (compiled.error) return failureResult(compiled.error, startedAt);

    return this.runChild({
      code: compiled.code,
      timeoutMs: normalizePositiveInt(script.timeoutMs, this.defaultTimeoutMs),
      maxOutputChars: normalizePositiveInt(script.maxOutputChars, this.defaultMaxOutputChars),
      startedAt,
    });
  }

  private runChild(options: RunChildOptions): Promise<ToolScriptResult> {
    const allowedToolNames = this.getAllowedToolNames();
    const child = spawnToolScriptChild();
    const state: RunChildState = { stderr: "", parentToolCallCount: 0, innerOutputChars: 0 };
    collectChildOutput(child, state);
    return new Promise<ToolScriptResult>((resolve) => {
      let settled = false;
      const settle = (result: ToolScriptResult) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        child.kill();
        resolve(result);
      };
      const timer = setTimeout(() => {
        const durationMs = Date.now() - options.startedAt;
        settle(failureResult(`Script timed out after ${options.timeoutMs}ms`, options.startedAt, {
          durationMs,
          innerOutputChars: state.innerOutputChars,
          timedOut: true,
          toolCallCount: state.parentToolCallCount,
        }));
      }, options.timeoutMs);

      child.on("message", (raw) => {
        void this.handleChildMessage(raw, {
          child,
          options,
          settle,
          nextParentToolCallCount: () => ++state.parentToolCallCount,
          addInnerOutputChars: (chars) => {
            state.innerOutputChars += chars;
          },
          getInnerOutputCharsTotal: () => state.innerOutputChars,
        });
      });
      child.on("error", (error) => {
        settle(failureResult(extractErrorMessage(error), options.startedAt));
      });
      child.on("exit", (code, signal) => {
        if (settled) return;
        const detail = state.stderr.trim() || `Child process exited unexpectedly (${signal ?? code ?? "unknown"})`;
        settle(failureResult(detail, options.startedAt));
      });

      sendChild(child, {
        type: "run",
        code: options.code,
        allowedToolNames,
        timeoutMs: options.timeoutMs,
        maxOutputChars: options.maxOutputChars,
        maxToolCalls: this.maxToolCalls,
      });
    });
  }

  private async handleChildMessage(
    raw: unknown,
    context: ChildMessageContext,
  ): Promise<void> {
    if (!isChildMessage(raw)) {
      context.settle(failureResult("Script runner protocol violation: invalid child message", context.options.startedAt));
      return;
    }
    if (raw.type === "done") {
      const totalDurationMs = Date.now() - context.options.startedAt;
      if (raw.output.trim().length === 0) {
        context.settle(failureResult("No output printed: call print(...) with the synthesized final answer.", context.options.startedAt, {
          durationMs: totalDurationMs,
          innerOutputChars: context.getInnerOutputCharsTotal(),
          toolCallCount: raw.toolCallCount,
        }));
        return;
      }
      context.settle(successResult(raw.output, raw.toolCallCount, totalDurationMs, context.getInnerOutputCharsTotal()));
      return;
    }
    if (raw.type === "error") {
      const totalDurationMs = Date.now() - context.options.startedAt;
      context.settle({
        success: false,
        output: raw.output,
        error: raw.error,
        artifacts: [],
        totalDurationMs,
        timedOut: false,
        truncated: raw.output.length >= context.options.maxOutputChars,
        toolCallCount: raw.toolCallCount,
        metadata: {
          toolScript: {
            toolCallCount: raw.toolCallCount,
            innerOutputChars: context.getInnerOutputCharsTotal(),
            finalOutputChars: raw.output.length,
            durationMs: totalDurationMs,
            timedOut: false,
            truncated: raw.output.length >= context.options.maxOutputChars,
          },
        },
      });
      return;
    }

    const toolCallNumber = context.nextParentToolCallCount();
    const result = await this.executeNestedTool(raw, toolCallNumber);
    context.addInnerOutputChars(getInnerOutputChars(result));
    sendChild(context.child, { type: "tool_result", id: raw.id, result });
  }

  private async executeNestedTool(
    call: Extract<ChildToParentMessage, { readonly type: "tool_call" }>,
    toolCallNumber: number,
  ): Promise<ToolResult> {
    const tool = this.registry.get(call.name);
    const callId = `${this.scriptCallIdPrefix}_script_${call.name}_${toolCallNumber}`;
    this.bus.emit("tool:before", {
      name: call.name,
      params: call.args,
      callId,
      ...this.getAgentEventFields(),
    });

    const startedAt = Date.now();
    let result: ToolResult;
    try {
      result = await tool.handler(call.args, {
        ...this.context,
        callId,
      });
      await syncLSPAfterToolResult(call.name, call.args, result, this.context.lspSync);
    } catch (error) {
      result = {
        success: false,
        output: "",
        error: extractErrorMessage(error),
        artifacts: [],
      };
    }
    const durationMs = Date.now() - startedAt;
    const fileEditSummary = extractToolFileChangePreviewSummary(result.metadata);
    this.bus.emit("tool:after", {
      name: call.name,
      result,
      fileEdits: fileEditSummary.fileEdits,
      fileEditHiddenCount: fileEditSummary.hiddenFileCount > 0 ? fileEditSummary.hiddenFileCount : undefined,
      callId,
      durationMs,
      ...this.getAgentEventFields(),
    });
    return result;
  }

  private getAgentEventFields(): {
    readonly agentId?: string;
    readonly parentAgentId?: string | null;
    readonly depth?: number;
    readonly agentType?: AgentType;
  } {
    return {
      ...(this.context.agentId ? { agentId: this.context.agentId } : {}),
      ...(this.context.parentAgentId !== undefined ? { parentAgentId: this.context.parentAgentId } : {}),
      ...(this.context.depth !== undefined ? { depth: this.context.depth } : {}),
      ...(this.context.agentType ? { agentType: this.context.agentType } : {}),
    };
  }

  private getAllowedToolNames(): string[] {
    return this.registry
      .getReadOnly()
      .map((tool) => tool.name)
      .filter((name) => !BLOCKED_TOOLS.has(name))
      .sort();
  }
}

function validateScript(script: string): string | null {
  if (typeof script !== "string" || script.trim().length === 0) {
    return "Invalid script parameter: execute_tool_script requires a non-empty TypeScript script string.";
  }
  if (IMPORT_EXPORT_PATTERN.test(script)) {
    return "Programmatic tool scripts cannot use import/export or dynamic import; use only the provided readonly tools object.";
  }
  return null;
}

function transpileScript(script: string): { readonly code: string; readonly error: null } | { readonly code: ""; readonly error: string } {
  const output = transpileModule(script, {
    compilerOptions: {
      module: ModuleKind.None,
      target: ScriptTarget.ES2022,
      noEmitOnError: true,
    },
    reportDiagnostics: true,
  });
  const error = output.diagnostics?.find((diagnostic) => diagnostic.category === DiagnosticCategory.Error);
  if (error) {
    return { code: "", error: `TypeScript transpile error: ${String(error.messageText)}` };
  }
  return { code: output.outputText, error: null };
}

function normalizePositiveInt(value: number | undefined, fallback: number): number {
  if (!Number.isFinite(value) || value === undefined) return fallback;
  return Math.max(1, Math.floor(value));
}

function successResult(output: string, toolCallCount: number, totalDurationMs: number, innerOutputChars = 0): ToolScriptResult {
  return {
    success: true,
    output,
    error: null,
    artifacts: [],
    totalDurationMs,
    timedOut: false,
    truncated: false,
    toolCallCount,
    metadata: {
      toolScript: {
        toolCallCount,
        innerOutputChars,
        finalOutputChars: output.length,
        durationMs: totalDurationMs,
        timedOut: false,
        truncated: false,
      },
    },
  };
}

function failureResult(
  error: string,
  startedAt: number,
  options: FailureResultOptions = {},
): ToolScriptResult {
  const output = options.output ?? "";
  const durationMs = options.durationMs ?? Date.now() - startedAt;
  const timedOut = options.timedOut ?? false;
  const truncated = options.truncated ?? false;
  const toolCallCount = options.toolCallCount ?? 0;
  const innerOutputChars = options.innerOutputChars ?? 0;
  return {
    success: false,
    output,
    error,
    artifacts: [],
    totalDurationMs: durationMs,
    timedOut,
    truncated,
    toolCallCount,
    metadata: {
      toolScript: {
        toolCallCount,
        innerOutputChars,
        finalOutputChars: output.length,
        durationMs,
        timedOut,
        truncated,
      },
    },
  };
}

function getInnerOutputChars(result: ToolResult): number {
  return result.output.length;
}

function spawnToolScriptChild(): ChildProcess {
  return spawn(process.execPath, ["--input-type=module", "--eval", CHILD_PROCESS_SOURCE], {
    stdio: ["ignore", "pipe", "pipe", "ipc"],
    env: {},
  });
}

function collectChildOutput(child: ChildProcess, state: RunChildState): void {
  child.stderr?.on("data", (chunk) => {
    state.stderr += String(chunk);
  });
  child.stdout?.on("data", (chunk) => {
    state.stderr += String(chunk);
  });
}

function sendChild(child: ChildProcess, message: ParentToChildMessage): void {
  if (!child.connected) return;
  child.send?.(message);
}

function isChildMessage(raw: unknown): raw is ChildToParentMessage {
  if (!raw || typeof raw !== "object") return false;
  const msg = raw as Record<string, unknown>;
  if (msg["type"] === "done" || msg["type"] === "error") return isChildCompletionMessage(msg);
  return isChildToolCallMessage(msg);
}

function isChildCompletionMessage(msg: Record<string, unknown>): boolean {
  return typeof msg["output"] === "string" && typeof msg["toolCallCount"] === "number";
}

function isChildToolCallMessage(msg: Record<string, unknown>): boolean {
  return msg["type"] === "tool_call"
    && typeof msg["id"] === "string"
    && typeof msg["name"] === "string"
    && Boolean(msg["args"])
    && typeof msg["args"] === "object"
    && !Array.isArray(msg["args"]);
}

const CHILD_PROCESS_SOURCE = String.raw`
import vm from "node:vm";

let stdout = "";
let toolCallCount = 0;
let activeConfig = null;
const pending = new Map();

process.on("uncaughtException", (error) => {
  sendError(error instanceof Error ? error.message : String(error));
});
process.on("unhandledRejection", (error) => {
  sendError(error instanceof Error ? error.message : String(error));
});

process.on("message", (message) => {
  if (!message || typeof message !== "object") return;
  if (message.type === "run") {
    void run(message);
    return;
  }
  if (message.type === "tool_result") {
    const entry = pending.get(message.id);
    if (!entry) return;
    pending.delete(message.id);
    entry.resolve(wrapToolResult(message.result));
  }
});

async function run(config) {
  activeConfig = config;
  try {
    const context = createSandbox(config);
    const script = new vm.Script('(async () => {\n"use strict";\n' + config.code + '\n})()');
    await script.runInContext(context, {
      timeout: Math.max(1, Math.min(config.timeoutMs, 1000)),
      breakOnSigint: false,
    });
    process.send?.({ type: "done", output: stdout, toolCallCount });
  } catch (error) {
    sendError(error instanceof Error ? error.message : String(error));
  }
}

function createSandbox(config) {
  const sandbox = {
    __toolNames: config.allowedToolNames,
    __print: print,
    __callTool: callTool,
  };
  const context = vm.createContext(sandbox, {
    codeGeneration: { strings: false, wasm: false },
  });
  new vm.Script([
    "(() => {",
    "  const localPrint = __print;",
    "  const localCallTool = __callTool;",
    "  const toolNames = Array.from(__toolNames);",
    "  const localTools = Object.create(null);",
    "  for (const name of toolNames) {",
    "    Object.defineProperty(localTools, name, {",
    "      enumerable: true,",
    "      value: async (args = {}) => localCallTool(name, args),",
    "    });",
    "  }",
    "  const proxyTools = new Proxy(localTools, {",
    "    get(target, prop) {",
    "      if (typeof prop !== \"string\") return undefined;",
    "      if (Object.prototype.hasOwnProperty.call(target, prop)) return target[prop];",
    "      return async (args = {}) => localCallTool(prop, args);",
    "    },",
    "  });",
    "  Object.defineProperty(globalThis, \"tools\", { enumerable: true, value: Object.freeze(proxyTools) });",
    "  const printWrapper = (...values) => localPrint(...values);",
    "  Object.defineProperty(globalThis, \"print\", { enumerable: true, value: printWrapper });",
    "  Object.defineProperty(globalThis, \"console\", {",
    "    enumerable: true,",
    "    value: Object.freeze({ log: printWrapper, info: printWrapper, warn: printWrapper, error: printWrapper }),",
    "  });",
    "  delete globalThis.__toolNames;",
    "  delete globalThis.__print;",
    "  delete globalThis.__callTool;",
    "})();",
  ].join("\n")).runInContext(context, { timeout: 1000 });
  return context;
}

async function callTool(name, args = {}) {
  if (!activeConfig.allowedToolNames.includes(name)) {
    throw new Error('Tool "' + name + '" is not available to programmatic scripts. Only readonly tools are exposed.');
  }
  if (toolCallCount >= activeConfig.maxToolCalls) {
    throw new Error("Script exceeded maximum of " + activeConfig.maxToolCalls + " tool call(s)");
  }
  toolCallCount++;
  const id = String(toolCallCount);
  const safeArgs = normalizeArgs(args);
  process.send?.({ type: "tool_call", id, name, args: safeArgs });
  return await new Promise((resolve) => pending.set(id, { resolve }));
}

function wrapToolResult(result) {
  if (result === null || typeof result !== "object") return result;
  return new Proxy(result, {
    get(target, prop) {
      if (prop === "content") {
        throw new Error("ToolResult has no result.content field; inspect result.output instead.");
      }
      return target[prop];
    },
  });
}

function print(...values) {
  appendStdout(values.map(formatValue).join(" ") + "\n");
}

function appendStdout(chunk) {
  stdout += chunk;
  if (stdout.length > activeConfig.maxOutputChars) {
    throw new Error("Script stdout exceeded maximum of " + activeConfig.maxOutputChars + " character(s)");
  }
}

function normalizeArgs(args) {
  if (args === null || typeof args !== "object" || Array.isArray(args)) return {};
  return JSON.parse(JSON.stringify(args));
}

function formatValue(value) {
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

function sendError(error) {
  process.send?.({
    type: "error",
    error,
    output: stdout,
    toolCallCount,
  });
}
`;
