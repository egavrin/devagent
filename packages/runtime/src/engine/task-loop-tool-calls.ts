import { normalizeRepoPath } from "./task-loop-paths.js";
import { parseToolScriptStepsArg } from "./tool-script.js";
import type { ToolScriptStep } from "./tool-script.js";
import { AgentType } from "../core/index.js";
import type { ToolResult, ToolSpec } from "../core/index.js";

interface PendingToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

interface NormalizedToolCall {
  readonly toolCall: PendingToolCall;
  readonly bypassResult: ToolResult | null;
  readonly scriptSteps: ToolScriptStep[] | null;
}

interface ToolExecutionBatchContext {
  readonly batchId?: string;
  readonly batchSize?: number;
}

interface ToolCallHost {
  readonly tools: { get(name: string): ToolSpec };
  iterations: number;
  parallelBatchCounter: number;
  delegateBatchId: string | null;
  delegateBatchIteration: number | null;
  successfulReadonlyCallKeys: Set<string>;
}

export function normalizeTaskLoopToolCall(
  loop: Pick<ToolCallHost, "successfulReadonlyCallKeys">,
  toolCall: PendingToolCall,
  category: ToolSpec["category"],
): NormalizedToolCall {
  if (category !== "readonly") return { toolCall, bypassResult: null, scriptSteps: null };
  if (toolCall.name === "execute_tool_script") {
    return normalizeTaskLoopToolScriptCall(loop, toolCall);
  }

  const key = buildReadonlyCallKey(toolCall.name, toolCall.arguments);
  if (!loop.successfulReadonlyCallKeys.has(key)) {
    return { toolCall, bypassResult: null, scriptSteps: null };
  }

  return {
    toolCall,
    bypassResult: {
      success: true,
      output: buildRedundantReadonlyMessage(toolCall),
      error: null,
      artifacts: [],
    },
    scriptSteps: null,
  };
}

export function collectSuccessfulTaskLoopScriptStepResults(
  steps: ReadonlyArray<ToolScriptStep>,
  scriptOutput: string,
): Array<{ step: ToolScriptStep; output: string }> {
  const sections = parseToolScriptOutputSections(scriptOutput);
  const successful: Array<{ step: ToolScriptStep; output: string }> = [];
  for (const step of steps) {
    const section = sections.get(step.id);
    if (!section || section.failed) continue;
    successful.push({ step, output: section.output });
  }
  return successful;
}

export function coalesceTaskLoopReplaceAllCalls(
  toolCalls: ReadonlyArray<PendingToolCall>,
): { toExecute: PendingToolCall[]; skipped: PendingToolCall[] } {
  const replaceAllTools = new Set(["update_plan"]);
  const lastIndex = findLastReplaceAllToolIndices(toolCalls, replaceAllTools);
  return splitReplaceAllCalls(toolCalls, replaceAllTools, lastIndex);
}

export function createTaskLoopBatchContextForCall(
  loop: ToolCallHost,
  call: PendingToolCall,
  allCalls: ReadonlyArray<PendingToolCall>,
): ToolExecutionBatchContext {
  if (call.name !== "delegate") return {};
  const tool = loop.tools.get(call.name);
  const parallelDelegates = allCalls.filter(
    (toolCall) => isTaskLoopParallelReadonlyDelegateCall(toolCall, tool),
  );
  if (parallelDelegates.length < 2) return {};
  return getDelegateBatchContext(loop, parallelDelegates.length);
}

export function isTaskLoopParallelReadonlyDelegateCall(
  toolCall: PendingToolCall,
  tool: ToolSpec,
): boolean {
  if (tool.name !== "delegate" || tool.category !== "workflow") return false;
  const agentType = toolCall.arguments["agent_type"];
  if (agentType === AgentType.EXPLORE || agentType === AgentType.REVIEWER) return true;
  return toolCall.arguments["parallel_safe"] === true;
}

function normalizeTaskLoopToolScriptCall(
  loop: Pick<ToolCallHost, "successfulReadonlyCallKeys">,
  toolCall: PendingToolCall,
): NormalizedToolCall {
  const parsedSteps = parseToolScriptStepsArg(toolCall.arguments["steps"]);
  if (!parsedSteps) return { toolCall, bypassResult: null, scriptSteps: null };
  const normalized = dedupeToolScriptSteps(loop.successfulReadonlyCallKeys, parsedSteps);
  if (normalized.skippedSteps.length === 0) {
    return { toolCall, bypassResult: null, scriptSteps: parsedSteps };
  }
  if (normalized.dedupedSteps.length === 0) {
    return buildFullySkippedScriptCall(toolCall, normalized.skippedSteps.length, parsedSteps);
  }
  return {
    toolCall: {
      ...toolCall,
      arguments: { ...toolCall.arguments, steps: normalized.dedupedSteps },
    },
    bypassResult: null,
    scriptSteps: normalized.dedupedSteps,
  };
}

function buildRedundantReadonlyMessage(toolCall: PendingToolCall): string {
  const path = toolCall.arguments["path"];
  const target = typeof path === "string" && path.trim().length > 0
    ? normalizeRepoPath(path)
    : toolCall.name;
  if (toolCall.name === "git_diff") {
    return `Skipped redundant git_diff for ${target}: identical diff already captured earlier in this run.`;
  }
  return `Skipped redundant readonly call ${toolCall.name}(${target}): this exact call already succeeded earlier in this run.`;
}

function dedupeToolScriptSteps(
  successfulReadonlyCallKeys: ReadonlySet<string>,
  parsedSteps: ReadonlyArray<ToolScriptStep>,
): { dedupedSteps: ToolScriptStep[]; skippedSteps: ToolScriptStep[] } {
  const referencedStepIds = collectReferencedStepIds(parsedSteps);
  const dedupedSteps: ToolScriptStep[] = [];
  const skippedSteps: ToolScriptStep[] = [];
  const seenInBatch = new Set<string>();
  for (const step of parsedSteps) {
    addOrSkipToolScriptStep(step, { referencedStepIds, successfulReadonlyCallKeys, seenInBatch, dedupedSteps, skippedSteps });
  }
  return { dedupedSteps, skippedSteps };
}

function addOrSkipToolScriptStep(
  step: ToolScriptStep,
  state: {
    readonly referencedStepIds: ReadonlySet<string>;
    readonly successfulReadonlyCallKeys: ReadonlySet<string>;
    readonly seenInBatch: Set<string>;
    readonly dedupedSteps: ToolScriptStep[];
    readonly skippedSteps: ToolScriptStep[];
  },
): void {
  const key = buildReadonlyCallKey(step.tool, step.args);
  if (state.referencedStepIds.has(step.id)) {
    state.seenInBatch.add(key);
    state.dedupedSteps.push(step);
    return;
  }
  if (state.successfulReadonlyCallKeys.has(key) || state.seenInBatch.has(key)) {
    state.skippedSteps.push(step);
    return;
  }
  state.seenInBatch.add(key);
  state.dedupedSteps.push(step);
}

function buildFullySkippedScriptCall(
  toolCall: PendingToolCall,
  skippedStepCount: number,
  parsedSteps: ToolScriptStep[],
): NormalizedToolCall {
  return {
    toolCall,
    bypassResult: {
      success: true,
      output:
        `Skipped execute_tool_script: all ${skippedStepCount} step(s) were already completed earlier in this run. Use session summaries and continue with new analysis.`,
      error: null,
      artifacts: [],
    },
    scriptSteps: parsedSteps,
  };
}

function findLastReplaceAllToolIndices(
  toolCalls: ReadonlyArray<PendingToolCall>,
  replaceAllTools: ReadonlySet<string>,
): Map<string, number> {
  const lastIndex = new Map<string, number>();
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const toolCall = toolCalls[i]!;
    if (replaceAllTools.has(toolCall.name) && !lastIndex.has(toolCall.name)) {
      lastIndex.set(toolCall.name, i);
    }
  }
  return lastIndex;
}

function splitReplaceAllCalls(
  toolCalls: ReadonlyArray<PendingToolCall>,
  replaceAllTools: ReadonlySet<string>,
  lastIndex: ReadonlyMap<string, number>,
): { toExecute: PendingToolCall[]; skipped: PendingToolCall[] } {
  const toExecute: PendingToolCall[] = [];
  const skipped: PendingToolCall[] = [];
  for (let i = 0; i < toolCalls.length; i++) {
    const toolCall = toolCalls[i]!;
    if (replaceAllTools.has(toolCall.name) && i !== lastIndex.get(toolCall.name)) {
      skipped.push(toolCall);
    } else {
      toExecute.push(toolCall);
    }
  }
  return { toExecute, skipped };
}

function getDelegateBatchContext(loop: ToolCallHost, batchSize: number): ToolExecutionBatchContext {
  if (loop.delegateBatchIteration !== loop.iterations) {
    loop.parallelBatchCounter++;
    loop.delegateBatchId = `delegate-batch-${loop.iterations}-${loop.parallelBatchCounter}`;
    loop.delegateBatchIteration = loop.iterations;
  }
  return {
    batchId: loop.delegateBatchId ?? undefined,
    batchSize,
  };
}

function collectReferencedStepIds(steps: ReadonlyArray<ToolScriptStep>): Set<string> {
  const referenced = new Set<string>();
  for (const step of steps) collectStepIdsFromValue(step.args, referenced);
  return referenced;
}

function collectStepIdsFromValue(value: unknown, out: Set<string>): void {
  if (typeof value === "string") {
    collectStepIdsFromString(value, out);
    return;
  }
  if (Array.isArray(value)) {
    for (const item of value) collectStepIdsFromValue(item, out);
    return;
  }
  if (value && typeof value === "object") {
    for (const child of Object.values(value as Record<string, unknown>)) {
      collectStepIdsFromValue(child, out);
    }
  }
}

function collectStepIdsFromString(value: string, out: Set<string>): void {
  const refPattern = /\$([a-zA-Z_][a-zA-Z0-9_]*)(?:\.lines\[(\d+)\])?/g;
  for (const match of value.matchAll(refPattern)) {
    const stepId = match[1];
    if (stepId) out.add(stepId);
  }
}

function parseToolScriptOutputSections(
  output: string,
): Map<string, { tool: string; failed: boolean; output: string }> {
  const sections = new Map<string, { tool: string; failed: boolean; output: string }>();
  const headerPattern = /^=== Step (\S+) \(([^)]+)\) \[(FAILED|\d+ms)\] ===$/gm;
  const headers = [...output.matchAll(headerPattern)];
  for (let i = 0; i < headers.length; i++) {
    sections.set(getScriptHeaderId(headers[i]!), getScriptOutputSection(output, headers, i));
  }
  return sections;
}

function getScriptHeaderId(header: RegExpMatchArray): string {
  return header[1]!;
}

function getScriptOutputSection(
  output: string,
  headers: ReadonlyArray<RegExpMatchArray>,
  index: number,
): { tool: string; failed: boolean; output: string } {
  const current = headers[index]!;
  const next = headers[index + 1];
  const headerEnd = (current.index ?? 0) + current[0].length;
  const tailEnd = next?.index ?? output.indexOf("\n\n[Script completed:", headerEnd);
  const segmentEnd = tailEnd === -1 ? output.length : tailEnd;
  const content = output.slice(headerEnd, segmentEnd).trim();
  return {
    tool: current[2]!,
    failed: current[3] === "FAILED",
    output: content.startsWith("Error: ") ? content.slice(7).trim() : content,
  };
}

function buildReadonlyCallKey(toolName: string, args: Record<string, unknown>): string {
  return `${toolName}:${JSON.stringify(normalizeArgsForReadonlyKey(args))}`;
}

function normalizeArgsForReadonlyKey(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => normalizeArgsForReadonlyKey(item));
  if (!value || typeof value !== "object") return value;
  return normalizeObjectArgsForReadonlyKey(value as Record<string, unknown>);
}

function normalizeObjectArgsForReadonlyKey(input: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(input).sort()) {
    const value = input[key];
    out[key] = key === "path" && typeof value === "string"
      ? normalizeRepoPath(value)
      : normalizeArgsForReadonlyKey(value);
  }
  return out;
}
