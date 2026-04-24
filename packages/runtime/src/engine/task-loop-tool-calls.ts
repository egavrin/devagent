import { normalizeRepoPath } from "./task-loop-paths.js";
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
  if (category !== "readonly") return { toolCall, bypassResult: null };

  const key = buildReadonlyCallKey(toolCall.name, toolCall.arguments);
  if (!loop.successfulReadonlyCallKeys.has(key)) {
    return { toolCall, bypassResult: null };
  }

  return {
    toolCall,
    bypassResult: {
      success: true,
      output: buildRedundantReadonlyMessage(toolCall),
      error: null,
      artifacts: [],
    },
  };
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
