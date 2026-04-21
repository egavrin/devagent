import { retryWithStrategy } from "./retry-strategy.js";
import { StreamingToolExecutor } from "./streaming-tool-executor.js";
import type {
  FinalTextValidator,
  TaskCompletionStatus,
  TaskLoopResult,
  TaskRunOptions,
} from "./task-loop.js";
import type {
  DevAgentConfig,
  EventBus,
  Message,
  ToolResult,
  ToolSpec,
} from "../core/index.js";
import {
  MessageRole,
  ProviderError,
  ProviderTlsCertificateError,
} from "../core/index.js";

interface PendingToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

interface TaskLoopRunHost {
  readonly config: DevAgentConfig;
  readonly bus: EventBus;
  readonly finalTextValidator: FinalTextValidator | null;
  readonly injectSessionStateOnFirstTurn: boolean;
  readonly sessionState: { hasContent(): boolean; hasPendingPlanSteps(): boolean } | null;
  readonly contextManager: unknown | null;
  readonly provider: { abort(): void };
  readonly messages: Message[];
  readonly totalCost: TaskLoopResult["cost"];
  aborted: boolean;
  iterations: number;
  estimatedTokens: number;
  lastReportedInputTokens: number;
  unresolvedDoubleCheckFailure: boolean;
  resetRunState(): void;
  installRunPrependedMessages(messages?: ReadonlyArray<Message>): void;
  removeRunPrependedMessages(): void;
  pushMessage(message: Message): void;
  getAgentEventFields(): Record<string, unknown>;
  getEffectiveContextBudget(): number;
  emitSubagentUpdate(event: { milestone: string; iteration: number; summary: string }): void;
  getAvailableTools(): ReadonlyArray<ToolSpec>;
  maybeCompactContext(options?: { force?: boolean }): Promise<void>;
  reactiveCompact(): Promise<void>;
  streamLLMResponse(tools: ReadonlyArray<ToolSpec>): Promise<{
    readonly textContent: string;
    readonly toolCalls: PendingToolCall[];
    readonly thinking: string;
  }>;
  coalesceReplaceAllCalls(calls: PendingToolCall[]): {
    readonly toExecute: PendingToolCall[];
    readonly skipped: PendingToolCall[];
  };
  isParallelReadonlyDelegateCall(call: PendingToolCall, tool: ToolSpec): boolean;
  createBatchContextForCall(
    call: PendingToolCall,
    allCalls: ReadonlyArray<PendingToolCall>,
  ): { readonly batchId?: string; readonly batchSize?: number };
  executeToolCall(
    call: PendingToolCall,
    availableToolNames: ReadonlySet<string>,
    availableTools: ReadonlyArray<ToolSpec>,
    batchContext: { readonly batchId?: string; readonly batchSize?: number },
  ): Promise<ToolResult>;
  maybeClassifyError(
    result: ToolResult,
    toolName: string,
    args: Record<string, unknown>,
  ): Promise<void>;
  maybeMergeDelegatedState(toolName: string, result: ToolResult): void;
  appendToolResult(
    callId: string,
    result: ToolResult,
    toolName: string,
    args: Record<string, unknown>,
  ): void;
  stagnationDetector: {
    checkStagnationWithLLM(
      provider: unknown,
      messages: ReadonlyArray<Message>,
      iteration: number,
    ): Promise<string | null>;
    checkDoomLoop(calls: ReadonlyArray<PendingToolCall>): string | null;
    checkToolFatigue(calls: ReadonlyArray<PendingToolCall>): string | null;
  };
  maybeMidpointBriefing(): Promise<void>;
  toolUseSummaryGenerator: {
    maybeSummarize(
      iteration: number,
      messages: ReadonlyArray<Message>,
      sessionState: TaskLoopRunHost["sessionState"],
    ): Message | null;
  };
  microcompact(): void;
}

interface RunState {
  readonly finalTextValidator: FinalTextValidator | null;
  hadToolCalls: boolean;
  summaryRequested: boolean;
  emptyResponseRetryUsed: boolean;
  budgetGraceUsed: boolean;
  planNudgeUsed: boolean;
  lastNonEmptyText: string | null;
  status: TaskCompletionStatus;
}

interface LLMRoundResult {
  readonly textContent: string;
  readonly toolCalls: PendingToolCall[];
  readonly thinkingContent: string;
}

export async function runTaskLoop(
  loop: TaskLoopRunHost,
  userQuery: string,
  options?: TaskRunOptions,
): Promise<TaskLoopResult> {
  const state = initializeRun(loop, userQuery, options);

  try {
    while (!loop.aborted) {
      if (handleBudgetLimit(loop, state) === "break") break;
      startIteration(loop);

      const availableTools = state.budgetGraceUsed ? [] : loop.getAvailableTools();
      await loop.maybeCompactContext();
      const response = await streamRoundWithRetry(loop, availableTools);

      if (response.toolCalls.length > 0) {
        await handleToolCalls(loop, state, response, availableTools);
        continue;
      }

      if (handleTextOrEmptyResponse(loop, state, response.textContent) === "continue") {
        continue;
      }
      break;
    }
  } finally {
    loop.removeRunPrependedMessages();
  }

  if (loop.aborted && state.status === "success") state.status = "aborted";
  return buildRunResult(loop, state);
}

function initializeRun(
  loop: TaskLoopRunHost,
  userQuery: string,
  options?: TaskRunOptions,
): RunState {
  loop.resetRunState();
  loop.installRunPrependedMessages(options?.prependedMessages);
  loop.pushMessage({ role: MessageRole.USER, content: userQuery });
  loop.bus.emit("message:user", {
    content: userQuery,
    ...loop.getAgentEventFields(),
  });
  if (loop.injectSessionStateOnFirstTurn && loop.sessionState?.hasContent()) {
    (loop as TaskLoopRunHost & { injectSessionState(): void }).injectSessionState();
  }
  return {
    finalTextValidator: options?.finalTextValidator ?? loop.finalTextValidator,
    hadToolCalls: false,
    summaryRequested: false,
    emptyResponseRetryUsed: false,
    budgetGraceUsed: false,
    planNudgeUsed: false,
    lastNonEmptyText: null,
    status: "success",
  };
}

function handleBudgetLimit(loop: TaskLoopRunHost, state: RunState): "continue" | "break" {
  const maxIterations = loop.config.budget.maxIterations;
  if (maxIterations <= 0 || loop.iterations < maxIterations) return "continue";
  if (state.budgetGraceUsed) {
    state.status = "budget_exceeded";
    return "break";
  }
  state.budgetGraceUsed = true;
  loop.pushMessage({
    role: MessageRole.SYSTEM,
    content: "You have reached the iteration limit. Please provide a concise summary of your progress and findings so far. Do not use any tools — respond with text only.",
  });
  return "continue";
}

function startIteration(loop: TaskLoopRunHost): void {
  loop.iterations++;
  const estimatedTokens = Math.max(loop.estimatedTokens, loop.lastReportedInputTokens);
  loop.bus.emit("iteration:start", {
    iteration: loop.iterations,
    maxIterations: loop.config.budget.maxIterations,
    estimatedTokens,
    maxContextTokens: loop.getEffectiveContextBudget(),
    ...loop.getAgentEventFields(),
  });
  loop.emitSubagentUpdate({
    milestone: "iteration:start",
    iteration: loop.iterations,
    summary: `Starting iteration ${loop.iterations}`,
  });
}

async function streamRoundWithRetry(
  loop: TaskLoopRunHost,
  availableTools: ReadonlyArray<ToolSpec>,
): Promise<LLMRoundResult> {
  let overflowCompactionUsed = false;
  const retryResult = await retryWithStrategy(
    () => loop.streamLLMResponse(availableTools),
    {
      overflowCompactionUsed,
      onRetry: (error, attempt, delayMs) => {
        loop.bus.emit("error", {
          message: formatProviderRetryMessage(error, attempt, delayMs),
          code: "PROVIDER_RETRY",
          fatal: false,
        });
      },
    },
  );

  if (retryResult.success) {
    return {
      textContent: retryResult.value!.textContent,
      toolCalls: retryResult.value!.toolCalls,
      thinkingContent: retryResult.value!.thinking,
    };
  }
  if (retryResult.shouldCompact && loop.contextManager) {
    overflowCompactionUsed = true;
    return retryAfterReactiveCompaction(loop, availableTools, retryResult.error!);
  }
  if (retryResult.shouldFallback) emitFallbackHint(loop);
  throw retryResult.error!;
}

function formatProviderRetryMessage(error: ProviderError, attempt: number, delayMs: number): string {
  if (error instanceof ProviderTlsCertificateError) {
    return `Provider certificate verification failed (attempt ${attempt}): ${error.detail}. Check NODE_EXTRA_CA_CERTS and HTTPS_PROXY/HTTP_PROXY/NO_PROXY. Retrying in ${delayMs}ms…`;
  }
  return `Provider error (attempt ${attempt}, ${error.name}): ${error.message}. Retrying in ${delayMs}ms…`;
}

async function retryAfterReactiveCompaction(
  loop: TaskLoopRunHost,
  availableTools: ReadonlyArray<ToolSpec>,
  originalError: ProviderError,
): Promise<LLMRoundResult> {
  await loop.reactiveCompact();
  loop.bus.emit("error", {
    message: "Provider rejected prompt for context size. Forced reactive compaction and retrying.",
    code: "CONTEXT_OVERFLOW_RETRY",
    fatal: false,
  });
  try {
    const result = await loop.streamLLMResponse(availableTools);
    return {
      textContent: result.textContent,
      toolCalls: result.toolCalls,
      thinkingContent: result.thinking,
    };
  } catch (retryErr) {
    throw retryErr instanceof ProviderError ? retryErr : originalError;
  }
}

function emitFallbackHint(loop: TaskLoopRunHost): void {
  const fallbackModel = loop.config.providers[loop.config.provider]?.fallbackModel;
  if (!fallbackModel) return;
  loop.bus.emit("error", {
    message: `Primary model exhausted retries. Falling back to ${fallbackModel}.`,
    code: "MODEL_FALLBACK",
    fatal: false,
  });
}

async function handleToolCalls(
  loop: TaskLoopRunHost,
  state: RunState,
  response: LLMRoundResult,
  availableTools: ReadonlyArray<ToolSpec>,
): Promise<void> {
  state.hadToolCalls = true;
  appendAssistantToolCallMessage(loop, response);
  const { toExecute, skipped } = loop.coalesceReplaceAllCalls(response.toolCalls);
  appendSkippedToolCalls(loop, skipped);
  await executeToolBatch(loop, toExecute, availableTools);
  await appendStagnationWarnings(loop, response.toolCalls);
  await loop.maybeCompactContext();
  await loop.maybeMidpointBriefing();
  appendToolUseSummary(loop);
  loop.microcompact();
}

function appendAssistantToolCallMessage(loop: TaskLoopRunHost, response: LLMRoundResult): void {
  const toolCalls = response.toolCalls.map((tc) => ({
    name: tc.name,
    arguments: tc.arguments,
    callId: tc.callId,
  }));
  loop.pushMessage({
    role: MessageRole.ASSISTANT,
    content: response.textContent,
    toolCalls,
    ...(response.thinkingContent ? { thinking: response.thinkingContent } : {}),
  });
  loop.bus.emit("message:assistant", {
    content: response.textContent,
    partial: false,
    toolCalls,
    ...loop.getAgentEventFields(),
  });
}

function appendSkippedToolCalls(loop: TaskLoopRunHost, skipped: ReadonlyArray<PendingToolCall>): void {
  for (const call of skipped) {
    const content = "Skipped: superseded by a later update_plan call in this batch.";
    loop.pushMessage({ role: MessageRole.TOOL, content, toolCallId: call.callId });
    loop.bus.emit("message:tool", {
      role: "tool" as const,
      content,
      toolCallId: call.callId,
      toolName: call.name,
      ...loop.getAgentEventFields(),
    });
  }
}

async function executeToolBatch(
  loop: TaskLoopRunHost,
  calls: ReadonlyArray<PendingToolCall>,
  availableTools: ReadonlyArray<ToolSpec>,
): Promise<void> {
  const availableToolNames = new Set(availableTools.map((tool) => tool.name));
  const executor = new StreamingToolExecutor(
    (name) => resolveToolCategory(loop, name, calls, availableToolNames),
    loop.config.budget.maxToolConcurrency ?? 10,
  );
  for (const call of calls) {
    const batchContext = loop.createBatchContextForCall(call, calls);
    executor.submit(call, (submitted) =>
      loop.executeToolCall(submitted, availableToolNames, availableTools, batchContext),
    );
  }
  for await (const { call, result } of executor.results()) {
    if (loop.aborted) break;
    await loop.maybeClassifyError(result, call.name, call.arguments);
    loop.maybeMergeDelegatedState(call.name, result);
    loop.appendToolResult(call.callId, result, call.name, call.arguments);
  }
}

function resolveToolCategory(
  loop: TaskLoopRunHost,
  name: string,
  calls: ReadonlyArray<PendingToolCall>,
  availableToolNames: ReadonlySet<string>,
) {
  if (!availableToolNames.has(name)) return null;
  const tool = (loop as TaskLoopRunHost & { tools: { get(name: string): ToolSpec } }).tools.get(name);
  const matchingCall = calls.find((call) => call.name === name);
  return matchingCall && loop.isParallelReadonlyDelegateCall(matchingCall, tool)
    ? "readonly"
    : tool.category;
}

async function appendStagnationWarnings(
  loop: TaskLoopRunHost,
  toolCalls: ReadonlyArray<PendingToolCall>,
): Promise<void> {
  const stagnationNudge = await loop.stagnationDetector.checkStagnationWithLLM(
    loop.provider,
    loop.messages,
    loop.iterations,
  );
  appendSystemNudge(loop, stagnationNudge);
  appendSystemNudge(loop, loop.stagnationDetector.checkDoomLoop(toolCalls));
  appendSystemNudge(loop, loop.stagnationDetector.checkToolFatigue(toolCalls));
}

function appendSystemNudge(loop: TaskLoopRunHost, content: string | null): void {
  if (content) loop.pushMessage({ role: MessageRole.SYSTEM, content });
}

function appendToolUseSummary(loop: TaskLoopRunHost): void {
  const summary = loop.toolUseSummaryGenerator.maybeSummarize(
    loop.iterations,
    loop.messages,
    loop.sessionState,
  );
  if (summary) loop.pushMessage(summary);
}

function handleTextOrEmptyResponse(
  loop: TaskLoopRunHost,
  state: RunState,
  textContent: string,
): "continue" | "break" {
  if (textContent) return handleTextResponse(loop, state, textContent);
  return handleEmptyResponse(loop, state);
}

function handleTextResponse(
  loop: TaskLoopRunHost,
  state: RunState,
  textContent: string,
): "continue" | "break" {
  if (appendDoubleCheckNudge(loop, state, textContent)) return "continue";
  if (appendPlanNudge(loop, state, textContent)) return "continue";
  if (appendFinalTextValidationNudge(loop, state, textContent)) return "continue";

  state.lastNonEmptyText = textContent;
  appendAssistantText(loop, textContent);
  state.status = "success";
  return "break";
}

function appendDoubleCheckNudge(loop: TaskLoopRunHost, state: RunState, text: string): boolean {
  if (!loop.unresolvedDoubleCheckFailure || !state.hadToolCalls) return false;
  appendAssistantText(loop, text);
  loop.pushMessage({
    role: MessageRole.SYSTEM,
    content: "Double-check still failing from prior edits. You must fix validation errors before finalizing.",
  });
  loop.unresolvedDoubleCheckFailure = false;
  return true;
}

function appendPlanNudge(loop: TaskLoopRunHost, state: RunState, text: string): boolean {
  const hasIncompleteSteps = loop.sessionState?.hasPendingPlanSteps() ?? false;
  if (!hasIncompleteSteps || !state.hadToolCalls || state.planNudgeUsed) return false;
  state.planNudgeUsed = true;
  appendAssistantText(loop, text);
  loop.pushMessage({
    role: MessageRole.SYSTEM,
    content: "Your plan has incomplete steps. Continue working — use tools to make progress on the next pending step.",
  });
  return true;
}

function appendFinalTextValidationNudge(loop: TaskLoopRunHost, state: RunState, text: string): boolean {
  const validation = state.finalTextValidator?.(text) ?? { valid: true };
  if (validation.valid) return false;
  appendAssistantText(loop, text);
  loop.pushMessage({
    role: MessageRole.SYSTEM,
    content: validation.retryMessage
      ?? "Your previous reply was not a valid final response for this stage. Return the final artifact now without commentary.",
  });
  return true;
}

function appendAssistantText(loop: TaskLoopRunHost, content: string): void {
  loop.pushMessage({ role: MessageRole.ASSISTANT, content });
  loop.bus.emit("message:assistant", {
    content,
    partial: false,
    ...loop.getAgentEventFields(),
  });
}

function handleEmptyResponse(loop: TaskLoopRunHost, state: RunState): "continue" | "break" {
  if (!state.hadToolCalls && !state.emptyResponseRetryUsed) {
    state.emptyResponseRetryUsed = true;
    loop.pushMessage({
      role: MessageRole.SYSTEM,
      content: "Your previous response was empty. Return the final artifact now as plain text and do not leave the reply blank.",
    });
    return "continue";
  }
  if (state.hadToolCalls && !state.summaryRequested) {
    state.summaryRequested = true;
    loop.pushMessage({
      role: MessageRole.SYSTEM,
      content: "Please provide a summary of your findings and conclusions based on the work done so far.",
    });
    return "continue";
  }
  state.status = "empty_response";
  return "break";
}

function buildRunResult(loop: TaskLoopRunHost, state: RunState): TaskLoopResult {
  return {
    messages: loop.messages,
    iterations: loop.iterations,
    cost: loop.totalCost,
    aborted: loop.aborted,
    status: state.status,
    lastText: state.lastNonEmptyText,
  };
}
