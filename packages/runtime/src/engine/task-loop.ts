/**
 * ReAct task loop — the core execution engine.
 * Streams LLM responses, parses tool calls, checks approval,
 * executes tools, feeds results back to LLM.
 * Fail fast: tool errors surface to LLM immediately.
 *
 * Auto-compaction: monitors estimated token usage and triggers
 * context truncation (sliding window or LLM-based summarization)
 * when approaching the budget limit, following the Codex pattern.
 */

import type { DoubleCheck } from "./double-check.js";
import type { SessionState } from "./session-state.js";
import { StagnationDetector } from "./stagnation-detector.js";
import { maybeCompactTaskLoopContext, reactiveCompactTaskLoop } from "./task-loop-compaction.js";
import {
  microcompactTaskLoop,
  pruneTaskLoopToolOutputs,
  resetTaskLoopPostCompactionState,
} from "./task-loop-pruning.js";
import { captureTaskLoopReviewScopeFiles } from "./task-loop-review-scope.js";
import { runTaskLoop } from "./task-loop-run.js";
import { injectTaskLoopSessionState } from "./task-loop-session-state.js";
import {
  coalesceTaskLoopReplaceAllCalls,
  collectSuccessfulTaskLoopScriptStepResults,
  createTaskLoopBatchContextForCall,
  isTaskLoopParallelReadonlyDelegateCall,
  normalizeTaskLoopToolCall,
} from "./task-loop-tool-calls.js";
import {
  executeTaskLoopToolCall,
  streamTaskLoopLLMResponse,
} from "./task-loop-tool-execution.js";
import {
  appendTaskLoopToolResult,
  getTaskLoopSummaryTarget,
  maybeMergeTaskLoopDelegatedState,
} from "./task-loop-tool-results.js";
export { truncateToolOutput } from "./task-loop-tool-results.js";
import type { ToolScriptStep } from "./tool-script.js";
import { ToolUseSummaryGenerator } from "./tool-use-summary.js";
import type {
  ApprovalGate,
  ContextManager,
  CostRecord,
  DevAgentConfig,
  EventBus,
  LLMProvider,
  Message,
  AgentType,
  ToolSpec,
  ToolResult,
} from "../core/index.js";
import {
  MessageRole,
  estimateMessageTokens,
  lookupModelPricing,
} from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";

// Re-export formatter functions that were previously defined here,
// preserving backward compatibility for external consumers.
export { summarizeDiff, summarizeTestOutput, extractStructuralDigest } from "./tool-summary-formatter.js";

// ─── Types ──────────────────────────────────────────────────

export type TaskMode = "plan" | "act";

export type TaskCompletionStatus =
  | "success"
  | "empty_response"
  | "budget_exceeded"
  | "aborted";

/**
 * Callback invoked at midpoint briefing intervals during long-running turns.
 * Receives the current messages and iteration count. Returns replacement
 * messages (synthesized context) or null to skip midpoint briefing.
 */
export type MidpointCallback = (
  messages: ReadonlyArray<Message>,
  iteration: number,
) => Promise<{ continueMessages: ReadonlyArray<Message> } | null>;

export interface FinalTextValidationResult {
  readonly valid: boolean;
  readonly retryMessage?: string;
}

export type FinalTextValidator = (
  candidate: string,
) => FinalTextValidationResult;

export interface TaskLoopOptions {
  readonly provider: LLMProvider;
  readonly tools: ToolRegistry;
  readonly bus: EventBus;
  readonly approvalGate: ApprovalGate;
  readonly config: DevAgentConfig;
  readonly systemPrompt: string;
  readonly repoRoot: string;
  readonly mode?: TaskMode;
  readonly contextManager?: ContextManager;
  readonly doubleCheck?: DoubleCheck;
  readonly initialMessages?: ReadonlyArray<Message>;
  /** Callback for midpoint context re-synthesis during long-running turns. */
  readonly midpointCallback?: MidpointCallback;
  /** Session state sidecar — structured facts that survive compaction. */
  readonly sessionState?: SessionState;
  /** Inject pre-seeded session state before the first provider call. */
  readonly injectSessionStateOnFirstTurn?: boolean;
  /** Optional validation hook for text-only terminal responses. */
  readonly finalTextValidator?: FinalTextValidator;
  /** Child-agent identity for nested execution and logging. */
  readonly agentContext?: AgentExecutionContext;
}

export interface TaskLoopResult {
  readonly messages: ReadonlyArray<Message>;
  readonly iterations: number;
  readonly cost: CostRecord;
  readonly aborted: boolean;
  readonly status: TaskCompletionStatus;
  readonly lastText: string | null;
}

export interface TaskRunOptions {
  readonly prependedMessages?: ReadonlyArray<Message>;
  readonly finalTextValidator?: FinalTextValidator;
}

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

interface AgentExecutionContext {
  readonly agentId: string;
  readonly parentAgentId: string | null;
  readonly depth: number;
  readonly agentType: AgentType;
  readonly laneLabel?: string | null;
  readonly batchId?: string;
  readonly batchSize?: number;
}

interface RunPrependedState {
  readonly insertionIndex: number;
  readonly messages: ReadonlyArray<Message>;
  readonly tokenCount: number;
}

interface TaskLoopInitialServices {
  readonly mode: TaskMode;
  readonly contextManager: ContextManager | null;
  readonly doubleCheck: DoubleCheck | null;
  readonly midpointCallback: MidpointCallback | null;
  readonly midpointInterval: number;
  readonly sessionState: SessionState | null;
  readonly injectSessionStateOnFirstTurn: boolean;
  readonly finalTextValidator: FinalTextValidator | null;
  readonly agentContext: AgentExecutionContext | null;
}

// ─── Tool Output Truncation ─────────────────────────────────

// ─── Deduplication Tools ────────────────────────────────────
// ─── Task Loop ──────────────────────────────────────────────

export class TaskLoop {
  private readonly provider: LLMProvider;
  private readonly tools: ToolRegistry;
  private readonly bus: EventBus;
  private readonly approvalGate: ApprovalGate;
  private readonly config: DevAgentConfig;
  private systemPrompt: string;
  private readonly repoRoot: string;
  private readonly contextManager: ContextManager | null;
  private readonly doubleCheck: DoubleCheck | null;
  private readonly midpointCallback: MidpointCallback | null;
  private readonly midpointInterval: number;
  private readonly sessionState: SessionState | null;
  private readonly finalTextValidator: FinalTextValidator | null;
  private readonly agentContext: AgentExecutionContext | null;
  private mode: TaskMode;
  private messages: Message[] = [];
  private iterations = 0;
  private totalCost: CostRecord = {
    inputTokens: 0,
    outputTokens: 0,
    cacheReadTokens: 0,
    cacheWriteTokens: 0,
    totalCost: 0,
  };
  private aborted = false;
  private readonly stagnationDetector: StagnationDetector;
  private unresolvedDoubleCheckFailure = false;
  private parallelBatchCounter = 0;
  private lastReportedInputTokens = 0;
  private readonly cachedPricing;
  /** Running estimate of total message tokens — avoids expensive full-array scans. */
  private estimatedTokens: number = 0;
  /** Tracks message index of the last tool result for each tool:target key (for deduplication). */
  private toolResultIndices = new Map<string, number>();
  /** Readonly tool calls that already succeeded in this run (tool + normalized args). */
  private successfulReadonlyCallKeys = new Set<string>();
  /** Whether the approaching-limit warning has been injected (reset on compaction). */
  private approachingLimitWarned = false;
  /** Number of auto-pinned git_diff results in this session. */
  private pinnedDiffCount = 0;
  /** Whether pre-seeded session state should be injected before the first provider call. */
  private readonly injectSessionStateOnFirstTurn: boolean;
  /** Microcompact: tracks aggregate chars of tool results for proactive clearing. */
  private toolResultTotalChars = 0;
  private toolResultEntries: Array<{ index: number; chars: number; tool: string; iteration: number }> = [];
  /** Reactive compaction circuit breaker: consecutive failures. */
  private reactiveCompactFailures = 0;
  /** Periodic tool-use summary generator. */
  private readonly toolUseSummaryGenerator: ToolUseSummaryGenerator;
  /** Number of compaction cycles completed (for session memory extraction). */
  private compactionCycles = 0;
  /** Invoked skill content that survives compaction for re-injection. */
  private invokedSkillContent = new Map<string, string>();
  constructor(options: TaskLoopOptions) {
    const services = getInitialServices(options);
    this.provider = options.provider;
    this.tools = options.tools;
    this.bus = options.bus;
    this.approvalGate = options.approvalGate;
    this.config = options.config;
    this.systemPrompt = options.systemPrompt;
    this.repoRoot = options.repoRoot;
    this.mode = services.mode;
    this.contextManager = services.contextManager;
    this.doubleCheck = services.doubleCheck;
    this.midpointCallback = services.midpointCallback;
    this.midpointInterval = services.midpointInterval;
    this.sessionState = services.sessionState;
    this.injectSessionStateOnFirstTurn = services.injectSessionStateOnFirstTurn;
    this.finalTextValidator = services.finalTextValidator;
    this.agentContext = services.agentContext;
    this.cachedPricing = lookupModelPricing(this.config.model, this.config.provider);
    this.stagnationDetector = new StagnationDetector({
      bus: this.bus,
      sessionState: this.sessionState,
    });
    this.toolUseSummaryGenerator = new ToolUseSummaryGenerator({
      interval: options.config.context.midpointBriefingInterval ?? 10,
    });

    this.messages = getInitialMessages(options);
    this.estimatedTokens = estimateMessageTokens(this.messages);
  }

  private getAgentEventFields(): {
    readonly agentId?: string;
    readonly parentAgentId?: string | null;
    readonly depth?: number;
    readonly agentType?: AgentType;
  } {
    if (!this.agentContext) return {};
    return {
      agentId: this.agentContext.agentId,
      parentAgentId: this.agentContext.parentAgentId,
      depth: this.agentContext.depth,
      agentType: this.agentContext.agentType,
    };
  }

  private emitSubagentUpdate(event: {
    readonly milestone: "iteration:start" | "tool:before" | "tool:after";
    readonly iteration?: number;
    readonly toolName?: string;
    readonly toolCallId?: string;
    readonly toolSuccess?: boolean;
    readonly durationMs?: number;
    readonly summary?: string;
  }): void {
    if (!this.agentContext || this.agentContext.parentAgentId === null) return;
    this.bus.emit("subagent:update", {
      agentId: this.agentContext.agentId,
      parentAgentId: this.agentContext.parentAgentId,
      depth: this.agentContext.depth,
      agentType: this.agentContext.agentType,
      laneLabel: this.agentContext.laneLabel,
      status: "running",
      batchId: this.agentContext.batchId,
      batchSize: this.agentContext.batchSize,
      ...event,
    });
  }

  /**
   * Run the task loop with a user query.
   * Returns when the LLM produces a final text response (no more tool calls)
   * or when the budget is exceeded.
   */
  async run(userQuery: string, options?: TaskRunOptions): Promise<TaskLoopResult> {
    return runTaskLoop(this as unknown as Parameters<typeof runTaskLoop>[0], userQuery, options);
  }

  abort(): void {
    this.aborted = true;
    this.provider.abort();
  }

  setMode(mode: TaskMode): void {
    this.mode = mode;
  }

  getMode(): TaskMode {
    return this.mode;
  }

  /**
   * Get the current message history.
   */
  getMessages(): ReadonlyArray<Message> {
    return this.messages;
  }

  /**
   * Replace the root system prompt without clearing accumulated conversation
   * history so future turns pick up the new ambient instructions.
   */
  updateSystemPrompt(systemPrompt: string): void {
    this.systemPrompt = systemPrompt;

    const firstSystemIndex = this.messages.findIndex((message) => message.role === MessageRole.SYSTEM);
    if (firstSystemIndex >= 0) {
      const prior = this.messages[firstSystemIndex]!;
      this.messages[firstSystemIndex] = {
        ...prior,
        content: systemPrompt,
      };
    } else {
      this.messages.unshift({
        role: MessageRole.SYSTEM,
        content: systemPrompt,
      });
    }

    this.estimatedTokens = estimateMessageTokens(this.messages);
  }

  /**
   * Get the current iteration count.
   */
  getIterations(): number {
    return this.iterations;
  }

  /**
   * Reset iteration counter for a new turn in multi-turn conversations.
   * Does NOT clear message history — messages persist for context.
   */
  resetIterations(): void {
    this.iterations = 0;
    this.aborted = false;
    this.stagnationDetector.resetAll();
    this.resetRunState();
  }

  /** Reset per-run transient state (shared between run() and resetIterations()). */
  private resetRunState(): void {
    this.unresolvedDoubleCheckFailure = false;
    this.successfulReadonlyCallKeys.clear();
    this.stagnationDetector.resetRunState();
    this.toolUseSummaryGenerator.reset();
    this.reactiveCompactFailures = 0;
    this.delegateBatchId = null;
    this.delegateBatchIteration = null;
  }

  /**
   * Push a message and increment the running token counter.
   * Use this instead of `this.messages.push(msg)` in hot paths
   * so that `this.estimatedTokens` stays in sync.
   */
  private pushMessage(msg: Message): void {
    this.messages.push(msg);
    this.estimatedTokens += estimateMessageTokens([msg]);
  }

  private installRunPrependedMessages(
    prependedMessages: ReadonlyArray<Message> | undefined,
  ): void {
    if (!prependedMessages || prependedMessages.length === 0) {
      this.runPrependedState = null;
      return;
    }

    const messages = [...prependedMessages];
    const tokenCount = estimateMessageTokens(messages);
    this.runPrependedState = {
      insertionIndex: this.messages.length,
      messages,
      tokenCount,
    };
    this.estimatedTokens += tokenCount;
  }

  private removeRunPrependedMessages(): void {
    if (!this.runPrependedState) {
      return;
    }

    this.estimatedTokens = Math.max(0, this.estimatedTokens - this.runPrependedState.tokenCount);
    this.runPrependedState = null;
  }

  private getMessagesForProvider(): ReadonlyArray<Message> {
    if (!this.runPrependedState) {
      return this.messages;
    }

    const { insertionIndex, messages } = this.runPrependedState;
    return [
      ...this.messages.slice(0, insertionIndex),
      ...messages,
      ...this.messages.slice(insertionIndex),
    ];
  }

  // ─── Private ────────────────────────────────────────────────

  /**
   * Auto-compact context when approaching the token budget.
   * Uses the ContextManager's truncateAsync (sliding window or hybrid
   * summarization) to compress older messages while preserving the
   * system prompt and original user task.
   */
  private async maybeCompactContext(
    options?: { force?: boolean },
  ): Promise<void> {
    await maybeCompactTaskLoopContext(
      this as unknown as Parameters<typeof maybeCompactTaskLoopContext>[0],
      options,
    );
  }

  /**
   * Microcompact: proactively clear oldest compactable tool results
   * when aggregate tool result chars exceed the configured budget.
   * Runs after every tool batch to prevent unbounded context growth.
   */
  private microcompact(): void {
    microcompactTaskLoop(this as unknown as Parameters<typeof microcompactTaskLoop>[0]);
  }

  /**
   * Reactive compaction: multi-stage compaction triggered by API rejection.
   * Runs microcompact → Phase-1 pruning → full hybrid compaction.
   * Circuit breaker: stops after MAX consecutive failures.
   */
  private async reactiveCompact(): Promise<void> {
    await reactiveCompactTaskLoop(
      this as unknown as Parameters<typeof reactiveCompactTaskLoop>[0],
    );
  }

  /**
   * Phase-1 pruning: incrementally replace old, large tool results with
   * compact summaries. Prunes from oldest to newest, stopping once enough
   * tokens have been freed to get back under the compaction threshold.
   * Protects the most recent tool outputs (protection window).
   *
   * Returns estimated pruning stats.
   */
  private pruneToolOutputs(
    currentTokens: number,
    threshold: number,
  ): { savedTokens: number; prunedCount: number } {
    return pruneTaskLoopToolOutputs(
      this as unknown as Parameters<typeof pruneTaskLoopToolOutputs>[0],
      currentTokens,
      threshold,
    );
  }

  /**
   * Inject session state as a SYSTEM message after compaction.
   * Placed right after the system prompt so the LLM sees it early.
   * Replaces any previous session-state message to avoid accumulation.
   *
   * Auto-selects tier based on available context headroom:
   * - "full" (>8k tokens): plan + files + env + tool summaries
   * - "compact" (>3k tokens): plan + files + env + recent summaries
   * - "minimal" (<=3k tokens): plan + recent summaries
   *
   * @param knownTokenEstimate - pre-computed token estimate to avoid redundant scan
   */
  private resetPostCompactionState(full: boolean): void {
    resetTaskLoopPostCompactionState(
      this as unknown as Parameters<typeof resetTaskLoopPostCompactionState>[0],
      full,
    );
  }
  private injectSessionState(knownTokenEstimate?: number): void {
    injectTaskLoopSessionState(
      this as unknown as Parameters<typeof injectTaskLoopSessionState>[0],
      knownTokenEstimate,
    );
  }

  /**
   * Re-inject invoked skill content after compaction.
   * Skills are tracked in invokedSkillContent (survives compaction)
   * and re-injected as a system message so the LLM doesn't lose
   * skill instructions mid-session.
   */
  private reinjectSkillContent(): void {
    if (this.invokedSkillContent.size === 0) return;

    const sections = [...this.invokedSkillContent.entries()]
      .map(([name, content]) => `## Skill: ${name}\n${content}`)
      .join("\n\n");
    // Budget: ~10K tokens max for re-injected skills
    const truncated = sections.length > 40_000
      ? sections.slice(0, 40_000) + "\n\n[... skill content truncated ...]"
      : sections;

    this.pushMessage({
      role: MessageRole.SYSTEM,
      content: `[PRESERVED SKILL CONTENT — re-injected after compaction]\n\n${truncated}`,
    });
  }

  private getEffectiveContextBudget(): number {
    const maxContextTokens = this.config.budget.maxContextTokens;
    if (maxContextTokens <= 0) return 0;
    const headroom = Math.max(0, this.config.budget.responseHeadroom);
    const effective = maxContextTokens - headroom;
    if (effective <= 0) {
      throw new Error(
        `Invalid context budget: budget.maxContextTokens (${maxContextTokens}) must be greater than budget.responseHeadroom (${headroom})`,
      );
    }
    return effective;
  }

  /**
   * Append co-located error guidance from the tool definition.
   * Deterministic, free (no LLM call), runs on every failure.
   */
  private applyErrorGuidance(result: ToolResult, tool: ToolSpec): void {
    if (result.success || !result.error || !tool.errorGuidance) return;
    const guidance = tool.errorGuidance;
    let hint = guidance.common;
    if (guidance.patterns) {
      const matched = guidance.patterns.find((p) => result.error!.includes(p.match));
      if (matched) hint = matched.hint;
    }
    (result as { error: string | null }).error += `\n[Recovery hint] ${hint}`;
  }

  /**
   * Classify tool errors using the LLM judge and append classification
   * to the error message. Mutates result.error in place when applicable.
   */
  private async maybeClassifyError(
    result: ToolResult,
    toolName: string,
    toolArgs: Record<string, unknown>,
  ): Promise<void> {
    if (result.success || !result.error) return;
    try {
      const classification = await this.stagnationDetector.classifyToolError(
        this.provider, toolName, toolArgs, result.error,
        this.messages, this.iterations,
      );
      if (classification) {
        (result as { error: string | null }).error += `\n${classification}`;
      }
    } catch {
      // Classification failure is non-fatal
    }
  }

  /**
   * Applies tool output truncation (head+tail) to prevent single tool calls
   * from dominating the context window. Also deduplicates: if the same
   * readonly tool was called on the same target before, the older result
   * is replaced with a superseded note.
   */
  private appendToolResult(
    callId: string,
    result: ToolResult,
    toolName?: string,
    toolArgs?: Record<string, unknown>,
  ): void {
    appendTaskLoopToolResult(
      this as unknown as Parameters<typeof appendTaskLoopToolResult>[0],
      callId,
      result,
      toolName,
      toolArgs,
    );
  }

  private maybeMergeDelegatedState(toolName: string, result: ToolResult): void {
    maybeMergeTaskLoopDelegatedState(
      this as unknown as Parameters<typeof maybeMergeTaskLoopDelegatedState>[0],
      toolName,
      result,
    );
  }

  /**
   * Proactive midpoint briefing: at regular intervals, re-synthesize
   * context to prevent accumulated history from degrading LLM accuracy.
   * Unlike reactive compaction (which triggers at token budget limit),
   * this fires proactively every N iterations regardless of token count.
   */
  private async maybeMidpointBriefing(): Promise<void> {
    if (!this.midpointCallback) return;
    if (this.midpointInterval <= 0) return;
    if (this.iterations <= 0 || this.iterations % this.midpointInterval !== 0) return;

    const tokensBefore = this.estimatedTokens;
    const result = await this.midpointCallback(this.messages, this.iterations);
    if (result) {
      this.messages = [...result.continueMessages];
      // Full recalculation: midpoint briefing replaces the entire message array.
      this.estimatedTokens = estimateMessageTokens(this.messages);
      this.injectSessionState();
      this.resetPostCompactionState(false);
      this.bus.emit("context:compacted", {
        removedCount: 0,
        estimatedTokens: this.estimatedTokens,
        tokensBefore,
      });
    }
  }

  private getAvailableTools(): ReadonlyArray<ToolSpec> {
    if (this.mode === "plan") {
      return this.tools.getPlanModeTools();
    }
    // Use getLoaded() to exclude deferred tools from API calls.
    // Deferred tools are listed in the system prompt as stubs
    // and resolved on demand via tool_search.
    return this.tools.getLoaded();
  }

  private normalizeToolCall(
    toolCall: PendingToolCall,
    category: ToolSpec["category"],
  ): NormalizedToolCall {
    return normalizeTaskLoopToolCall(
      this as unknown as Parameters<typeof normalizeTaskLoopToolCall>[0],
      toolCall,
      category,
    );
  }

  private collectSuccessfulScriptStepResults(
    steps: ReadonlyArray<ToolScriptStep>,
    scriptOutput: string,
  ): Array<{ step: ToolScriptStep; output: string }> {
    return collectSuccessfulTaskLoopScriptStepResults(steps, scriptOutput);
  }
  private getSummaryTarget(
    toolName: string,
    args: Record<string, unknown>,
  ): string | null {
    return getTaskLoopSummaryTarget(toolName, args);
  }

  /**
   * Coalesce replace-all tool calls: when multiple calls to the same tool
   * with replace-all semantics (like update_plan) appear in a single LLM
   * response batch, only execute the last one. Earlier calls are skipped
   * with synthetic results.
   */
  private coalesceReplaceAllCalls(
    toolCalls: ReadonlyArray<PendingToolCall>,
  ): { toExecute: PendingToolCall[]; skipped: PendingToolCall[] } {
    return coalesceTaskLoopReplaceAllCalls(toolCalls);
  }

  /**
   * Partition tool calls into batches for parallel/sequential execution.
   *
   * Rules:
   * - Consecutive readonly calls → single parallel batch (safe to run concurrently)
   * - A mutating/workflow/external call → its own sequential batch (must run alone)
   * - Unknown tools → sequential batch (fail gracefully)
   *
   * A single readonly call is still "parallel" (batch of 1) — no overhead.
   */
  /**
   * Create a batch context for a single call within the streaming executor.
   * For delegate calls that are part of a parallel group, assigns a shared batch ID.
   */
  private createBatchContextForCall(
    call: PendingToolCall,
    allCalls: ReadonlyArray<PendingToolCall>,
  ): ToolExecutionBatchContext {
    return createTaskLoopBatchContextForCall(
      this as unknown as Parameters<typeof createTaskLoopBatchContextForCall>[0],
      call,
      allCalls,
    );
  }

  private delegateBatchId: string | null = null;
  private delegateBatchIteration: number | null = null;
  private runPrependedState: RunPrependedState | null = null;

  private isParallelReadonlyDelegateCall(
    toolCall: PendingToolCall,
    tool: ToolSpec,
  ): boolean {
    return isTaskLoopParallelReadonlyDelegateCall(toolCall, tool);
  }
  private async streamLLMResponse(
    tools: ReadonlyArray<ToolSpec>,
  ): Promise<{ textContent: string; toolCalls: PendingToolCall[]; thinking: string }> {
    return streamTaskLoopLLMResponse(
      this as unknown as Parameters<typeof streamTaskLoopLLMResponse>[0],
      tools,
    );
  }
  private async executeToolCall(
    toolCall: PendingToolCall,
    availableToolNames: ReadonlySet<string>,
    availableTools: ReadonlyArray<ToolSpec>,
    batchContext: ToolExecutionBatchContext = {},
  ): Promise<ToolResult> {
    return executeTaskLoopToolCall(
      this as unknown as Parameters<typeof executeTaskLoopToolCall>[0],
      toolCall,
      availableToolNames,
      availableTools,
      batchContext,
    );
  }

  /**
   * Capture file scope discovered during readonly review workflows.
   * This survives compaction and helps prevent repeated full-diff scans.
   */
  private captureReviewScopeFiles(
    toolCall: PendingToolCall,
    originalOutput: string,
  ): void {
    captureTaskLoopReviewScopeFiles(this.sessionState, toolCall, originalOutput);
  }

}

// ─── Midpoint Briefing ──────────────────────────────────────

/** Default iterations between midpoint briefings. */
const DEFAULT_MIDPOINT_INTERVAL = 15;

function getInitialServices(options: TaskLoopOptions): TaskLoopInitialServices {
  return {
    mode: options.mode ?? "act",
    contextManager: options.contextManager ?? null,
    doubleCheck: options.doubleCheck ?? null,
    midpointCallback: options.midpointCallback ?? null,
    midpointInterval: options.config.context.midpointBriefingInterval ?? DEFAULT_MIDPOINT_INTERVAL,
    sessionState: options.sessionState ?? null,
    injectSessionStateOnFirstTurn: options.injectSessionStateOnFirstTurn ?? false,
    finalTextValidator: options.finalTextValidator ?? null,
    agentContext: options.agentContext ?? null,
  };
}

function getInitialMessages(options: TaskLoopOptions): Message[] {
  if (options.initialMessages && options.initialMessages.length > 0) {
    return [...options.initialMessages];
  }
  return [{ role: MessageRole.SYSTEM, content: options.systemPrompt }];
}

// ─── Retry Constants ─────────────────────────────────────────
// Retry logic has been moved to retry-strategy.ts.
// The retryWithStrategy function handles per-error-type retry
// with exponential backoff, jitter, and model fallback hints.

// ─── Helpers ────────────────────────────────────────────────
