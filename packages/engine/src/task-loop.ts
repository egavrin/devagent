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

import type {
  LLMProvider,
  Message,
  ToolSpec,
  StreamChunk,
  ToolResult,
  CostRecord,
  DevAgentConfig,
} from "@devagent/core";
import {
  MessageRole,
  EventBus,
  ApprovalGate,
  ProviderError,
  ContextManager,
  estimateMessageTokens,
  estimateTokens,
  lookupModelPricing,
} from "@devagent/core";
import type { MemoryStore } from "@devagent/core";
import type { ToolRegistry } from "@devagent/tools";
import type { CheckpointManager } from "./checkpoints.js";
import type { DoubleCheck } from "./double-check.js";
import { SessionState, extractEnvFact, SESSION_STATE_MARKER, PRUNED_MARKER_PREFIX, SUPERSEDED_MARKER_PREFIX, SUMMARY_MAX_CHARS } from "./session-state.js";

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
  readonly memoryStore?: MemoryStore;
  readonly checkpointManager?: CheckpointManager;
  readonly doubleCheck?: DoubleCheck;
  readonly initialMessages?: ReadonlyArray<Message>;
  /** Callback for midpoint context re-synthesis during long-running turns. */
  readonly midpointCallback?: MidpointCallback;
  /** Session state sidecar — structured facts that survive compaction. */
  readonly sessionState?: SessionState;
}

export interface TaskLoopResult {
  readonly messages: ReadonlyArray<Message>;
  readonly iterations: number;
  readonly cost: CostRecord;
  readonly aborted: boolean;
  readonly status: TaskCompletionStatus;
  readonly lastText: string | null;
}

interface PendingToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

interface NormalizedToolCall {
  readonly toolCall: PendingToolCall;
  readonly bypassResult: ToolResult | null;
  readonly scriptSteps: ToolScriptStepLike[] | null;
}

interface ToolScriptStepLike {
  readonly id: string;
  readonly tool: string;
  readonly args: Record<string, unknown>;
}

interface ToolCallBatch {
  readonly parallel: boolean;
  readonly calls: ReadonlyArray<PendingToolCall>;
}

interface ProgressSnapshot {
  readonly toolSummaries: number;
  readonly findings: number;
  readonly coverageTargets: number;
  readonly completedPlan: number;
}

// ─── Tool Output Truncation ─────────────────────────────────

/** Maximum chars for a single tool output in the message array (~12K tokens). */
const MAX_TOOL_OUTPUT_CHARS = 48_000;
/** Lines to keep from the start of a truncated tool output. */
const TRUNCATION_HEAD_LINES = 200;
/** Lines to keep from the end of a truncated tool output. */
const TRUNCATION_TAIL_LINES = 100;

/**
 * Truncate tool output using head+tail strategy (Codex pattern).
 * Shows first N lines + last N lines with a truncation marker in between.
 * Short outputs pass through unchanged.
 */
export function truncateToolOutput(output: string, maxChars: number = MAX_TOOL_OUTPUT_CHARS): string {
  if (output.length <= maxChars) return output;

  const lines = output.split("\n");
  if (lines.length <= TRUNCATION_HEAD_LINES + TRUNCATION_TAIL_LINES) {
    // Few lines but each line is very long — truncate by chars
    return output.slice(0, maxChars) + "\n\n[... output truncated ...]";
  }

  const headLines = lines.slice(0, TRUNCATION_HEAD_LINES);
  const tailLines = lines.slice(-TRUNCATION_TAIL_LINES);
  const omitted = lines.length - TRUNCATION_HEAD_LINES - TRUNCATION_TAIL_LINES;
  const marker = `\n[... ${omitted} lines truncated ...]\n`;

  const joined = [...headLines, marker, ...tailLines].join("\n");
  if (joined.length > maxChars) {
    return joined.slice(0, maxChars) + "\n\n[... output truncated ...]";
  }
  return joined;
}

// ─── Deduplication Tools ────────────────────────────────────

/** Tools whose output can be safely deduplicated (readonly, replaceable). */
const DEDUP_TOOLS = new Set(["read_file", "git_diff", "git_status"]);
/** Max reads of the same file before injecting a stagnation nudge (sessionState-independent). */
const PER_FILE_READ_LIMIT = 8;
/** Hard block on reads after this many reads of the same file. */
const PER_FILE_READ_HARD_LIMIT = 12;
/** Readonly inspection tools to pause when no-progress loops are detected.
 *  Only category:"readonly" tools belong here — the stall lock gate
 *  (normalizeToolCall) skips non-readonly categories before checking this set. */
const STALL_LOCK_TOOLS = new Set([
  "read_file",
  "git_diff",
  "git_status",
  "search_files",
  "find_files",
  "symbols",
  "diagnostics",
  "execute_tool_script",
]);

// ─── Prune Priority ─────────────────────────────────────────
/** Tool pruning priority: lower = pruned first. Tools not listed default to 1. */
const PRUNE_PRIORITY = new Map<string, number>([
  ["git_status", 0],
  ["find_files", 0],
  ["search_files", 1],
  ["run_command", 1],
  ["symbols", 1],
  ["read_file", 2],
  ["git_diff", 3],
  ["diagnostics", 3],
]);

// ─── Task Loop ──────────────────────────────────────────────

export class TaskLoop {
  private readonly provider: LLMProvider;
  private readonly tools: ToolRegistry;
  private readonly bus: EventBus;
  private readonly approvalGate: ApprovalGate;
  private readonly config: DevAgentConfig;
  private readonly systemPrompt: string;
  private readonly repoRoot: string;
  private readonly contextManager: ContextManager | null;
  private readonly memoryStore: MemoryStore | null;
  private readonly checkpointManager: CheckpointManager | null;
  private readonly doubleCheck: DoubleCheck | null;
  private readonly midpointCallback: MidpointCallback | null;
  private readonly midpointInterval: number;
  private readonly sessionState: SessionState | null;
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
  private recentToolCalls: Array<{ name: string; argsKey: string }> = [];
  private doomLoopWarned = false;
  // Per-tool consecutive failure counts (reset on any success of that tool)
  private toolFailureCounts: Map<string, number> = new Map();
  private toolFatigueWarned: Set<string> = new Set();
  private unresolvedDoubleCheckFailure = false;
  private lastReportedInputTokens = 0;
  private readonly cachedPricing;
  /** Tracks message index of the last tool result for each tool:target key (for deduplication). */
  private toolResultIndices = new Map<string, number>();
  /** Readonly tool calls that already succeeded in this run (tool + normalized args). */
  private successfulReadonlyCallKeys = new Set<string>();
  /** Whether the approaching-limit warning has been injected (reset on compaction). */
  private approachingLimitWarned = false;
  /** Consecutive readonly-only cycles with no coverage/findings progress. */
  private stagnantReadonlyCycles = 0;
  /** Last persisted progress snapshot for no-progress loop detection. */
  private lastProgressSnapshot: ProgressSnapshot | null = null;
  /** Temporarily pauses readonly inspection tools after sustained no-progress loops. */
  private readonlyStallLock = false;
  /** Per-file read count — catches stagnation even without sessionState. */
  private perFileReadCount = new Map<string, number>();
  /** Number of auto-pinned git_diff results in this session. */
  private pinnedDiffCount = 0;
  /** Iteration at which the last compaction occurred. */
  private lastCompactionIteration = 0;
  /** Re-read count since the last compaction. */
  private postCompactionRereadCount = 0;

  constructor(options: TaskLoopOptions) {
    this.provider = options.provider;
    this.tools = options.tools;
    this.bus = options.bus;
    this.approvalGate = options.approvalGate;
    this.config = options.config;
    this.systemPrompt = options.systemPrompt;
    this.repoRoot = options.repoRoot;
    this.mode = options.mode ?? "act";
    this.contextManager = options.contextManager ?? null;
    this.memoryStore = options.memoryStore ?? null;
    this.checkpointManager = options.checkpointManager ?? null;
    this.doubleCheck = options.doubleCheck ?? null;
    this.midpointCallback = options.midpointCallback ?? null;
    this.midpointInterval =
      options.config.context.midpointBriefingInterval ?? DEFAULT_MIDPOINT_INTERVAL;
    this.sessionState = options.sessionState ?? null;
    this.cachedPricing = lookupModelPricing(this.config.model);

    // Initialize messages: from previous session or fresh system prompt
    if (options.initialMessages && options.initialMessages.length > 0) {
      this.messages = [...options.initialMessages];
    } else {
      this.messages.push({
        role: MessageRole.SYSTEM,
        content: this.systemPrompt,
      });
    }
  }

  /**
   * Run the task loop with a user query.
   * Returns when the LLM produces a final text response (no more tool calls)
   * or when the budget is exceeded.
   */
  async run(userQuery: string): Promise<TaskLoopResult> {
    this.resetRunState();

    // Add user message
    this.messages.push({
      role: MessageRole.USER,
      content: userQuery,
    });
    this.bus.emit("message:user", { content: userQuery });

    let hadToolCalls = false;
    let summaryRequested = false;
    let budgetGraceUsed = false;
    let lastNonEmptyText: string | null = null;
    let textOnlyContinuations = 0;
    const MAX_TEXT_CONTINUATIONS = 3;
    let status: TaskCompletionStatus = "success";

    while (!this.aborted) {
      // Check budget (0 = unlimited)
      if (this.config.budget.maxIterations > 0 && this.iterations >= this.config.budget.maxIterations) {
        if (!budgetGraceUsed) {
          // Grace iteration: ask the model to summarize before stopping
          budgetGraceUsed = true;
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: "You have reached the iteration limit. Please provide a concise summary of your progress and findings so far. Do not use any tools — respond with text only.",
          });
          // Fall through to allow one more LLM call
        } else {
          status = "budget_exceeded";
          break;
        }
      }

      this.iterations++;
      const estimatedTokens = Math.max(
        estimateMessageTokens(this.messages),
        this.lastReportedInputTokens,
      );
      this.bus.emit("iteration:start", {
        iteration: this.iterations,
        maxIterations: this.config.budget.maxIterations,
        estimatedTokens,
        maxContextTokens: this.getEffectiveContextBudget(),
      });

      // Get available tools based on mode (no tools during grace iteration)
      const availableTools = budgetGraceUsed ? [] : this.getAvailableTools();

      // Preflight compaction: ensure context fits before calling the provider.
      await this.maybeCompactContext();

      // Stream LLM response with retry on transient provider errors
      let textContent = "";
      let toolCalls: PendingToolCall[] = [];
      let overflowCompactionUsed = false;
      for (let attempt = 0; attempt < MAX_RETRY_ATTEMPTS; attempt++) {
        try {
          const result = await this.streamLLMResponse(availableTools);
          textContent = result.textContent;
          toolCalls = result.toolCalls;
          break;
        } catch (err) {
          if (!(err instanceof ProviderError)) throw err;

          // If provider reports a context overflow, force one compaction pass and retry immediately.
          if (
            !overflowCompactionUsed &&
            this.contextManager &&
            this.isContextOverflowError(err.message)
          ) {
            overflowCompactionUsed = true;
            await this.maybeCompactContext({ force: true });
            this.bus.emit("error", {
              message: "Provider rejected prompt for context size. Forced compaction and retrying immediately.",
              code: "CONTEXT_OVERFLOW_RETRY",
              fatal: false,
            });
            continue;
          }

          if (attempt >= RETRY_DELAYS.length) throw err; // Exhausted retries
          this.bus.emit("error", {
            message: `Provider error (attempt ${attempt + 1}/${MAX_RETRY_ATTEMPTS}): ${(err as Error).message}. Retrying in ${RETRY_DELAYS[attempt]!}ms…`,
            code: "PROVIDER_RETRY",
            fatal: false,
          });
          await sleep(RETRY_DELAYS[attempt]!);
        }
      }

      // Track last non-empty text from the LLM (even if tool calls follow)
      if (textContent.trim()) {
        lastNonEmptyText = textContent;
      }

      if (toolCalls.length > 0) {
        hadToolCalls = true;
        textOnlyContinuations = 0; // Reset: LLM is making progress

        // Add single assistant message with both text and tool calls
        const mappedToolCalls = toolCalls.map((tc) => ({
          name: tc.name,
          arguments: tc.arguments,
          callId: tc.callId,
        }));
        this.messages.push({
          role: MessageRole.ASSISTANT,
          content: textContent,
          toolCalls: mappedToolCalls,
        });
        this.bus.emit("message:assistant", {
          content: textContent,
          partial: false,
          toolCalls: mappedToolCalls,
        });

        // Coalesce replace-all tool calls (e.g., multiple update_plan in one batch)
        const { toExecute, skipped } =
          this.coalesceReplaceAllCalls(toolCalls);
        for (const tc of skipped) {
          const skipContent = "Skipped: superseded by a later update_plan call in this batch.";
          this.messages.push({
            role: MessageRole.TOOL,
            content: skipContent,
            toolCallId: tc.callId,
          });
          this.bus.emit("message:tool", {
            role: "tool" as const,
            content: skipContent,
            toolCallId: tc.callId,
          });
        }

        // Execute tool calls — parallel for independent readonly, sequential for mutating.
        // Partition into batches: consecutive readonly calls form a parallel batch,
        // a mutating/workflow call is its own sequential batch.
        const availableToolNames = new Set(availableTools.map((t) => t.name));
        const batches = this.partitionToolCalls(toExecute, availableToolNames);

        for (const batch of batches) {
          if (this.aborted) break;

          if (batch.parallel) {
            // Run all calls in the batch concurrently
            const promises = batch.calls.map((tc) =>
              this.executeToolCall(tc, availableToolNames, availableTools).then((result) => ({
                callId: tc.callId,
                result,
              })),
            );
            const settled = await Promise.all(promises);

            // Append results in original order (API requires matching order)
            for (const { callId, result } of settled) {
              const tc = batch.calls.find((c) => c.callId === callId)!;
              this.appendToolResult(callId, result, tc.name, tc.arguments);
            }
          } else {
            // Sequential execution (mutating tools, or single call)
            for (const tc of batch.calls) {
              if (this.aborted) break;
              const result = await this.executeToolCall(tc, availableToolNames, availableTools);
              this.appendToolResult(tc.callId, result, tc.name, tc.arguments);
            }
          }
        }

        // Generic no-progress detection for readonly inspection loops.
        this.maybeInjectNoProgressNudge(toolCalls);

        // Doom loop detection: warn the LLM if it's repeating identical failing calls
        const doomLoopWarning = this.checkDoomLoop(toolCalls);
        if (doomLoopWarning) {
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: doomLoopWarning,
          });
        }

        // Tool fatigue detection: same tool failing repeatedly with different args
        const fatigueWarning = this.checkToolFatigue(toolCalls);
        if (fatigueWarning) {
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: fatigueWarning,
          });
        }

        // Auto-compact: check if context is approaching token budget
        await this.maybeCompactContext();

        // Proactive midpoint briefing: re-synthesize context every N iterations
        // to prevent accumulated history from degrading accuracy (paper finding)
        await this.maybeMidpointBriefing();

        // Continue loop — feed tool results back to LLM
        continue;
      }

      // No tool calls — LLM produced a text response
      if (textContent) {
        if (this.unresolvedDoubleCheckFailure) {
          this.messages.push({
            role: MessageRole.ASSISTANT,
            content: textContent,
          });
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: "Double-check still failing from prior edits. You must fix validation errors before finalizing.",
          });
          continue;
        }

        // Plan-aware continuation: if the plan has incomplete steps,
        // the LLM likely produced a "progress update" rather than a
        // final answer. Auto-continue up to MAX_TEXT_CONTINUATIONS times.
        // Also continue early in the session when no plan exists yet —
        // the LLM may outline future steps without actually doing them.
        const hasIncompleteSteps = this.sessionState?.hasPendingPlanSteps() ?? false;
        // Also continue early when no plan exists but the LLM has already
        // used tools (it was working, then produced a text-only "progress update").
        // Only when sessionState is configured (indicates a full session, not a test).
        const isEarlyNoPlan = this.sessionState != null && this.sessionState.getPlan() == null
          && this.iterations <= 5 && textOnlyContinuations === 0
          && this.iterations > 1;
        const shouldContinue = hasIncompleteSteps || isEarlyNoPlan;

        if (shouldContinue && textOnlyContinuations < MAX_TEXT_CONTINUATIONS) {
          textOnlyContinuations++;
          this.messages.push({
            role: MessageRole.ASSISTANT,
            content: textContent,
          });
          this.bus.emit("message:assistant", {
            content: textContent,
            partial: false,
          });
          const nudge = hasIncompleteSteps
            ? "Your plan has incomplete steps. Continue working — use tools to make progress on the next pending step."
            : "You outlined next steps but did not use any tools. Use update_plan to create a plan, then use tools to start working.";
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: nudge,
          });
          continue;
        }

        this.messages.push({
          role: MessageRole.ASSISTANT,
          content: textContent,
        });
        this.bus.emit("message:assistant", {
          content: textContent,
          partial: false,
        });
        status = "success";
        break;
      }

      // Empty response — try to get a summary if work was done
      if (hadToolCalls && !summaryRequested) {
        summaryRequested = true;
        this.messages.push({
          role: MessageRole.SYSTEM,
          content: "Please provide a summary of your findings and conclusions based on the work done so far.",
        });
        continue;
      }

      // Still empty after summary request — give up gracefully
      status = hadToolCalls ? "empty_response" : "success";
      break;
    }

    if (this.aborted && status === "success") {
      status = "aborted";
    }

    // Extract lessons from this session into persistent memory
    this.extractLessons();

    return {
      messages: this.messages,
      iterations: this.iterations,
      cost: this.totalCost,
      aborted: this.aborted,
      status,
      lastText: lastNonEmptyText,
    };
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
    this.recentToolCalls = [];
    this.doomLoopWarned = false;
    this.toolFailureCounts.clear();
    this.toolFatigueWarned.clear();
    this.resetRunState();
  }

  /** Reset per-run transient state (shared between run() and resetIterations()). */
  private resetRunState(): void {
    this.unresolvedDoubleCheckFailure = false;
    this.successfulReadonlyCallKeys.clear();
    this.stagnantReadonlyCycles = 0;
    this.lastProgressSnapshot = null;
    this.readonlyStallLock = false;
    this.perFileReadCount.clear();
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
    if (!this.contextManager) return;

    const maxTokens = this.getEffectiveContextBudget();
    if (maxTokens <= 0) return;

    const estimatedTokens = Math.max(
      estimateMessageTokens(this.messages),
      this.lastReportedInputTokens,
    );
    const threshold = maxTokens * this.config.context.triggerRatio;

    if (!options?.force && estimatedTokens <= threshold) {
      // Approaching-limit warning: nudge the LLM to persist findings before
      // compaction fires. Fires once at ~60% of trigger threshold to give
      // the LLM enough time to persist findings before pruning strips context.
      const warningThreshold = threshold * 0.6;
      if (!this.approachingLimitWarned && estimatedTokens > warningThreshold && this.sessionState) {
        this.approachingLimitWarned = true;
        this.messages.push({
          role: MessageRole.SYSTEM,
          content: "Context is filling up. You MUST persist any analysis conclusions or review findings NOW using save_finding. After context pruning, old tool outputs will be replaced with summaries. Do NOT re-read files already listed in session state — rely on the summaries and findings you've saved.",
        });
      }
      return;
    }

    // Pre-Phase 1: enforce pinned token budget — unpin oldest when over limit
    const PINNED_TOKEN_BUDGET = 80_000;
    let pinnedTokens = 0;
    const pinnedIndices: number[] = [];
    for (let i = 0; i < this.messages.length; i++) {
      const m = this.messages[i]!;
      if (m.pinned) {
        pinnedTokens += estimateTokens(m.content ?? "");
        pinnedIndices.push(i);
      }
    }
    if (pinnedTokens > PINNED_TOKEN_BUDGET) {
      // Unpin oldest pinned messages until within budget
      for (const idx of pinnedIndices) {
        if (pinnedTokens <= PINNED_TOKEN_BUDGET) break;
        const m = this.messages[idx]!;
        const msgTokens = estimateTokens(m.content ?? "");
        this.messages[idx] = { ...m, pinned: undefined };
        pinnedTokens -= msgTokens;
      }
    }

    // Phase 1: Try lightweight tool-output pruning before full compaction.
    // Replaces old, large tool results with compact summaries — INCREMENTALLY:
    // prunes from oldest first and stops once enough tokens are freed.
    //
    // Provider-reported tokens include untouchable overhead (tool definitions,
    // system prompt encoding, tokenizer overhead) that pruning cannot affect.
    // Compute overhead and subtract from threshold so pruning targets savings
    // based only on what it can actually remove — message content.
    const messageTokens = estimateMessageTokens(this.messages);
    const overhead = Math.max(0, this.lastReportedInputTokens - messageTokens);
    const messageThreshold = Math.max(0, threshold - overhead);
    const pruneResult = this.pruneToolOutputs(messageTokens, messageThreshold);
    if (pruneResult.savedTokens > 0) {
      this.resetPostCompactionState(false);

      // Re-inject session state so the LLM sees updated summaries
      this.injectSessionState();
      const postPruneTokens = estimateMessageTokens(this.messages);

      // Account for overhead when checking whether pruning was sufficient
      if (!options?.force && (postPruneTokens + overhead) <= threshold) {
        this.resetPostCompactionState(true);
        this.bus.emit("context:compacted", {
          removedCount: 0,
          prunedCount: pruneResult.prunedCount,
          tokensSaved: pruneResult.savedTokens,
          estimatedTokens: postPruneTokens,
          tokensBefore: estimatedTokens,
        });
        return; // Pruning was enough — skip full compaction
      }
    }

    // Phase 2: Full compaction (hybrid summarization)
    this.bus.emit("context:compacting", { estimatedTokens, maxTokens });

    try {
      const result = await this.contextManager.truncateAsync(
        this.messages,
        maxTokens,
        { force: true },
      );

      if (result.truncated) {
        this.messages = [...result.messages];
        this.injectSessionState();
        this.resetPostCompactionState(true);
        const postCompactTokens = estimateMessageTokens(this.messages);
        this.bus.emit("context:compacted", {
          removedCount: result.removedCount,
          prunedCount: pruneResult.prunedCount > 0 ? pruneResult.prunedCount : undefined,
          tokensSaved: pruneResult.savedTokens > 0 ? pruneResult.savedTokens : undefined,
          estimatedTokens: postCompactTokens,
          tokensBefore: estimatedTokens,
        });
        if (postCompactTokens > maxTokens) {
          throw new Error(
            `Compaction did not fit budget: ${postCompactTokens} > ${maxTokens}`,
          );
        }
      } else if (result.estimatedTokens > maxTokens) {
        throw new Error(
          `Compaction did not fit budget: ${result.estimatedTokens} > ${maxTokens}`,
        );
      }
    } catch (err) {
      this.bus.emit("error", {
        message: `Context compaction failed: ${(err as Error).message}`,
        code: "COMPACTION_FAILED",
        fatal: true,
      });
      throw err;
    }
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
    if (!this.sessionState) return { savedTokens: 0, prunedCount: 0 };

    const PRUNE_PROTECT_TOKENS = this.config.context.pruneProtectTokens ?? 60_000;
    const MIN_PRUNE_MSG_TOKENS = 500;

    // Target: free enough tokens to reach 85% of threshold (headroom to avoid re-triggering).
    const targetTokens = threshold * 0.85;
    const targetSavings = currentTokens - targetTokens;
    if (targetSavings <= 0) return { savedTokens: 0, prunedCount: 0 };

    // Identify protected message indices (most recent 30K tokens of tool output)
    const protectedIndices = new Set<number>();
    let protectedTokens = 0;
    for (let i = this.messages.length - 1; i >= 0; i--) {
      const msg = this.messages[i]!;
      if (msg.role !== MessageRole.TOOL) continue;
      if (protectedTokens >= PRUNE_PROTECT_TOKENS) break;
      const msgTokens = estimateTokens(msg.content ?? "");
      protectedTokens += msgTokens;
      protectedIndices.add(i);
    }

    // Pre-fetch summaries once (reversed for most-recent-first lookup)
    const reversedSummaries = [...this.sessionState.getToolSummaries()].reverse();

    // Build toolCallId → tool info index once (avoids O(candidates × messages) backward scans)
    const toolCallIndex = new Map<string, { name: string; arguments: Record<string, unknown> }>();
    for (const msg of this.messages) {
      if (msg.role === MessageRole.ASSISTANT && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          toolCallIndex.set(tc.callId, { name: tc.name, arguments: tc.arguments });
        }
      }
    }

    // Build priority-ordered candidate list instead of linear oldest-first scan.
    // Lower priority = pruned first. Within same priority, oldest first.
    const candidates: Array<{ index: number; priority: number }> = [];
    for (let i = 0; i < this.messages.length; i++) {
      const msg = this.messages[i]!;
      if (msg.role !== MessageRole.TOOL) continue;
      if (protectedIndices.has(i)) continue;
      if (msg.pinned) continue;

      const msgTokens = estimateTokens(msg.content ?? "");
      if (msgTokens <= MIN_PRUNE_MSG_TOKENS) continue;
      if (msg.content?.startsWith(PRUNED_MARKER_PREFIX) || msg.content?.startsWith(SUPERSEDED_MARKER_PREFIX)) continue;

      const toolName = msg.toolCallId ? toolCallIndex.get(msg.toolCallId)?.name : undefined;
      const priority = PRUNE_PRIORITY.get(toolName ?? "") ?? 1;
      candidates.push({ index: i, priority });
    }

    // Sort: lowest priority first, then oldest first within same priority
    candidates.sort((a, b) => a.priority - b.priority || a.index - b.index);

    let savedTokens = 0;
    let prunedCount = 0;
    for (const { index } of candidates) {
      if (savedTokens >= targetSavings) break;
      const msg = this.messages[index]!;
      const msgTokens = estimateTokens(msg.content ?? "");

      const replacement = this.buildPrunedToolPlaceholder(msg, msgTokens, reversedSummaries, toolCallIndex);
      const replacementTokens = estimateTokens(replacement);
      this.messages[index] = { ...msg, content: replacement };
      savedTokens += msgTokens - replacementTokens;
      prunedCount++;
    }

    return { savedTokens, prunedCount };
  }

  private buildPrunedToolPlaceholder(
    message: Message,
    prunedTokens: number,
    reversedSummaries: ReadonlyArray<import("./session-state.js").ToolResultSummary>,
    toolCallIndex: ReadonlyMap<string, { name: string; arguments: Record<string, unknown> }>,
  ): string {
    const fallback = `${PRUNED_MARKER_PREFIX} tool output pruned (${prunedTokens} tokens). Check session state for details.]`;

    if (!message.toolCallId) return fallback;

    const toolCall = toolCallIndex.get(message.toolCallId);
    if (!toolCall) return fallback;

    const rawTarget = (toolCall.arguments["path"] as string | undefined) ?? toolCall.name;
    const target = typeof rawTarget === "string" ? rawTarget : String(rawTarget);
    const normalizedTarget = normalizeRepoPath(target);

    const summary = reversedSummaries.find((s) => {
      if (s.tool !== toolCall.name) return false;
      const summaryTarget = normalizeRepoPath(s.target);
      return summaryTarget === normalizedTarget || s.target === target;
    });

    if (!summary) return fallback;

    const inline = `${summary.tool}(${summary.target}): ${summary.summary}`;
    const maxInlineChars = 600;
    const snippet = inline.length > maxInlineChars
      ? `${inline.slice(0, maxInlineChars - 3)}...`
      : inline;
    return `${PRUNED_MARKER_PREFIX} ${snippet} (pruned from ${prunedTokens} tokens)]`;
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
    this.lastReportedInputTokens = 0;
    this.toolResultIndices.clear();
    this.approachingLimitWarned = false;
    if (full) {
      this.successfulReadonlyCallKeys.clear();
      this.lastCompactionIteration = this.iterations;
      this.postCompactionRereadCount = 0;
    }
  }

  private injectSessionState(knownTokenEstimate?: number): void {
    if (!this.sessionState) return;

    // Determine tier based on available context headroom
    let tier: "full" | "compact" | "minimal" = "full";
    const maxBudget = this.getEffectiveContextBudget();
    if (maxBudget > 0) {
      const totalEstimate = knownTokenEstimate ?? estimateMessageTokens(this.messages);
      const headroom = maxBudget - totalEstimate;
      tier = headroom > 8000 ? "full"
        : headroom > 3000 ? "compact"
        : "minimal";
    }

    const content = this.sessionState.toSystemMessage(tier);
    if (!content) return;

    // Remove any existing session-state message
    this.messages = this.messages.filter(
      (m) => !(m.role === MessageRole.SYSTEM && m.content?.startsWith(SESSION_STATE_MARKER)),
    );

    // Insert after the first SYSTEM message (the system prompt)
    const insertIdx = this.messages[0]?.role === MessageRole.SYSTEM ? 1 : 0;
    this.messages.splice(insertIdx, 0, {
      role: MessageRole.SYSTEM,
      content,
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
   * Format a tool result, push to messages, and emit the bus event.
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
    const rawContent = result.success
      ? result.output
      : result.output
        ? `Error: ${result.error}\n\n${result.output}`
        : `Error: ${result.error}`;

    // Truncate large tool outputs (Codex head+tail pattern)
    const toolContent = truncateToolOutput(rawContent);

    // Deduplicate readonly tool results for the same target (Cline pattern)
    if (toolName && toolArgs && DEDUP_TOOLS.has(toolName)) {
      let target = (toolArgs["path"] as string | undefined) ?? toolName;
      // For git_diff, include ref/staged to avoid cross-dedup between
      // e.g. "git_diff --staged" and "git_diff" (unstaged).
      if (toolName === "git_diff") {
        const ref = toolArgs["ref"] as string | undefined;
        const staged = toolArgs["staged"] as boolean | undefined;
        target = `${target}:${ref ?? ""}:${staged ? "staged" : ""}`;
      }
      // For read_file with line ranges, include them in the dedup key to avoid
      // superseding different ranges of the same file (which causes ping-pong loops).
      if (toolName === "read_file") {
        const startLine = toolArgs["start_line"] as number | undefined;
        const endLine = toolArgs["end_line"] as number | undefined;
        if (startLine !== undefined || endLine !== undefined) {
          target = `${target}:${startLine ?? ""}:${endLine ?? ""}`;
        }
      }
      const dedupKey = `${toolName}:${target}`;
      const prevIdx = this.toolResultIndices.get(dedupKey);
      if (prevIdx !== undefined && prevIdx < this.messages.length) {
        const prevMsg = this.messages[prevIdx];
        if (prevMsg && prevMsg.role === MessageRole.TOOL) {
          this.messages[prevIdx] = {
            ...prevMsg,
            content: `${SUPERSEDED_MARKER_PREFIX} by later ${toolName}. See recent activity in session state.]`,
          };
        }
        // Check for post-compaction re-read storm
        this.checkRereadStorm(toolName, target);
      }
      // Track this result's index for future dedup
      this.toolResultIndices.set(dedupKey, this.messages.length);
    }

    // Per-file read counter: catches stagnation even without sessionState.
    // When the same file is read too many times (with any line ranges),
    // inject escalating nudges then hard-block.
    if (toolName === "read_file" && toolArgs) {
      const filePath = (toolArgs["path"] as string | undefined) ?? "";
      if (filePath) {
        const normalizedPath = normalizeRepoPath(filePath);
        const count = (this.perFileReadCount.get(normalizedPath) ?? 0) + 1;
        this.perFileReadCount.set(normalizedPath, count);
        if (count >= PER_FILE_READ_HARD_LIMIT) {
          // Hard block: replace tool content with a refusal
          this.messages.push({
            role: MessageRole.TOOL,
            content: `Blocked: "${normalizedPath}" has been read ${count} times. Use the summaries and content you already have. Do NOT attempt to read this file again.`,
            toolCallId: callId,
          });
          this.bus.emit("message:tool", {
            role: "tool" as const,
            content: `[blocked: ${normalizedPath} read limit exceeded]`,
            toolCallId: callId,
          });
          return;
        }
        if (count >= PER_FILE_READ_LIMIT) {
          this.messages.push({
            role: MessageRole.SYSTEM,
            content: `You have read "${normalizedPath}" an excessive number of reads (${count} times). You likely already have enough information from this file. Stop re-reading it and synthesize your findings from what you already know. If you need specific details, reference the content you already received.`,
          });
        }
      }
    }

    // Auto-pin git_diff results so they survive compaction
    const MAX_PINNED_DIFFS = 20;
    const shouldPin = toolName === "git_diff" && result.success && this.pinnedDiffCount < MAX_PINNED_DIFFS;
    if (shouldPin) this.pinnedDiffCount++;

    this.messages.push({
      role: MessageRole.TOOL,
      content: toolContent,
      toolCallId: callId,
      ...(shouldPin ? { pinned: true } : {}),
    });
    this.bus.emit("message:tool", {
      role: "tool" as const,
      content: toolContent,
      toolCallId: callId,
    });
  }

  private isContextOverflowError(message: string): boolean {
    const normalized = message.toLowerCase();
    return (
      normalized.includes("context length") ||
      normalized.includes("maximum context") ||
      normalized.includes("max context") ||
      normalized.includes("token limit") ||
      normalized.includes("too many tokens") ||
      normalized.includes("prompt is too long")
    );
  }

  /**
   * Detect post-compaction re-read storms: when the model re-reads the same
   * readonly targets shortly after compaction, it indicates compaction was
   * too aggressive. Emits COMPACTION_REREAD_STORM error event at 3+ re-reads.
   */
  private checkRereadStorm(toolName: string, target: string): void {
    const REREAD_STORM_WINDOW = 5;
    const REREAD_STORM_THRESHOLD = 3;

    if (this.lastCompactionIteration === 0) return;
    if (this.iterations - this.lastCompactionIteration > REREAD_STORM_WINDOW) return;

    this.postCompactionRereadCount++;
    if (this.postCompactionRereadCount === REREAD_STORM_THRESHOLD) {
      this.bus.emit("error", {
        message: `Post-compaction re-read storm detected: ${this.postCompactionRereadCount} readonly tool calls within ${REREAD_STORM_WINDOW} iterations of compaction. The model is re-fetching data that was pruned. Consider increasing keepRecentMessages or pruneProtectTokens.`,
        code: "COMPACTION_REREAD_STORM",
        fatal: false,
      });
    }
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

    const tokensBefore = estimateMessageTokens(this.messages);
    const result = await this.midpointCallback(this.messages, this.iterations);
    if (result) {
      this.messages = [...result.continueMessages];
      this.injectSessionState();
      this.resetPostCompactionState(false);
      this.bus.emit("context:compacted", {
        removedCount: 0,
        estimatedTokens: estimateMessageTokens(this.messages),
        tokensBefore,
      });
    }
  }

  private getAvailableTools(): ReadonlyArray<ToolSpec> {
    if (this.mode === "plan") {
      return this.tools.getPlanModeTools();
    }
    return this.tools.getAll();
  }

  private normalizeToolCall(
    toolCall: PendingToolCall,
    category: ToolSpec["category"],
  ): NormalizedToolCall {
    if (category !== "readonly") return { toolCall, bypassResult: null, scriptSteps: null };
    if (this.readonlyStallLock && STALL_LOCK_TOOLS.has(toolCall.name)) {
      return {
        toolCall,
        bypassResult: {
          success: true,
          output:
            `Readonly inspection paused due to repeated no-progress cycles. Stop re-running ${toolCall.name}; either persist findings and finalize, or switch to a different action that produces new evidence.`,
          error: null,
          artifacts: [],
        },
        scriptSteps: null,
      };
    }
    if (toolCall.name === "execute_tool_script") {
      return this.normalizeToolScriptCall(toolCall);
    }

    const key = buildReadonlyCallKey(toolCall.name, toolCall.arguments);
    if (!this.successfulReadonlyCallKeys.has(key)) {
      return { toolCall, bypassResult: null, scriptSteps: null };
    }

    const path = toolCall.arguments["path"];
    const target = typeof path === "string" && path.trim().length > 0
      ? normalizeRepoPath(path)
      : toolCall.name;
    const message = toolCall.name === "git_diff"
      ? `Skipped redundant git_diff for ${target}: identical diff already captured earlier in this run.`
      : `Skipped redundant readonly call ${toolCall.name}(${target}): this exact call already succeeded earlier in this run.`;
    return {
      toolCall,
      bypassResult: {
        success: true,
        output: message,
        error: null,
        artifacts: [],
      },
      scriptSteps: null,
    };
  }

  private normalizeToolScriptCall(
    toolCall: PendingToolCall,
  ): NormalizedToolCall {
    const parsedSteps = parseToolScriptStepsArg(toolCall.arguments["steps"]);
    if (!parsedSteps) return { toolCall, bypassResult: null, scriptSteps: null };
    const referencedStepIds = collectReferencedStepIds(parsedSteps);

    const dedupedSteps: ToolScriptStepLike[] = [];
    const skippedSteps: ToolScriptStepLike[] = [];
    const seenInBatch = new Set<string>();

    for (const step of parsedSteps) {
      const key = buildReadonlyCallKey(step.tool, step.args);
      // Keep any step whose ID is referenced by later interpolation.
      // Removing it would break script validation ("unknown step id").
      if (referencedStepIds.has(step.id)) {
        seenInBatch.add(key);
        dedupedSteps.push(step);
        continue;
      }
      if (this.successfulReadonlyCallKeys.has(key) || seenInBatch.has(key)) {
        skippedSteps.push(step);
        continue;
      }
      seenInBatch.add(key);
      dedupedSteps.push(step);
    }

    if (skippedSteps.length === 0) {
      return { toolCall, bypassResult: null, scriptSteps: parsedSteps };
    }

    if (dedupedSteps.length === 0) {
      return {
        toolCall,
        bypassResult: {
          success: true,
          output:
            `Skipped execute_tool_script: all ${skippedSteps.length} step(s) were already completed earlier in this run. Use session summaries and continue with new analysis.`,
          error: null,
          artifacts: [],
        },
        scriptSteps: parsedSteps,
      };
    }

    const normalizedCall: PendingToolCall = {
      ...toolCall,
      arguments: {
        ...toolCall.arguments,
        steps: JSON.stringify(dedupedSteps),
      },
    };

    return {
      toolCall: normalizedCall,
      bypassResult: null,
      scriptSteps: dedupedSteps,
    };
  }

  private collectSuccessfulScriptStepResults(
    steps: ReadonlyArray<ToolScriptStepLike>,
    scriptOutput: string,
  ): Array<{ step: ToolScriptStepLike; output: string }> {
    const sections = parseToolScriptOutputSections(scriptOutput);
    const successful: Array<{ step: ToolScriptStepLike; output: string }> = [];
    for (const step of steps) {
      const section = sections.get(step.id);
      if (!section || section.failed) continue;
      successful.push({ step, output: section.output });
    }
    return successful;
  }

  private getSummaryTarget(
    toolName: string,
    args: Record<string, unknown>,
  ): string | null {
    const path = args["path"];
    if (typeof path === "string" && path.trim().length > 0) {
      const normalized = normalizeRepoPath(path);
      // For read_file with line ranges, include them in the target so each
      // range gets its own summary slot in SessionState. Prevents overwrite
      // where reading lines 560-720 and 720-920 would clobber each other.
      if (toolName === "read_file") {
        const startLine = args["start_line"] as number | undefined;
        const endLine = args["end_line"] as number | undefined;
        if (startLine !== undefined || endLine !== undefined) {
          return `${normalized}:${startLine ?? ""}:${endLine ?? ""}`;
        }
      }
      return normalized;
    }
    if (toolName === "git_status") return "git_status";
    if (toolName === "search_files") {
      const pattern = args["pattern"];
      if (typeof pattern === "string") {
        const truncated = pattern.length > 60 ? pattern.slice(0, 57) + "..." : pattern;
        return `search:${truncated}`;
      }
    }
    if (toolName === "find_files") {
      const pattern = args["pattern"];
      if (typeof pattern === "string") {
        const truncated = pattern.length > 60 ? pattern.slice(0, 57) + "..." : pattern;
        return `find:${truncated}`;
      }
    }
    return null;
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
    // Tools where only the last call in a batch matters
    const replaceAllTools = new Set(["update_plan"]);

    // Find the last index of each replace-all tool
    const lastIndex = new Map<string, number>();
    for (let i = toolCalls.length - 1; i >= 0; i--) {
      const tc = toolCalls[i]!;
      if (replaceAllTools.has(tc.name) && !lastIndex.has(tc.name)) {
        lastIndex.set(tc.name, i);
      }
    }

    const toExecute: PendingToolCall[] = [];
    const skipped: PendingToolCall[] = [];

    for (let i = 0; i < toolCalls.length; i++) {
      const tc = toolCalls[i]!;
      if (replaceAllTools.has(tc.name) && i !== lastIndex.get(tc.name)) {
        skipped.push(tc);
      } else {
        toExecute.push(tc);
      }
    }

    return { toExecute, skipped };
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
  private partitionToolCalls(
    toolCalls: ReadonlyArray<PendingToolCall>,
    availableToolNames: ReadonlySet<string>,
  ): ToolCallBatch[] {
    const batches: ToolCallBatch[] = [];
    let currentReadonly: PendingToolCall[] = [];

    const flushReadonly = (): void => {
      if (currentReadonly.length > 0) {
        batches.push({ parallel: true, calls: currentReadonly });
        currentReadonly = [];
      }
    };

    for (const tc of toolCalls) {
      if (!availableToolNames.has(tc.name)) {
        // Unknown tool — flush readonly batch, add as sequential
        flushReadonly();
        batches.push({ parallel: false, calls: [tc] });
        continue;
      }

      const tool = this.tools.get(tc.name);
      if (tool.category === "readonly") {
        currentReadonly.push(tc);
      } else {
        // Mutating/workflow/external — flush readonly, add as sequential
        flushReadonly();
        batches.push({ parallel: false, calls: [tc] });
      }
    }

    flushReadonly();
    return batches;
  }

  private async streamLLMResponse(
    tools: ReadonlyArray<ToolSpec>,
  ): Promise<{ textContent: string; toolCalls: PendingToolCall[] }> {
    let textContent = "";
    const toolCalls: PendingToolCall[] = [];
    const pendingToolArgs = new Map<string, { name: string; chunks: string[] }>();

    const stream = this.provider.chat(this.messages, tools);

    for await (const chunk of stream) {
      switch (chunk.type) {
        case "text":
          textContent += chunk.content;
          this.bus.emit("message:assistant", {
            content: chunk.content,
            partial: true,
            chunk,
          });
          break;

        case "tool_call": {
          // Tool call comes as a single chunk with full args
          const args = parseToolArgs(chunk.content);
          toolCalls.push({
            name: chunk.toolName ?? "",
            arguments: args,
            callId: chunk.toolCallId ?? `call_${toolCalls.length}`,
          });
          break;
        }

        case "done":
          if (chunk.usage) {
            // Track actual provider-reported input tokens for compaction decisions
            this.lastReportedInputTokens = chunk.usage.promptTokens;

            // Compute cost from registry pricing (cached at construction)
            const pricing = this.cachedPricing;
            const iterationCost = pricing
              ? (chunk.usage.promptTokens * pricing.inputPricePerMillion
                + chunk.usage.completionTokens * pricing.outputPricePerMillion) / 1_000_000
              : 0;

            this.totalCost = {
              inputTokens: this.totalCost.inputTokens + chunk.usage.promptTokens,
              outputTokens: this.totalCost.outputTokens + chunk.usage.completionTokens,
              cacheReadTokens: this.totalCost.cacheReadTokens,
              cacheWriteTokens: this.totalCost.cacheWriteTokens,
              totalCost: this.totalCost.totalCost + iterationCost,
            };
            this.bus.emit("cost:update", {
              inputTokens: chunk.usage.promptTokens,
              outputTokens: chunk.usage.completionTokens,
              totalCost: iterationCost,
              model: this.config.model,
            });
          }
          break;
      }
    }

    return { textContent, toolCalls };
  }

  private async executeToolCall(
    toolCall: PendingToolCall,
    availableToolNames: ReadonlySet<string>,
    availableTools: ReadonlyArray<ToolSpec>,
  ): Promise<ToolResult> {
    const callId = toolCall.callId;

    // Check tool exists and is available in current mode
    if (!availableToolNames.has(toolCall.name)) {
      const namespaceHint = namespacedToolHint(toolCall.name, availableTools);
      return {
        success: false,
        output: "",
        error: namespaceHint
          ? `Unknown tool: ${toolCall.name}. ${namespaceHint}`
          : `Unknown tool: ${toolCall.name}`,
        artifacts: [],
      };
    }

    const tool = this.tools.get(toolCall.name);
    const normalizedCall = this.normalizeToolCall(toolCall, tool.category);
    if (normalizedCall.bypassResult) {
      this.bus.emit("tool:after", {
        name: toolCall.name,
        result: normalizedCall.bypassResult,
        callId,
        durationMs: 0,
      });
      return normalizedCall.bypassResult;
    }
    const effectiveCall = normalizedCall.toolCall;
    const scriptSteps = normalizedCall.scriptSteps;

    // Fire tool:before event
    this.bus.emit("tool:before", {
      name: effectiveCall.name,
      params: effectiveCall.arguments,
      callId,
    });

    // Check approval
    const approvalResult = await this.approvalGate.check({
      toolName: effectiveCall.name,
      toolCategory: tool.category,
      filePath: (effectiveCall.arguments["path"] as string) ?? null,
      description: `${effectiveCall.name}: ${JSON.stringify(effectiveCall.arguments).substring(0, 200)}`,
    });

    if (!approvalResult.approved) {
      const result: ToolResult = {
        success: false,
        output: "",
        error: `Tool execution denied: ${approvalResult.reason}`,
        artifacts: [],
      };
      this.bus.emit("tool:after", {
        name: effectiveCall.name,
        result,
        callId,
        durationMs: 0,
      });
      return result;
    }

    if (tool.category === "mutating") {
      // Any approved mutating execution may change the workspace, even when
      // the tool reports failure (partial writes/artifacts). Drop stale
      // readonly dedup snapshots before the mutation attempt.
      this.successfulReadonlyCallKeys.clear();
    }

    // Capture diagnostic baseline BEFORE the edit so we can filter pre-existing errors.
    // Must happen before tool.handler() to avoid TOCTOU (capturing post-edit state).
    let preEditBaseline: import("./double-check.js").DiagnosticBaseline | undefined;
    if (tool.category === "mutating" && this.doubleCheck?.isEnabled()) {
      const targetPath = effectiveCall.arguments["path"] as string | undefined;
      if (targetPath) {
        preEditBaseline = await this.doubleCheck.captureBaseline([targetPath]);
      }
    }

    // Execute tool — fail fast
    const startTime = Date.now();
    let result: ToolResult;
    try {
      result = await tool.handler(effectiveCall.arguments, {
        repoRoot: this.repoRoot,
        config: this.config,
        sessionId: "", // Filled by engine wrapper
      });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      result = {
        success: false,
        output: "",
        error: message,
        artifacts: [],
      };
    }

    const durationMs = Date.now() - startTime;

    // Track recent tool calls for doom loop + tool fatigue detection
    if (result.success) {
      // Success resets doom loop tracking — the LLM found a working approach
      this.recentToolCalls = [];
      this.doomLoopWarned = false;
      // Reset fatigue counter for this specific tool
      this.toolFailureCounts.delete(effectiveCall.name);
      this.toolFatigueWarned.delete(effectiveCall.name);
    } else {
      const argsKey = JSON.stringify(effectiveCall.arguments);
      this.recentToolCalls.push({ name: effectiveCall.name, argsKey });
      if (this.recentToolCalls.length > DOOM_LOOP_THRESHOLD) {
        this.recentToolCalls.shift();
      }
      // Increment per-tool failure count (regardless of args)
      const prevCount = this.toolFailureCounts.get(effectiveCall.name) ?? 0;
      this.toolFailureCounts.set(effectiveCall.name, prevCount + 1);
    }

    // Fire tool:after event
    this.bus.emit("tool:after", {
      name: effectiveCall.name,
      result,
      callId,
      durationMs,
    });

    // Checkpoint + double-check for successful mutating tools
    // Save original output before DoubleCheck may append validation noise.
    const originalOutput = result.output;
    const successfulScriptResults = scriptSteps && result.success
      ? this.collectSuccessfulScriptStepResults(scriptSteps, originalOutput)
      : [];

    if (result.success && tool.category === "mutating") {
      // Create checkpoint snapshot
      this.checkpointManager?.create(
        `${effectiveCall.name}: ${(effectiveCall.arguments["path"] as string) ?? ""}`,
        effectiveCall.name,
      );

      // Run double-check using the pre-edit baseline captured before tool execution.
      // For files beyond the initially predicted path (e.g., batch operations),
      // fall back to no baseline (all errors treated as new).
      if (this.doubleCheck?.isEnabled()) {
        const modifiedFiles = result.artifacts
          .filter((a): a is string => typeof a === "string");
        if (modifiedFiles.length > 0) {
          const checkResult = await this.doubleCheck.check(modifiedFiles, preEditBaseline);
          if (!checkResult.passed) {
            this.unresolvedDoubleCheckFailure = true;
            const feedback = this.doubleCheck.formatResults(checkResult);
            // Append validation errors inline with tool output (OpenCode pattern)
            result = {
              ...result,
              output: `${result.output}\n\nVALIDATION ERRORS:\n${feedback}\nFix these errors before continuing.`,
            };
          } else {
            this.unresolvedDoubleCheckFailure = false;
          }
        }
      }
    }

    // Record modified files, tool summary, and environment facts in session state.
    // Use originalOutput (before DoubleCheck noise) for the summary
    // so that post-compaction context contains useful info, not validation spam.
    // All mutations are batched into a single autosave.
    if (this.sessionState) {
      this.sessionState.batch(() => {
        const hasMutatingArtifacts = result.artifacts.length > 0 && tool.category === "mutating";
        const shouldRecord = result.success || hasMutatingArtifacts;

        if (shouldRecord) {
          // Use getSummaryTarget for the summary key so read_file line ranges
          // get unique slots (prevents overwrite in SessionState).
          const target = this.getSummaryTarget(effectiveCall.name, effectiveCall.arguments)
            ?? (effectiveCall.arguments["path"] as string | undefined)
            ?? (result.artifacts.find((a): a is string => typeof a === "string"))
            ?? effectiveCall.name;
          // Record modified file paths for mutating tools
          for (const artifact of result.artifacts) {
            if (typeof artifact === "string") {
              this.sessionState!.recordModifiedFile(artifact);
            }
          }
          // Persist readonly review scope so post-compaction turns can continue
          // without re-running broad diff discovery commands.
          this.captureReviewScopeFiles(effectiveCall, originalOutput);
          // Record tool summary for ALL recorded tools so readonly analysis
          // survives compaction and the LLM doesn't re-read the same files.
          this.sessionState!.addToolSummary({
            tool: effectiveCall.name,
            target: typeof target === "string" ? target : String(target),
            summary: this.formatToolSummary(effectiveCall, originalOutput),
            iteration: this.iterations,
          });
          if (tool.category === "readonly") {
            const coverageTarget = this.getSummaryTarget(effectiveCall.name, effectiveCall.arguments);
            if (coverageTarget) {
              this.sessionState!.recordReadonlyCoverage(effectiveCall.name, coverageTarget);
            }
          }

          // Script-level memory persistence: retain successful inner steps
          // as first-class summaries so compaction doesn't erase coverage.
          for (const stepResult of successfulScriptResults) {
            const stepTarget = this.getSummaryTarget(stepResult.step.tool, stepResult.step.args);
            if (stepTarget) {
              this.sessionState!.addToolSummary({
                tool: stepResult.step.tool,
                target: stepTarget,
                summary: this.formatToolSummary(
                  {
                    name: stepResult.step.tool,
                    arguments: stepResult.step.args,
                    callId: `script_${stepResult.step.id}`,
                  },
                  stepResult.output,
                ),
                iteration: this.iterations,
              });

              if (stepResult.step.tool === "git_diff" || stepResult.step.tool === "read_file") {
                this.sessionState!.recordModifiedFile(stepTarget);
              }
              this.sessionState!.recordReadonlyCoverage(stepResult.step.tool, stepTarget);
            }
          }
        }

        // Extract environment facts from failures
        if (!result.success) {
          const fact = extractEnvFact(effectiveCall.name, result.error ?? "", result.output);
          if (fact) {
            this.sessionState!.addEnvFact(fact.key, fact.message);
          }
        }
      });
    }

    if (result.success && tool.category === "readonly") {
      this.successfulReadonlyCallKeys.add(
        buildReadonlyCallKey(effectiveCall.name, effectiveCall.arguments),
      );
      if (scriptSteps) {
        for (const stepResult of successfulScriptResults) {
          this.successfulReadonlyCallKeys.add(
            buildReadonlyCallKey(stepResult.step.tool, stepResult.step.args),
          );
        }
      }
    }

    return result;
  }

  /**
   * Capture file scope discovered during readonly review workflows.
   * This survives compaction and helps prevent repeated full-diff scans.
   */
  private captureReviewScopeFiles(
    toolCall: PendingToolCall,
    originalOutput: string,
  ): void {
    if (!this.sessionState) return;

    if (toolCall.name === "git_diff") {
      const path = toolCall.arguments["path"];
      if (typeof path === "string" && path.trim().length > 0) {
        this.sessionState.recordModifiedFile(normalizeRepoPath(path));
      }
      return;
    }

    if (toolCall.name === "execute_tool_script") {
      const steps = parseToolScriptStepsArg(toolCall.arguments["steps"]);
      if (!steps) return;
      for (const step of steps) {
        if (step.tool !== "git_diff" && step.tool !== "read_file") continue;
        const path = step.args["path"];
        if (typeof path !== "string" || path.trim().length === 0) continue;
        this.sessionState.recordModifiedFile(normalizeRepoPath(path));
      }
      return;
    }

    if (toolCall.name !== "run_command") return;
    const command = toolCall.arguments["command"];
    if (typeof command !== "string" || !isGitDiffNameOnlyCommand(command)) return;

    for (const file of parseGitNameOnlyOutput(originalOutput)) {
      this.sessionState.recordModifiedFile(normalizeRepoPath(file));
    }
  }

  private makeProgressSnapshot(): ProgressSnapshot | null {
    if (!this.sessionState) return null;
    return {
      toolSummaries: this.sessionState.getToolSummariesCount(),
      findings: this.sessionState.getFindingsCount(),
      coverageTargets: this.sessionState.getReadonlyCoverageTargetCount(),
      completedPlan: this.sessionState.getPlanCompletedCount(),
    };
  }

  private maybeInjectNoProgressNudge(
    toolCalls: ReadonlyArray<PendingToolCall>,
  ): void {
    if (!this.sessionState || toolCalls.length === 0) return;

    const snapshot = this.makeProgressSnapshot();
    if (!snapshot) return;

    const previous = this.lastProgressSnapshot;
    this.lastProgressSnapshot = snapshot;
    if (!previous) return;

    const hasReadonly = toolCalls.some((tc) => this.tools.get(tc.name).category === "readonly");
    const hasMutating = toolCalls.some((tc) => this.tools.get(tc.name).category === "mutating");
    if (hasMutating) {
      this.stagnantReadonlyCycles = 0;
      this.readonlyStallLock = false;
      return;
    }
    if (!hasReadonly) {
      // State/agent-only batches should not reset readonly stagnation history.
      return;
    }

    const progressed = snapshot.toolSummaries > previous.toolSummaries
      || snapshot.findings > previous.findings
      || snapshot.coverageTargets > previous.coverageTargets
      || snapshot.completedPlan > previous.completedPlan;

    if (progressed) {
      this.stagnantReadonlyCycles = 0;
      this.readonlyStallLock = false;
      return;
    }

    this.stagnantReadonlyCycles++;
    const NO_PROGRESS_THRESHOLD = 5;
    if (this.stagnantReadonlyCycles < NO_PROGRESS_THRESHOLD) return;

    this.stagnantReadonlyCycles = 0;
    this.readonlyStallLock = true;
    this.bus.emit("error", {
      message: "Readonly no-progress loop detected: repeated inspections are not increasing coverage/findings.",
      code: "NO_PROGRESS_LOOP",
      fatal: false,
    });
    this.messages.push({
      role: MessageRole.SYSTEM,
      content:
        "Readonly inspections are no longer increasing coverage or findings. Stop repetitive reads/diffs. If you already have enough evidence, persist remaining issues with save_finding and finalize your response. Otherwise switch to a different tool/action that produces new evidence.",
    });
  }

  /**
   * Format a structured summary for a tool result.
   * For replace_in_file, includes search→replace details and count.
   * For other tools, falls back to truncated output.
   */
  private formatToolSummary(
    toolCall: PendingToolCall,
    originalOutput: string,
  ): string {

    if (toolCall.name === "replace_in_file") {
      const search = toolCall.arguments["search"] as string | undefined;
      const replace = toolCall.arguments["replace"] as string | undefined;
      const replacements = toolCall.arguments["replacements"] as unknown[] | undefined;
      // Extract count from output like "Replaced 4 occurrence(s)" or "Applied 3 replacement(s)"
      const countMatch = originalOutput.match(/(\d+)\s+(?:replacement|occurrence)/);
      const count = countMatch ? countMatch[1] : "?";

      if (replacements && Array.isArray(replacements)) {
        return `batch: ${replacements.length} pairs (${count} total replacements)`;
      }

      if (search && replace) {
        // Truncate search/replace to keep summary compact
        const s = search.length > 40 ? search.slice(0, 37) + "..." : search;
        const r = replace.length > 40 ? replace.slice(0, 37) + "..." : replace;
        return `'${s}' → '${r}' (${count} occurrences)`;
      }
    }

    if (toolCall.name === "write_file") {
      const path = toolCall.arguments["path"] as string | undefined;
      if (path) return `Wrote ${path}`;
    }

    // Readonly tools: compact summaries to avoid bloating session state
    if (toolCall.name === "read_file") {
      const lines = originalOutput.split("\n");
      const lineCount = lines.length;
      const startLine = toolCall.arguments["start_line"] as number | undefined;
      const endLine = toolCall.arguments["end_line"] as number | undefined;
      const rangeHint = startLine !== undefined || endLine !== undefined
        ? ` (lines ${startLine ?? 1}-${endLine ?? "end"})`
        : "";
      const digest = extractStructuralDigest(originalOutput, 1000);
      // Include content context: first and last non-blank lines for orientation
      const contentSnippets = extractContentSnippets(lines, 500);
      const parts = [`Read ${lineCount} lines${rangeHint}`];
      if (digest) parts.push(digest);
      if (contentSnippets) parts.push(contentSnippets);
      return parts.join(": ");
    }

    if (toolCall.name === "search_files") {
      return this.formatSearchFilesSummary(toolCall, originalOutput);
    }

    if (toolCall.name === "find_files") {
      return this.formatFindFilesSummary(toolCall, originalOutput);
    }

    if (toolCall.name === "git_diff") {
      return summarizeDiff(originalOutput);
    }

    if (toolCall.name === "git_status") {
      return formatGitStatusSummary(originalOutput);
    }

    if (toolCall.name === "run_command") {
      return this.formatRunCommandSummary(toolCall, originalOutput);
    }

    if (toolCall.name === "diagnostics") {
      return formatDiagnosticsSummary(originalOutput);
    }

    if (toolCall.name === "symbols") {
      return formatSymbolsSummary(originalOutput);
    }

    // Default: truncated original output (no DoubleCheck noise)
    return originalOutput.slice(0, SUMMARY_MAX_CHARS);
  }

  /**
   * Format search_files summary preserving pattern, file paths, and match lines.
   */
  private formatSearchFilesSummary(
    toolCall: PendingToolCall,
    originalOutput: string,
  ): string {
    const pattern = toolCall.arguments["pattern"] as string | undefined;
    const lines = originalOutput.split("\n");
    const nonEmpty = lines.filter((l) => l.trim());

    // Extract header line (e.g., "Found 15 matches for ...")
    const headerMatch = originalOutput.match(/^(Found \d+ match[^\n]*)/);
    const header = headerMatch
      ? headerMatch[1]
      : `${nonEmpty.length} matches for "${pattern ?? "?"}"`;

    // Collect file paths and match lines
    const fileLines: string[] = [];
    const matchLines: string[] = [];
    for (const line of nonEmpty) {
      const trimmed = line.trim();
      if (trimmed.startsWith("Found ")) continue;
      // Match lines typically start with whitespace + line number
      if (/^\s+\d+:/.test(line)) {
        matchLines.push(trimmed);
      } else if (trimmed.length > 0 && !trimmed.startsWith("---")) {
        fileLines.push(trimmed);
      }
    }

    const parts = [header];
    if (fileLines.length > 0) {
      parts.push(`Files: ${fileLines.join(", ")}`);
    }
    for (const ml of matchLines) {
      parts.push(ml);
    }

    return truncateToSummary(parts.join("\n"));
  }

  /**
   * Format find_files summary preserving glob pattern and file paths.
   */
  private formatFindFilesSummary(
    toolCall: PendingToolCall,
    originalOutput: string,
  ): string {
    const pattern = toolCall.arguments["pattern"] as string | undefined;
    const lines = originalOutput.split("\n").filter((l) => l.trim());

    // Extract header (e.g., "Found 12 files matching ...")
    const headerMatch = originalOutput.match(/^(Found \d+ file[^\n]*)/);
    const filePaths = lines.filter((l) => !l.startsWith("Found "));
    const header = headerMatch
      ? headerMatch[1]
      : `${filePaths.length} files matching "${pattern ?? "?"}"`;

    const parts = [header, ...filePaths];
    return truncateToSummary(parts.join("\n"));
  }

  /**
   * Format run_command summary with head+tail and test output extraction.
   */
  private formatRunCommandSummary(
    toolCall: PendingToolCall,
    originalOutput: string,
  ): string {
    const cmd = toolCall.arguments["command"] as string | undefined;

    // Special case: git diff
    if (cmd && /\bgit\s+diff\b/.test(cmd)) {
      return summarizeDiff(originalOutput);
    }

    // Special case: test/typecheck/lint commands
    if (cmd && /\b(?:test|vitest|jest|mocha|pytest|typecheck|tsc|lint|eslint|biome)\b/.test(cmd)) {
      const testSummary = summarizeTestOutput(originalOutput);
      if (testSummary) {
        const prefix = `$ ${cmd}\n`;
        return truncateToSummary(prefix + testSummary);
      }
    }

    // General case: head+tail with command prefix
    const lines = originalOutput.split("\n");
    const prefix = cmd ? `$ ${cmd}\n` : "";

    if (lines.length <= 10) {
      // Short output: keep it all
      return truncateToSummary(prefix + originalOutput);
    }

    // Head (first 5 lines) + tail (last 3 lines)
    const head = lines.slice(0, 5).join("\n");
    const tail = lines.slice(-3).join("\n");
    const omitted = lines.length - 8;
    return truncateToSummary(`${prefix}${head}\n[... ${omitted} lines omitted ...]\n${tail}`);
  }

  /**
   * Doom loop detection: check if the LLM keeps calling the same tool
   * with identical arguments and it keeps failing.
   * Returns a warning message to inject, or null if no doom loop detected.
   *
   * Following the OpenCode pattern (DOOM_LOOP_THRESHOLD = 3):
   * - Does NOT kill the loop — the LLM gets to try a different approach
   * - Warning is injected once per doom loop pattern
   * - Resets when any tool call succeeds (see executeToolCall)
   */
  private checkDoomLoop(toolCalls: ReadonlyArray<PendingToolCall>): string | null {
    if (this.recentToolCalls.length < DOOM_LOOP_THRESHOLD) return null;

    // Check if all recent calls are identical (same tool + same args)
    const first = this.recentToolCalls[0]!;
    const isDoomLoop = this.recentToolCalls.every(
      (tc) => tc.name === first.name && tc.argsKey === first.argsKey,
    );

    if (!isDoomLoop) return null;
    if (this.doomLoopWarned) return null; // Only warn once per pattern

    this.doomLoopWarned = true;
    const toolName = first.name;

    this.bus.emit("error", {
      message: `Doom loop detected: "${toolName}" called ${DOOM_LOOP_THRESHOLD} times with identical arguments and keeps failing.`,
      code: "DOOM_LOOP",
      fatal: false,
    });

    return `WARNING: You have called "${toolName}" ${DOOM_LOOP_THRESHOLD} times with the exact same arguments, and it keeps failing. This approach is not working. Try a completely different strategy — change the command arguments, use a different tool, or modify your approach. Do NOT repeat the same failing call.`;
  }

  /**
   * Tool fatigue detection: same tool keeps failing with different arguments.
   * Complementary to doom loop (which catches identical args).
   * Returns an escalated warning, or null.
   */
  private checkToolFatigue(
    toolCalls: ReadonlyArray<PendingToolCall>,
  ): string | null {
    for (const tc of toolCalls) {
      const count = this.toolFailureCounts.get(tc.name) ?? 0;
      if (
        count >= TOOL_FATIGUE_THRESHOLD &&
        !this.toolFatigueWarned.has(tc.name)
      ) {
        this.toolFatigueWarned.add(tc.name);

        this.bus.emit("error", {
          message: `Tool fatigue: "${tc.name}" has failed ${count} times consecutively with different arguments.`,
          code: "TOOL_FATIGUE",
          fatal: false,
        });

        return `ESCALATED WARNING: The tool "${tc.name}" has failed ${count} consecutive times, even with different arguments. This tool is not working for the current task. You MUST try a fundamentally different approach: use a different tool, break the problem into smaller steps, or ask the user for guidance. Do NOT call "${tc.name}" again unless you have resolved the underlying issue.`;
      }
    }
    return null;
  }

  /**
   * Extract lessons from this session into persistent memory.
   * Heuristic-based — no extra LLM call. Called once at the end of run().
   */
  private extractLessons(): void {
    if (!this.memoryStore) return;

    // NOTE: Decay + prune now run at startup via runMaintenance(), not per-turn.

    // Extract lessons from doom loop warnings
    const doomMessages = this.messages.filter(
      (m) =>
        m.role === MessageRole.SYSTEM &&
        m.content?.includes("same arguments"),
    );
    for (const msg of doomMessages) {
      const match = msg.content?.match(/called "([^"]+)"/);
      if (match?.[1]) {
        this.memoryStore.store(
          "mistake",
          `doom-loop-${match[1]}`,
          `Tool "${match[1]}" was called repeatedly with identical failing arguments. Try different approaches or verify preconditions first.`,
          { tags: ["doom-loop", match[1]] },
        );
      }
    }

    // Extract lessons from budget exhaustion
    if (this.messages.some(
      (m) => m.role === MessageRole.SYSTEM && m.content?.includes("iteration limit"),
    )) {
      this.memoryStore.store(
        "mistake",
        "budget-exhausted",
        "Session hit the iteration limit. Break complex tasks into smaller steps and verify progress frequently.",
        { tags: ["budget-exhausted"] },
      );
    }
  }
}

// ─── Doom Loop Detection ─────────────────────────────────────

/** Number of identical failing tool calls (same name + same args) that triggers a doom loop warning. */
const DOOM_LOOP_THRESHOLD = 3;

/** Number of consecutive failures of the same tool (regardless of args) that triggers a "tool fatigue" warning. */
const TOOL_FATIGUE_THRESHOLD = 5;

// ─── Midpoint Briefing ──────────────────────────────────────

/** Default iterations between midpoint briefing checkpoints. */
const DEFAULT_MIDPOINT_INTERVAL = 15;

// ─── Retry Constants ─────────────────────────────────────────

/** Delay (ms) before each retry attempt. Length = max retries. */
const RETRY_DELAYS = [300, 900, 1800] as const;

/** Total attempts = 1 initial + retries. */
const MAX_RETRY_ATTEMPTS = RETRY_DELAYS.length + 1;

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

// ─── Tool Summary Formatters (free functions) ────────────────

/** Truncate text to SUMMARY_MAX_CHARS with trailing "..." when exceeded. */
function truncateToSummary(text: string): string {
  return text.length <= SUMMARY_MAX_CHARS
    ? text
    : text.slice(0, SUMMARY_MAX_CHARS - 3) + "...";
}

/**
 * Format git_status output into grouped status summary.
 * Groups files by status code (M, A, D, ?, etc.) for compact display.
 */
function formatGitStatusSummary(output: string): string {
  const lines = output.split("\n").filter((l) => l.trim());
  const groups = new Map<string, string[]>();

  for (const line of lines) {
    const trimmed = line.trim();
    // git status --porcelain format: XY filename
    const match = trimmed.match(/^([MADRCU?! ]{1,2})\s+(.+)$/);
    if (match) {
      const statusCode = match[1]!.trim() || "M";
      const fileName = match[2]!.split("/").pop() ?? match[2]!;
      const key = statusCode.startsWith("?") ? "?" : statusCode.charAt(0);
      const existing = groups.get(key) ?? [];
      existing.push(fileName);
      groups.set(key, existing);
    }
  }

  if (groups.size === 0) {
    return `${lines.length} entries`;
  }

  const parts = [`${lines.length} entries`];
  for (const [status, files] of groups) {
    parts.push(`[${status}] ${files.join(", ")}`);
  }

  return truncateToSummary(parts.join("\n"));
}

/**
 * Format diagnostics output with severity counts and diagnostic lines.
 * Prioritizes errors over warnings.
 */
function formatDiagnosticsSummary(output: string): string {
  const lines = output.split("\n").filter((l) => l.trim());
  let errorCount = 0;
  let warningCount = 0;
  const errorLines: string[] = [];
  const warningLines: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (/\berror\b/i.test(trimmed)) {
      errorCount++;
      errorLines.push(trimmed);
    } else if (/\bwarning\b/i.test(trimmed)) {
      warningCount++;
      warningLines.push(trimmed);
    }
  }

  const total = errorCount + warningCount;
  if (total === 0) {
    return output.slice(0, SUMMARY_MAX_CHARS);
  }

  const countParts: string[] = [];
  if (errorCount > 0) countParts.push(`${errorCount} errors`);
  if (warningCount > 0) countParts.push(`${warningCount} warnings`);
  const header = `${total} diagnostics (${countParts.join(", ")})`;

  // Include all diagnostic lines, errors first
  const allDiagLines = [...errorLines, ...warningLines];
  const parts = [header, ...allDiagLines];
  return truncateToSummary(parts.join("\n"));
}

/**
 * Format symbols output preserving the symbol list.
 * Symbols are compact (~50 chars each), so 2000 chars fits ~35 symbols.
 */
function formatSymbolsSummary(output: string): string {
  const lines = output.split("\n").filter((l) => l.trim());
  const header = `${lines.length} symbols`;
  const parts = [header, ...lines];
  return truncateToSummary(parts.join("\n"));
}

/**
 * Extract structured summary from test/typecheck/lint output.
 * Returns pass/fail counts, error lines, and failing test names.
 * Returns null if the output doesn't look like test/typecheck output.
 */
export function summarizeTestOutput(output: string): string | null {
  const parts: string[] = [];
  let isTestOutput = false;

  // Vitest/Jest pass/fail counts
  const testCountMatch = output.match(/Tests?:\s*(.+total)/i);
  if (testCountMatch) {
    parts.push(testCountMatch[0]);
    isTestOutput = true;
  }

  // Bun test counts
  const bunTestMatch = output.match(/(\d+)\s+pass(?:ed)?.*?(\d+)\s+fail/i);
  if (!testCountMatch && bunTestMatch) {
    parts.push(`${bunTestMatch[1]} passed, ${bunTestMatch[2]} failed`);
    isTestOutput = true;
  }

  // TypeScript error count
  const tsErrorMatch = output.match(/Found (\d+) errors? in (\d+) files?\./);
  if (tsErrorMatch) {
    parts.push(`${tsErrorMatch[1]} errors in ${tsErrorMatch[2]} files`);
    isTestOutput = true;
  }

  // Individual TS errors (e.g., "error TS2322:")
  const tsErrors: string[] = [];
  for (const line of output.split("\n")) {
    const trimmed = line.trim();
    if (/error TS\d+/.test(trimmed)) {
      tsErrors.push(trimmed);
    }
  }
  if (tsErrors.length > 0) {
    isTestOutput = true;
    for (const e of tsErrors.slice(0, 10)) {
      parts.push(e);
    }
    if (tsErrors.length > 10) {
      parts.push(`... +${tsErrors.length - 10} more errors`);
    }
  }

  // Failing test files (FAIL lines)
  const failLines: string[] = [];
  for (const line of output.split("\n")) {
    const trimmed = line.trim();
    if (/^FAIL\s/.test(trimmed)) {
      failLines.push(trimmed);
    }
  }
  if (failLines.length > 0) {
    isTestOutput = true;
    for (const f of failLines.slice(0, 5)) {
      parts.push(f);
    }
  }

  // Error assertion lines (● test name)
  const assertionLines: string[] = [];
  for (const line of output.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("●") || trimmed.startsWith("✕") || trimmed.startsWith("×")) {
      assertionLines.push(trimmed);
    }
  }
  if (assertionLines.length > 0) {
    for (const a of assertionLines.slice(0, 5)) {
      parts.push(a);
    }
  }

  if (!isTestOutput) return null;

  return truncateToSummary(parts.join("\n"));
}

// ─── Structural Digest ───────────────────────────────────────

const STRUCTURAL_LINE_PATTERNS: ReadonlyArray<RegExp> = [
  /^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+[A-Za-z_]\w*\s*\(/,
  /^(?:export\s+)?(?:abstract\s+)?class\s+[A-Za-z_]\w*/,
  /^(?:export\s+)?interface\s+[A-Za-z_]\w*/,
  /^(?:export\s+)?type\s+[A-Za-z_]\w*\s*=/,
  /^(?:export\s+)?enum\s+[A-Za-z_]\w*/,
  /^(?:export\s+)?const\s+[A-Za-z_]\w*\s*=\s*(?:async\s*)?(?:\(|<)/,
  /^def\s+[A-Za-z_]\w*\s*\(/,
  /^class\s+[A-Za-z_]\w*(?:\(|:|\s|$)/,
  /^(?:pub\s+)?fn\s+[A-Za-z_]\w*\s*\(/,
  /^(?:pub\s+)?struct\s+[A-Za-z_]\w*/,
  /^impl\s+[A-Za-z_]\w*/,
  /^func\s+[A-Za-z_]\w*\s*\(/,
];

/**
 * Extract a compact structural digest from source text.
 * Captures top-level declaration-like lines so summaries retain
 * semantic anchors after compaction.
 */
export function extractStructuralDigest(
  source: string,
  maxChars: number = 500,
): string {
  if (maxChars <= 0) return "";

  const declarations: string[] = [];
  const seen = new Set<string>();

  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (
      trimmed.startsWith("//")
      || trimmed.startsWith("#")
      || trimmed.startsWith("/*")
      || trimmed.startsWith("*")
    ) {
      continue;
    }
    if (!STRUCTURAL_LINE_PATTERNS.some((pattern) => pattern.test(trimmed))) continue;

    const normalized = trimmed
      .replace(/\s*\{\s*$/, "")
      .replace(/\s+/g, " ");
    const snippet = normalized.length > 80
      ? `${normalized.slice(0, 77)}...`
      : normalized;
    if (seen.has(snippet)) continue;
    seen.add(snippet);
    declarations.push(snippet);
    if (declarations.length >= 20) break;
  }

  if (declarations.length === 0) return "";

  const joined = declarations.join("; ");
  if (joined.length <= maxChars) return joined;
  if (maxChars <= 3) return joined.slice(0, maxChars);
  return `${joined.slice(0, maxChars - 3)}...`;
}

/**
 * Extract a few meaningful non-blank lines from source content for summary context.
 * Captures first 2 and last 2 non-blank, non-comment lines so the LLM
 * retains orientation cues (variable assignments, key logic) beyond just declarations.
 */
function extractContentSnippets(lines: string[], maxChars: number): string {
  const meaningful: string[] = [];
  for (const line of lines) {
    const trimmed = line.replace(/^\d+\t/, "").trim(); // strip line number prefix
    if (!trimmed) continue;
    if (trimmed.startsWith("//") || trimmed.startsWith("#") || trimmed.startsWith("/*") || trimmed.startsWith("*")) continue;
    if (trimmed === "{" || trimmed === "}" || trimmed === ");") continue;
    meaningful.push(trimmed);
  }
  if (meaningful.length === 0) return "";

  const snippets: string[] = [];
  const firstFour = meaningful.slice(0, 4);
  const lastFour = meaningful.length > 8 ? meaningful.slice(-4) : [];
  for (const s of [...firstFour, ...lastFour]) {
    const truncated = s.length > 100 ? s.slice(0, 97) + "..." : s;
    snippets.push(truncated);
  }

  const joined = `[${snippets.join(" | ")}]`;
  return joined.length <= maxChars ? joined : joined.slice(0, maxChars - 3) + "...";
}

// ─── Diff Summary ────────────────────────────────────────────

/**
 * Extract a compact semantic summary from unified diff output.
 * Parses hunk headers to identify modified functions/methods and
 * counts additions/deletions. Returns a human-readable summary
 * that captures WHAT changed, not just how many lines.
 */
export function summarizeDiff(diffOutput: string): string {
  // Check for diffstat first (e.g., "3 files changed, 10 insertions(+), 5 deletions(-)")
  const statMatch = diffOutput.match(/(\d+)\s+files?\s+changed/);
  if (statMatch) {
    const insertions = diffOutput.match(/(\d+)\s+insertions?\(\+\)/);
    const deletions = diffOutput.match(/(\d+)\s+deletions?\(-\)/);
    const parts = [`${statMatch[1]} files changed`];
    if (insertions) parts.push(`+${insertions[1]}`);
    if (deletions) parts.push(`-${deletions[1]}`);
    return parts.join(", ");
  }

  // Parse unified diff: extract hunk headers and count changes
  const lines = diffOutput.split("\n");
  const hunks: Array<{ context: string; added: number; removed: number }> = [];
  let currentHunk: { context: string; added: number; removed: number } | null = null;

  for (const line of lines) {
    // Hunk header: @@ -a,b +c,d @@ optional context
    const hunkMatch = line.match(/^@@\s+[^@]+@@\s*(.*)$/);
    if (hunkMatch) {
      if (currentHunk) hunks.push(currentHunk);
      currentHunk = {
        context: hunkMatch[1]?.trim() ?? "",
        added: 0,
        removed: 0,
      };
      continue;
    }

    if (currentHunk) {
      if (line.startsWith("+") && !line.startsWith("+++")) {
        currentHunk.added++;
      } else if (line.startsWith("-") && !line.startsWith("---")) {
        currentHunk.removed++;
      }
    }
  }
  if (currentHunk) hunks.push(currentHunk);

  if (hunks.length === 0) {
    const lineCount = lines.length;
    return `diff: ${lineCount} lines`;
  }

  // Build summary from hunks — take up to 4 most significant
  const sorted = [...hunks].sort((a, b) =>
    (b.added + b.removed) - (a.added + a.removed),
  );
  const top = sorted.slice(0, 4);
  const totalAdded = hunks.reduce((s, h) => s + h.added, 0);
  const totalRemoved = hunks.reduce((s, h) => s + h.removed, 0);

  const parts: string[] = [];
  for (const h of top) {
    if (h.context) {
      parts.push(`${h.context} +${h.added}/-${h.removed}`);
    }
  }

  if (parts.length > 0) {
    const extra = hunks.length > 4 ? ` (+${hunks.length - 4} more)` : "";
    return `${parts.join("; ")}${extra} [total +${totalAdded}/-${totalRemoved}]`;
  }

  return `${hunks.length} hunks, +${totalAdded}/-${totalRemoved}`;
}

// ─── Helpers ────────────────────────────────────────────────

function parseToolArgs(content: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(content);
    // JSON.parse can return non-objects (null, arrays, primitives) — wrap them
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { raw: content };
    }
    return parsed as Record<string, unknown>;
  } catch {
    // Fail fast: surface the malformed JSON so the LLM can see the real error
    // instead of silently dropping structured arguments.
    return { _parseError: `Malformed tool arguments (invalid JSON): ${content.substring(0, 200)}` };
  }
}

function namespacedToolHint(
  toolName: string,
  availableTools: ReadonlyArray<ToolSpec>,
): string | null {
  if (!hasDisallowedToolPrefix(toolName)) return null;

  const canonical = toolName.replace(/^(?:functions|function|tools)\./, "");
  if (availableTools.some((t) => t.name === canonical)) {
    return `Use canonical tool names only. Try "${canonical}" (without namespace prefixes).`;
  }

  return "Use canonical tool names only (no prefixes like functions./function./tools.).";
}

function hasDisallowedToolPrefix(toolName: string): boolean {
  return /^(?:functions|function|tools)\./.test(toolName);
}

function isGitDiffNameOnlyCommand(command: string): boolean {
  return /\bgit\s+diff\b/.test(command) && /\b--name-only\b/.test(command);
}

function parseGitNameOnlyOutput(output: string): string[] {
  return output
    .split(/\r?\n/)
    .map((line) => normalizeRepoPath(line))
    .filter((line) => line.length > 0)
    .filter((line) => !line.startsWith("fatal:"))
    .filter((line) => !line.startsWith("warning:"));
}

function normalizeRepoPath(filePath: string): string {
  const normalized = filePath.trim().replaceAll("\\", "/");
  return normalized.startsWith("./") ? normalized.slice(2) : normalized;
}

function buildReadonlyCallKey(
  toolName: string,
  args: Record<string, unknown>,
): string {
  const normalizedArgs = normalizeArgsForReadonlyKey(args);
  return `${toolName}:${JSON.stringify(normalizedArgs)}`;
}

function normalizeArgsForReadonlyKey(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map((item) => normalizeArgsForReadonlyKey(item));
  }
  if (value && typeof value === "object") {
    const input = value as Record<string, unknown>;
    const out: Record<string, unknown> = {};
    for (const key of Object.keys(input).sort()) {
      const v = input[key];
      if (typeof v === "string" && key === "path") {
        out[key] = normalizeRepoPath(v);
      } else {
        out[key] = normalizeArgsForReadonlyKey(v);
      }
    }
    return out;
  }
  return value;
}

function parseToolScriptStepsArg(raw: unknown): ToolScriptStepLike[] | null {
  if (typeof raw !== "string") return null;
  try {
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return null;
    const steps: ToolScriptStepLike[] = [];
    for (const entry of parsed) {
      if (!entry || typeof entry !== "object") return null;
      const step = entry as Record<string, unknown>;
      const id = step["id"];
      const tool = step["tool"];
      const args = step["args"];
      if (typeof id !== "string" || typeof tool !== "string" || !args || typeof args !== "object") {
        return null;
      }
      steps.push({
        id,
        tool,
        args: { ...(args as Record<string, unknown>) },
      });
    }
    return steps;
  } catch {
    return null;
  }
}

function collectReferencedStepIds(
  steps: ReadonlyArray<ToolScriptStepLike>,
): Set<string> {
  const referenced = new Set<string>();
  for (const step of steps) {
    collectStepIdsFromValue(step.args, referenced);
  }
  return referenced;
}

function collectStepIdsFromValue(
  value: unknown,
  out: Set<string>,
): void {
  if (typeof value === "string") {
    const refPattern = /\$([a-zA-Z_][a-zA-Z0-9_]*)(?:\.lines\[(\d+)\])?/g;
    for (const match of value.matchAll(refPattern)) {
      const stepId = match[1];
      if (stepId) out.add(stepId);
    }
    return;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      collectStepIdsFromValue(item, out);
    }
    return;
  }

  if (value && typeof value === "object") {
    for (const child of Object.values(value as Record<string, unknown>)) {
      collectStepIdsFromValue(child, out);
    }
  }
}

function parseToolScriptOutputSections(
  output: string,
): Map<string, { tool: string; failed: boolean; output: string }> {
  const sections = new Map<string, { tool: string; failed: boolean; output: string }>();
  const headerPattern = /^=== Step (\S+) \(([^)]+)\) \[(FAILED|\d+ms)\] ===$/gm;
  const headers = [...output.matchAll(headerPattern)];
  for (let i = 0; i < headers.length; i++) {
    const current = headers[i]!;
    const next = headers[i + 1];
    const id = current[1]!;
    const tool = current[2]!;
    const failed = current[3] === "FAILED";
    const headerEnd = (current.index ?? 0) + current[0].length;
    const tailEnd = next?.index
      ?? output.indexOf("\n\n[Script completed:", headerEnd);
    const segmentEnd = tailEnd === -1 ? output.length : tailEnd;
    const content = output.slice(headerEnd, segmentEnd).trim();
    sections.set(id, {
      tool,
      failed,
      output: content.startsWith("Error: ") ? content.slice(7).trim() : content,
    });
  }
  return sections;
}
