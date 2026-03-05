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
  extractErrorMessage,
} from "@devagent/core";
import type { MemoryStore } from "@devagent/core";
import type { ToolRegistry } from "@devagent/tools";
import type { CheckpointManager } from "./checkpoints.js";
import type { DoubleCheck } from "./double-check.js";
import { SessionState, extractEnvFact, SESSION_STATE_MARKER, PRUNED_MARKER_PREFIX, SUPERSEDED_MARKER_PREFIX } from "./session-state.js";
import { formatToolSummary } from "./tool-summary-formatter.js";

// Re-export formatter functions that were previously defined here,
// preserving backward compatibility for external consumers.
export { summarizeDiff, summarizeTestOutput, extractStructuralDigest } from "./tool-summary-formatter.js";
import { StagnationDetector } from "./stagnation-detector.js";
import { judgeCompactionQuality, buildPreCompactionSummary } from "./compaction-judge.js";
import { judgeCompletion } from "./completion-judge.js";
import { parseToolScriptStepsArg } from "./tool-script.js";
import type { ToolScriptStep } from "./tool-script.js";

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
  readonly scriptSteps: ToolScriptStep[] | null;
}

interface ToolCallBatch {
  readonly parallel: boolean;
  readonly calls: ReadonlyArray<PendingToolCall>;
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
// ─── Context & Pruning Thresholds ───────────────────────────

/** Token budget for pinned messages — oldest unpinned when exceeded. */
const PINNED_TOKEN_BUDGET = 80_000;
/** Minimum token size for a tool message to be eligible for pruning.
 * At 5000 tokens, net savings after ~150-token summary replacement is ~4850 tokens.
 * Below this threshold, the context loss outweighs the negligible savings. */
const MIN_PRUNE_MSG_TOKENS = 5_000;
/** After pruning, target 75% of threshold to leave larger headroom and reduce Phase 2 triggers. */
const PRUNE_THRESHOLD_RATIO = 0.75;
/** Warning fires at 60% of compaction threshold so the LLM can persist findings. */
const APPROACHING_LIMIT_RATIO = 0.6;
/** Max chars for an inline pruned-tool summary placeholder. */
const MAX_INLINE_SUMMARY_CHARS = 600;
/** Max auto-pinned git_diff results per session. */
const MAX_PINNED_DIFFS = 20;
/** Max text-only continuations before the loop accepts a text-only response. */
const MAX_TEXT_CONTINUATIONS = 3;

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
  private readonly stagnationDetector: StagnationDetector;
  private unresolvedDoubleCheckFailure = false;
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
    this.stagnationDetector = new StagnationDetector({
      bus: this.bus,
      sessionState: this.sessionState,
    });

    // Initialize messages: from previous session or fresh system prompt
    if (options.initialMessages && options.initialMessages.length > 0) {
      this.messages = [...options.initialMessages];
    } else {
      this.messages.push({
        role: MessageRole.SYSTEM,
        content: this.systemPrompt,
      });
    }

    // Initialize running token counter from initial messages.
    this.estimatedTokens = estimateMessageTokens(this.messages);
  }

  /**
   * Run the task loop with a user query.
   * Returns when the LLM produces a final text response (no more tool calls)
   * or when the budget is exceeded.
   */
  async run(userQuery: string): Promise<TaskLoopResult> {
    this.resetRunState();

    // Add user message
    this.pushMessage({
      role: MessageRole.USER,
      content: userQuery,
    });
    this.bus.emit("message:user", { content: userQuery });

    let hadToolCalls = false;
    let summaryRequested = false;
    let budgetGraceUsed = false;
    let lastNonEmptyText: string | null = null;
    let textOnlyContinuations = 0;
    let status: TaskCompletionStatus = "success";

    while (!this.aborted) {
      // Check budget (0 = unlimited)
      if (this.config.budget.maxIterations > 0 && this.iterations >= this.config.budget.maxIterations) {
        if (!budgetGraceUsed) {
          // Grace iteration: ask the model to summarize before stopping
          budgetGraceUsed = true;
          this.pushMessage({
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
      const iterationTokenEstimate = Math.max(
        this.estimatedTokens,
        this.lastReportedInputTokens,
      );
      this.bus.emit("iteration:start", {
        iteration: this.iterations,
        maxIterations: this.config.budget.maxIterations,
        estimatedTokens: iterationTokenEstimate,
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
        this.pushMessage({
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
          this.pushMessage({
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
              await this.maybeClassifyError(result, tc.name, tc.arguments);
              this.appendToolResult(callId, result, tc.name, tc.arguments);
            }
          } else {
            // Sequential execution (mutating tools, or single call)
            for (const tc of batch.calls) {
              if (this.aborted) break;
              const result = await this.executeToolCall(tc, availableToolNames, availableTools);
              await this.maybeClassifyError(result, tc.name, tc.arguments);
              this.appendToolResult(tc.callId, result, tc.name, tc.arguments);
            }
          }
        }

        // LLM-as-judge stagnation check: periodic review of conversation history.
        const stagnationNudge = await this.stagnationDetector.checkStagnationWithLLM(
          this.provider, this.messages, this.iterations,
        );
        if (stagnationNudge) {
          this.pushMessage({ role: MessageRole.SYSTEM, content: stagnationNudge });
        }

        // Doom loop detection: warn the LLM if it's repeating identical failing calls
        const doomLoopWarning = this.stagnationDetector.checkDoomLoop(toolCalls);
        if (doomLoopWarning) {
          this.pushMessage({
            role: MessageRole.SYSTEM,
            content: doomLoopWarning,
          });
        }

        // Tool fatigue detection: same tool failing repeatedly with different args
        const fatigueWarning = this.stagnationDetector.checkToolFatigue(toolCalls);
        if (fatigueWarning) {
          this.pushMessage({
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
          this.pushMessage({
            role: MessageRole.ASSISTANT,
            content: textContent,
          });
          this.pushMessage({
            role: MessageRole.SYSTEM,
            content: "Double-check still failing from prior edits. You must fix validation errors before finalizing.",
          });
          continue;
        }

        // Plan-aware continuation: if the plan has incomplete steps,
        // the LLM likely produced a "progress update" rather than a
        // final answer. Auto-continue up to MAX_TEXT_CONTINUATIONS times.
        const hasIncompleteSteps = this.sessionState?.hasPendingPlanSteps() ?? false;

        // Fast-path: plan has pending steps → continue without LLM call
        if (hasIncompleteSteps && textOnlyContinuations < MAX_TEXT_CONTINUATIONS) {
          textOnlyContinuations++;
          this.pushMessage({
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
          this.pushMessage({
            role: MessageRole.SYSTEM,
            content: nudge,
          });
          continue;
        }

        this.pushMessage({
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
        this.pushMessage({
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
    this.stagnationDetector.resetAll();
    this.resetRunState();
  }

  /** Reset per-run transient state (shared between run() and resetIterations()). */
  private resetRunState(): void {
    this.unresolvedDoubleCheckFailure = false;
    this.successfulReadonlyCallKeys.clear();
    this.stagnationDetector.resetRunState();
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
      this.estimatedTokens,
      this.lastReportedInputTokens,
    );
    const threshold = maxTokens * this.config.context.triggerRatio;

    if (!options?.force && estimatedTokens <= threshold) {
      // Approaching-limit warning: nudge the LLM to persist findings before
      // compaction fires. Fires once at ~60% of trigger threshold to give
      // the LLM enough time to persist findings before pruning strips context.
      const warningThreshold = threshold * APPROACHING_LIMIT_RATIO;
      if (!this.approachingLimitWarned && estimatedTokens > warningThreshold && this.sessionState) {
        this.approachingLimitWarned = true;
        this.pushMessage({
          role: MessageRole.SYSTEM,
          content: "Context is filling up. You MUST persist any analysis conclusions or review findings NOW using save_finding. After context pruning, old tool outputs will be replaced with summaries. Do NOT re-read files already listed in session state — rely on the summaries and findings you've saved.",
        });
      }
      return;
    }

    // Pre-Phase 1: enforce pinned token budget — unpin oldest when over limit
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
    const messageTokens = this.estimatedTokens;
    const overhead = Math.max(0, this.lastReportedInputTokens - messageTokens);
    const messageThreshold = Math.max(0, threshold - overhead);
    const pruneResult = this.pruneToolOutputs(messageTokens, messageThreshold);
    if (pruneResult.savedTokens > 0) {
      this.resetPostCompactionState(false);

      // Re-inject session state so the LLM sees updated summaries
      this.injectSessionState();
      // Full recalculation: pruning + injectSessionState mutate messages in place.
      this.estimatedTokens = estimateMessageTokens(this.messages);
      const postPruneTokens = this.estimatedTokens;

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
    // Capture pre-compaction summary for quality assessment
    const preCompactionSummary = buildPreCompactionSummary(
      this.sessionState, this.messages, this.iterations,
    );
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
        // Full recalculation: compaction replaces the entire message array.
        this.estimatedTokens = estimateMessageTokens(this.messages);
        const postCompactTokens = this.estimatedTokens;
        this.bus.emit("context:compacted", {
          removedCount: result.removedCount,
          prunedCount: pruneResult.prunedCount > 0 ? pruneResult.prunedCount : undefined,
          tokensSaved: pruneResult.savedTokens > 0 ? pruneResult.savedTokens : undefined,
          estimatedTokens: postCompactTokens,
          tokensBefore: estimatedTokens,
        });

        // Post-compaction quality assessment
        try {
          const judgeResult = await judgeCompactionQuality(
            this.provider, preCompactionSummary, this.messages, this.sessionState,
          );
          if (judgeResult && judgeResult.quality_loss >= 0.6) {
            this.pushMessage({
              role: MessageRole.SYSTEM,
              content: `COMPACTION GAP WARNING: ${judgeResult.recommendation}\nMissing context: ${judgeResult.missing_context.join("; ")}`,
            });
            this.bus.emit("error", {
              message: `Compaction quality loss: ${judgeResult.quality_loss.toFixed(2)}`,
              code: "COMPACTION_QUALITY_LOSS",
              fatal: false,
            });
          }
        } catch {
          // Judge failure is non-fatal — continue without quality assessment
        }

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

    // Target: free enough tokens to reach 85% of threshold (headroom to avoid re-triggering).
    const targetTokens = threshold * PRUNE_THRESHOLD_RATIO;
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
      const delta = msgTokens - replacementTokens;
      savedTokens += delta;
      this.estimatedTokens -= delta;
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
    const snippet = inline.length > MAX_INLINE_SUMMARY_CHARS
      ? `${inline.slice(0, MAX_INLINE_SUMMARY_CHARS - 3)}...`
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
      // Preserve successfulReadonlyCallKeys through compaction — the in-memory
      // dedup set is still accurate because compaction does not change the workspace.
      // Only mutating tool execution clears it (see executeTool).
      // Clearing here caused post-compaction re-read storms for all tool types
      // (especially search_files/find_files whose coverage targets can't be
      // round-tripped back to the original dedup key format).
      this.stagnationDetector.notifyCompaction(this.iterations);
    }
  }

  private injectSessionState(knownTokenEstimate?: number): void {
    if (!this.sessionState) return;

    // Determine tier based on available context headroom
    let tier: "full" | "compact" | "minimal" = "full";
    const maxBudget = this.getEffectiveContextBudget();
    if (maxBudget > 0) {
      const totalEstimate = knownTokenEstimate ?? this.estimatedTokens;
      const headroom = maxBudget - totalEstimate;
      tier = headroom > 8000 ? "full"
        : headroom > 3000 ? "compact"
        : "minimal";
    }

    const content = this.sessionState.toSystemMessage(tier);
    if (!content) return;

    // Remove any existing session-state message and adjust running counter
    const prevMessages = this.messages;
    this.messages = [];
    for (const m of prevMessages) {
      if (m.role === MessageRole.SYSTEM && m.content?.startsWith(SESSION_STATE_MARKER)) {
        this.estimatedTokens -= estimateMessageTokens([m]);
      } else {
        this.messages.push(m);
      }
    }

    // Insert after the first SYSTEM message (the system prompt)
    const newMsg: Message = {
      role: MessageRole.SYSTEM,
      content,
    };
    const insertIdx = this.messages[0]?.role === MessageRole.SYSTEM ? 1 : 0;
    this.messages.splice(insertIdx, 0, newMsg);
    this.estimatedTokens += estimateMessageTokens([newMsg]);
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
          const replacement = `${SUPERSEDED_MARKER_PREFIX} by later ${toolName}. See recent activity in session state.]`;
          // Adjust running counter: subtract old content, add new (shorter) content
          this.estimatedTokens -= estimateMessageTokens([prevMsg]);
          this.messages[prevIdx] = {
            ...prevMsg,
            content: replacement,
          };
          this.estimatedTokens += estimateMessageTokens([this.messages[prevIdx]!]);
        }
        // Check for post-compaction re-read storm
        this.stagnationDetector.checkRereadStorm(toolName, target, this.iterations);
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
        const readResult = this.stagnationDetector.trackFileRead(filePath);
        if (readResult.action === "block") {
          // Hard block: replace tool content with a refusal
          this.pushMessage({
            role: MessageRole.TOOL,
            content: readResult.message!,
            toolCallId: callId,
          });
          this.bus.emit("message:tool", {
            role: "tool" as const,
            content: `[blocked: ${filePath} read limit exceeded]`,
            toolCallId: callId,
          });
          return;
        }
        if (readResult.action === "nudge") {
          this.pushMessage({
            role: MessageRole.SYSTEM,
            content: readResult.message!,
          });
        }
      }
    }

    // Auto-pin git_diff results so they survive compaction
    const shouldPin = toolName === "git_diff" && result.success && this.pinnedDiffCount < MAX_PINNED_DIFFS;
    if (shouldPin) this.pinnedDiffCount++;

    this.pushMessage({
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
    return this.tools.getAll();
  }

  private normalizeToolCall(
    toolCall: PendingToolCall,
    category: ToolSpec["category"],
  ): NormalizedToolCall {
    if (category !== "readonly") return { toolCall, bypassResult: null, scriptSteps: null };
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

    const dedupedSteps: ToolScriptStep[] = [];
    const skippedSteps: ToolScriptStep[] = [];
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
        steps: dedupedSteps,
      },
    };

    return {
      toolCall: normalizedCall,
      bypassResult: null,
      scriptSteps: dedupedSteps,
    };
  }

  private collectSuccessfulScriptStepResults(
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
      // For search/find tools, include the pattern so that different patterns
      // on the same path each get a unique SessionState slot. Without this,
      // successive searches scoped to the same directory overwrite each other,
      // suppressing the stagnation detector's progress signal and causing
      // premature stall-lock before all source directories are discovered.
      if (toolName === "search_files") {
        const pattern = args["pattern"];
        if (typeof pattern === "string") {
          const truncated = pattern.length > 60 ? pattern.slice(0, 57) + "..." : pattern;
          // Omit path suffix for root/cwd — identical to the no-path form.
          const pathSuffix = normalized !== "." ? `@${normalized}` : "";
          return `search:${truncated}${pathSuffix}`;
        }
      }
      if (toolName === "find_files") {
        const pattern = args["pattern"];
        if (typeof pattern === "string") {
          const truncated = pattern.length > 60 ? pattern.slice(0, 57) + "..." : pattern;
          const pathSuffix = normalized !== "." ? `@${normalized}` : "";
          return `find:${truncated}${pathSuffix}`;
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
      const message = extractErrorMessage(err);
      result = {
        success: false,
        output: "",
        error: message,
        artifacts: [],
      };
    }

    const durationMs = Date.now() - startTime;

    // Track recent tool calls for doom loop + tool fatigue detection
    this.stagnationDetector.recordToolResult(effectiveCall.name, effectiveCall.arguments, result.success);

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
            summary: formatToolSummary(effectiveCall, originalOutput),
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
                summary: formatToolSummary(
                  {
                    name: stepResult.step.tool,
                    arguments: stepResult.step.args,
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


function collectReferencedStepIds(
  steps: ReadonlyArray<ToolScriptStep>,
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
