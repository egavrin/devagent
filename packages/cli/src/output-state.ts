import {
  aggregateDelegatedWork,
} from "@devagent/runtime";
import type { AgentType, DelegatedWorkSummary, LoggedSubagentRun, ReasoningEffort } from "@devagent/runtime";

export interface SubagentDisplayState {
  readonly agentId: string;
  readonly agentType: AgentType;
  readonly laneLabel?: string | null;
  readonly model: string;
  readonly reasoningEffort?: ReasoningEffort;
  readonly status: "running" | "completed" | "error";
  readonly currentIteration: number;
  readonly startedAtMs: number;
  readonly durationMs?: number;
  readonly currentActivity: string;
  readonly recentActivity: ReadonlyArray<string>;
  readonly quality?: {
    readonly score: number;
    readonly completeness: string;
    readonly note?: string;
  };
}

/**
 * OutputState — encapsulates all mutable output-tracking state that was
 * previously scattered across module-level `let` declarations in main.ts.
 *
 * Two tiers of counters:
 *   - **Per-query**: reset at the start of each user query.
 *   - **Per-session**: accumulate across resumed or related CLI queries.
 *
 * Also owns the text-buffering state used to defer streamed assistant output
 * until we know whether tool calls follow (thinking text) or not (final response).
 */
export class OutputState {
  // ─── Per-turn state ──────────────────────────────────────────

  /** Current iteration counter — set by iteration:start events. */
  currentIteration = 0;

  /** Whether any tool was called this turn (for visual separator). */
  hadToolCalls = false;

  /** Current token gauge info — updated by iteration:start events. */
  currentTokens = 0;
  maxContextTokens = 0;

  /** Per-turn accumulators. */
  turnToolCallCount = 0;
  turnInputTokens = 0;
  turnCostDelta = 0;

  /**
   * Buffer for streamed assistant text. Accumulated during each LLM iteration.
   * Discarded if tool calls follow; flushed to stdout on final response.
   */
  textBuffer = "";

  /** Timestamp when thinking started (for duration display). */
  thinkingStartMs: number | null = null;

  /** Pending tool group for collapsing consecutive same-tool calls. */
  pendingToolGroup: {
    name: string;
    count: number;
    params: string[];
    totalDurationMs: number;
    lastSuccess: boolean;
    lastError: string | undefined;
    iteration: number;
    maxIter: number;
  } | null = null;

  /** Live child panel state keyed by subagent id. */
  readonly subagentDisplay = new Map<string, SubagentDisplayState>();
  /** Parallel delegate batches already announced in the UI. */
  readonly announcedSubagentBatches = new Set<string>();
  /** Completed/started subagent runs for session summary and metadata. */
  readonly sessionSubagents = new Map<string, LoggedSubagentRun>();

  // ─── Session-level state ────────────────────────────────────

  sessionTotalIterations = 0;
  sessionTotalToolCalls = 0;
  sessionTotalInputTokens = 0;
  sessionTotalOutputTokens = 0;
  sessionTotalCost = 0;
  readonly sessionToolUsage = new Map<string, number>();

  // ─── Methods ────────────────────────────────────────────────

  /** Reset per-query counters at the start of each query. */
  resetTurn(): void {
    this.currentIteration = 0;
    this.hadToolCalls = false;
    this.currentTokens = 0;
    this.maxContextTokens = 0;
    this.turnToolCallCount = 0;
    this.turnInputTokens = 0;
    this.turnCostDelta = 0;
    this.textBuffer = "";
    this.thinkingStartMs = null;
    this.pendingToolGroup = null;
    this.subagentDisplay.clear();
    this.announcedSubagentBatches.clear();
  }

  /** Reset session-level counters (e.g. on a fresh CLI invocation). */
  resetSession(): void {
    this.resetTurn();
    this.sessionTotalIterations = 0;
    this.sessionTotalToolCalls = 0;
    this.sessionTotalInputTokens = 0;
    this.sessionTotalOutputTokens = 0;
    this.sessionTotalCost = 0;
    this.sessionToolUsage.clear();
    this.sessionSubagents.clear();
  }

  buildDelegatedWorkSummary(): DelegatedWorkSummary {
    return aggregateDelegatedWork([...this.sessionSubagents.values()]);
  }
}
