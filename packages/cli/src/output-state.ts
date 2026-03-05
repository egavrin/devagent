/**
 * OutputState — encapsulates all mutable output-tracking state that was
 * previously scattered across module-level `let` declarations in main.ts.
 *
 * Two tiers of counters:
 *   - **Per-turn**: reset at the start of each user query / interactive turn.
 *   - **Per-session**: accumulate across every turn in the CLI session.
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
  turnOutputTokens = 0;
  turnCostDelta = 0;

  /**
   * Buffer for streamed assistant text. Accumulated during each LLM iteration.
   * Discarded if tool calls follow; flushed to stdout on final response.
   */
  textBuffer = "";

  /** Whether we're currently buffering text. */
  isBufferingText = false;

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

  // ─── Session-level state ────────────────────────────────────

  sessionTotalIterations = 0;
  sessionTotalToolCalls = 0;
  sessionTotalInputTokens = 0;
  sessionTotalOutputTokens = 0;
  sessionTotalCost = 0;
  readonly sessionToolUsage = new Map<string, number>();

  // ─── Methods ────────────────────────────────────────────────

  /** Reset per-turn counters at the start of each query / interactive turn. */
  resetTurn(): void {
    this.currentIteration = 0;
    this.hadToolCalls = false;
    this.currentTokens = 0;
    this.maxContextTokens = 0;
    this.turnToolCallCount = 0;
    this.turnInputTokens = 0;
    this.turnOutputTokens = 0;
    this.turnCostDelta = 0;
    this.textBuffer = "";
    this.isBufferingText = false;
    this.pendingToolGroup = null;
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
  }
}
