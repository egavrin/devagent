/**
 * StagnationDetector — extracts all stagnation/doom-loop detection
 * logic from TaskLoop into a cohesive, testable unit.
 *
 * Detects:
 * - Doom loops (identical failing tool calls repeated N times)
 * - Tool fatigue (same tool failing with different args)
 * - No-progress readonly inspection loops
 * - Per-file re-read stagnation (with soft nudge + hard block)
 * - Post-compaction re-read storms
 */

import type { EventBus, ToolCategory } from "@devagent/core";
import type { SessionState } from "./session-state.js";

// ─── Constants ────────────────────────────────────────────────

/** Number of identical failing tool calls (same name + same args) that triggers a doom loop warning. */
const DOOM_LOOP_THRESHOLD = 3;

/** Number of consecutive failures of the same tool (regardless of args) that triggers a "tool fatigue" warning. */
const TOOL_FATIGUE_THRESHOLD = 5;

/** Consecutive readonly-only cycles with no progress before stall lock engages. */
const NO_PROGRESS_THRESHOLD = 5;

/** Iterations after compaction in which re-reads count toward a storm. */
const REREAD_STORM_WINDOW = 5;

/** Re-read count within the storm window that triggers the warning. */
const REREAD_STORM_THRESHOLD = 3;

/** Max reads of the same file before injecting a stagnation nudge (sessionState-independent). */
const PER_FILE_READ_LIMIT = 8;

/** Hard block on reads after this many reads of the same file. */
const PER_FILE_READ_HARD_LIMIT = 12;

// ─── Types ────────────────────────────────────────────────────

interface ProgressSnapshot {
  readonly toolSummaries: number;
  readonly findings: number;
  readonly coverageTargets: number;
  readonly completedPlan: number;
}

/** Minimal tool-call shape consumed by the detector. */
export interface StagnationToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

/** Dependency: resolve tool category by name. */
export type ToolCategoryResolver = (toolName: string) => ToolCategory;

/** Configuration passed from TaskLoop at construction time. */
export interface StagnationDetectorOptions {
  readonly bus: EventBus;
  readonly sessionState: SessionState | null;
  readonly resolveCategory: ToolCategoryResolver;
}

// ─── StagnationDetector ──────────────────────────────────────

export class StagnationDetector {
  private readonly bus: EventBus;
  private readonly sessionState: SessionState | null;
  private readonly resolveCategory: ToolCategoryResolver;

  // Doom loop state
  private recentToolCalls: Array<{ name: string; argsKey: string }> = [];
  private doomLoopWarned = false;

  // Tool fatigue state
  private toolFailureCounts = new Map<string, number>();
  private toolFatigueWarned = new Set<string>();

  // No-progress readonly loop state
  private stagnantReadonlyCycles = 0;
  private lastProgressSnapshot: ProgressSnapshot | null = null;
  private readonlyStallLock = false;

  // Per-file read stagnation
  private perFileReadCount = new Map<string, number>();

  // Post-compaction re-read storm
  private lastCompactionIteration = 0;
  private postCompactionRereadCount = 0;

  constructor(options: StagnationDetectorOptions) {
    this.bus = options.bus;
    this.sessionState = options.sessionState;
    this.resolveCategory = options.resolveCategory;
  }

  // ─── Public API ──────────────────────────────────────────────

  /**
   * Record a tool call result for doom loop and tool fatigue tracking.
   * Called by TaskLoop after each tool execution.
   */
  recordToolResult(toolName: string, args: Record<string, unknown>, success: boolean): void {
    if (success) {
      // Success resets doom loop tracking — the LLM found a working approach
      this.recentToolCalls = [];
      this.doomLoopWarned = false;
      // Reset fatigue counter for this specific tool
      this.toolFailureCounts.delete(toolName);
      this.toolFatigueWarned.delete(toolName);
    } else {
      const argsKey = JSON.stringify(args);
      this.recentToolCalls.push({ name: toolName, argsKey });
      if (this.recentToolCalls.length > DOOM_LOOP_THRESHOLD) {
        this.recentToolCalls.shift();
      }
      // Increment per-tool failure count (regardless of args)
      const prevCount = this.toolFailureCounts.get(toolName) ?? 0;
      this.toolFailureCounts.set(toolName, prevCount + 1);
    }
  }

  /**
   * Doom loop detection: check if the LLM keeps calling the same tool
   * with identical arguments and it keeps failing.
   * Returns a warning message to inject, or null if no doom loop detected.
   *
   * Following the OpenCode pattern (DOOM_LOOP_THRESHOLD = 3):
   * - Does NOT kill the loop — the LLM gets to try a different approach
   * - Warning is injected once per doom loop pattern
   * - Resets when any tool call succeeds (see recordToolResult)
   */
  checkDoomLoop(toolCalls: ReadonlyArray<StagnationToolCall>): string | null {
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
  checkToolFatigue(
    toolCalls: ReadonlyArray<StagnationToolCall>,
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
   * Generic no-progress detection for readonly inspection loops.
   * Injects a nudge into messages when the LLM repeatedly runs readonly
   * tools without making progress (no new coverage, findings, etc.).
   * Returns a system message to inject, or null.
   */
  maybeInjectNoProgressNudge(
    toolCalls: ReadonlyArray<StagnationToolCall>,
  ): string | null {
    if (!this.sessionState || toolCalls.length === 0) return null;

    const snapshot = this.makeProgressSnapshot();
    if (!snapshot) return null;

    const previous = this.lastProgressSnapshot;
    this.lastProgressSnapshot = snapshot;
    if (!previous) return null;

    const hasReadonly = toolCalls.some((tc) => this.resolveCategory(tc.name) === "readonly");
    const hasMutating = toolCalls.some((tc) => this.resolveCategory(tc.name) === "mutating");
    if (hasMutating) {
      this.stagnantReadonlyCycles = 0;
      this.readonlyStallLock = false;
      return null;
    }
    if (!hasReadonly) {
      // State/agent-only batches should not reset readonly stagnation history.
      return null;
    }

    const progressed = snapshot.toolSummaries > previous.toolSummaries
      || snapshot.findings > previous.findings
      || snapshot.coverageTargets > previous.coverageTargets
      || snapshot.completedPlan > previous.completedPlan;

    if (progressed) {
      this.stagnantReadonlyCycles = 0;
      this.readonlyStallLock = false;
      return null;
    }

    this.stagnantReadonlyCycles++;
    if (this.stagnantReadonlyCycles < NO_PROGRESS_THRESHOLD) return null;

    this.stagnantReadonlyCycles = 0;
    this.readonlyStallLock = true;
    this.bus.emit("error", {
      message: "Readonly no-progress loop detected: repeated inspections are not increasing coverage/findings.",
      code: "NO_PROGRESS_LOOP",
      fatal: false,
    });
    return "Readonly inspections are no longer increasing coverage or findings. Stop repetitive reads/diffs. If you already have enough evidence, persist remaining issues with save_finding and finalize your response. Otherwise switch to a different tool/action that produces new evidence.";
  }

  /**
   * Check whether the readonly stall lock is currently engaged.
   * When engaged, readonly inspection tools should be bypassed.
   */
  isReadonlyStallLocked(): boolean {
    return this.readonlyStallLock;
  }

  /**
   * Track a per-file read. Returns an action describing what TaskLoop should do:
   * - "allow": normal processing
   * - "nudge": allow but inject a stagnation warning message
   * - "block": reject the read entirely
   */
  trackFileRead(filePath: string): {
    action: "allow" | "nudge" | "block";
    message?: string;
  } {
    const normalizedPath = normalizeRepoPath(filePath);
    const count = (this.perFileReadCount.get(normalizedPath) ?? 0) + 1;
    this.perFileReadCount.set(normalizedPath, count);

    if (count >= PER_FILE_READ_HARD_LIMIT) {
      return {
        action: "block",
        message: `Blocked: "${normalizedPath}" has been read ${count} times. Use the summaries and content you already have. Do NOT attempt to read this file again.`,
      };
    }

    if (count >= PER_FILE_READ_LIMIT) {
      return {
        action: "nudge",
        message: `You have read "${normalizedPath}" an excessive number of reads (${count} times). You likely already have enough information from this file. Stop re-reading it and synthesize your findings from what you already know. If you need specific details, reference the content you already received.`,
      };
    }

    return { action: "allow" };
  }

  /**
   * Detect post-compaction re-read storms: when the model re-reads the same
   * readonly targets shortly after compaction, it indicates compaction was
   * too aggressive. Emits COMPACTION_REREAD_STORM error event at 3+ re-reads.
   */
  checkRereadStorm(toolName: string, target: string, currentIteration: number): void {
    if (this.lastCompactionIteration === 0) return;
    if (currentIteration - this.lastCompactionIteration > REREAD_STORM_WINDOW) return;

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
   * Notify the detector that a compaction occurred at the given iteration.
   * Resets post-compaction re-read storm tracking.
   */
  notifyCompaction(iteration: number): void {
    this.lastCompactionIteration = iteration;
    this.postCompactionRereadCount = 0;
  }

  /**
   * Reset per-run transient state. Called at the start of each run()
   * and from resetIterations().
   */
  resetRunState(): void {
    this.stagnantReadonlyCycles = 0;
    this.lastProgressSnapshot = null;
    this.readonlyStallLock = false;
    this.perFileReadCount.clear();
  }

  /**
   * Reset all stagnation state for a new turn in multi-turn conversations.
   * Preserves nothing — full reset.
   */
  resetAll(): void {
    this.recentToolCalls = [];
    this.doomLoopWarned = false;
    this.toolFailureCounts.clear();
    this.toolFatigueWarned.clear();
    this.resetRunState();
  }

  // ─── Private ─────────────────────────────────────────────────

  private makeProgressSnapshot(): ProgressSnapshot | null {
    if (!this.sessionState) return null;
    return {
      toolSummaries: this.sessionState.getToolSummariesCount(),
      findings: this.sessionState.getFindingsCount(),
      coverageTargets: this.sessionState.getReadonlyCoverageTargetCount(),
      completedPlan: this.sessionState.getPlanCompletedCount(),
    };
  }
}

// ─── Helpers ────────────────────────────────────────────────

function normalizeRepoPath(filePath: string): string {
  const normalized = filePath.trim().replaceAll("\\", "/");
  return normalized.startsWith("./") ? normalized.slice(2) : normalized;
}
