/**
 * StagnationDetector — extracts all stagnation/doom-loop detection
 * logic from TaskLoop into a cohesive, testable unit.
 *
 * Detects:
 * - Doom loops (identical failing tool calls repeated N times)
 * - Tool fatigue (same tool failing with different args)
 * - LLM-as-judge stagnation (periodic LLM review of conversation history)
 * - Per-file re-read stagnation (with soft nudge + hard block)
 * - Post-compaction re-read storms
 */

import type { EventBus, LLMProvider, Message } from "../core/index.js";
import { MessageRole } from "../core/index.js";
import type { SessionState } from "./session-state.js";
import {
  collectStreamText,
  parseJudgeResponse,
  formatMessageForJudge,
  buildSessionStateContext,
} from "./llm-judge.js";
import { classifyError } from "./error-judge.js";

// ─── Constants ────────────────────────────────────────────────

/** Number of identical failing tool calls (same name + same args) that triggers a doom loop warning. */
const DOOM_LOOP_THRESHOLD = 3;

/** Number of consecutive failures of the same tool (regardless of args) that triggers a "tool fatigue" warning. */
const TOOL_FATIGUE_THRESHOLD = 5;

/** Iterations after compaction in which re-reads count toward a storm. */
const REREAD_STORM_WINDOW = 5;

/** Re-read count within the storm window that triggers the warning. */
const REREAD_STORM_THRESHOLD = 3;

/** Max reads of the same file before injecting a stagnation nudge (sessionState-independent). */
const PER_FILE_READ_LIMIT = 8;

/** Hard block on reads after this many reads of the same file. */
const PER_FILE_READ_HARD_LIMIT = 12;

// ─── LLM-as-judge constants ─────────────────────────────────

/** Don't run the judge before this iteration. */
const MIN_JUDGE_ITERATION = 15;
/** Default interval (iterations) between judge checks. */
const DEFAULT_JUDGE_INTERVAL = 8;
/** High confidence → check more often. */
const MIN_JUDGE_INTERVAL = 5;
/** Low confidence → check less often. */
const MAX_JUDGE_INTERVAL = 12;
/** Confidence threshold at which the judge fires a stagnation nudge. */
const JUDGE_CONFIDENCE_THRESHOLD = 0.85;
/** Number of recent messages sent to the judge for assessment. */
const JUDGE_HISTORY_COUNT = 20;

const STAGNATION_JUDGE_SYSTEM_PROMPT = `You are a diagnostic agent that determines whether a conversational AI coding assistant is stuck in an unproductive state. Analyze the conversation history and session state to make this determination.

## What constitutes stagnation

Stagnation requires BOTH of these to be true:
1. The assistant has exhibited a repetitive pattern over at least 5 consecutive tool calls or text responses.
2. The repetition produces NO net change or forward progress toward the user's goal.

Specific patterns to look for:
- **Alternating cycles with no net effect:** The assistant cycles between the same actions (e.g., read_file → search_files → read_file → search_files) where each iteration targets the same files/patterns and produces the same results, making zero progress.
- **Search diversification without edits:** Running many search variants targeting the same information without making any file edits IS stagnation, even if each search query is slightly different. The key signal is: many readonly operations, zero file modifications.
- **Semantic repetition with identical outcomes:** The assistant calls the same tool with semantically equivalent arguments multiple times consecutively, and each call produces the same outcome.
- **Stuck reasoning:** Multiple consecutive text responses that restate the same plan or analysis without taking any new action.

## What is NOT stagnation

You MUST distinguish repetitive-looking but productive work from true stagnation:
- **Cross-file batch operations:** A series of tool calls targeting different files (different file paths in arguments) — this is distinct work, not repetition.
- **Incremental same-file edits:** Multiple edits to the same file that target different line ranges, functions, or content.
- **Initial codebase exploration:** Early-phase exploration before the first edit (roughly iterations 1-15) is expected and productive.
- **Running tests after edits:** Re-running build/test commands after making code changes is normal verification workflow.
- **Retry with meaningful variation:** Re-attempting a failed operation with modified arguments or a different approach.
- **Verification phase:** After completing all plan steps, readonly operations (git_diff, git_status, tests) are expected verification.

## Argument analysis (critical)

When evaluating tool calls, you MUST compare the ARGUMENTS of each call, not just the tool name:
- **File paths:** Different file paths mean different targets — this is distinct work.
- **Search queries and patterns:** Different search terms with genuinely different intent indicate information gathering. However, minor rephrasing of the same search (e.g., "find class Foo" → "search for Foo class" → "locate Foo") targeting the same information IS stagnation.
- **Line numbers and text content:** Different line ranges or different edit content indicate distinct edits.

Respond ONLY with valid JSON (no markdown fences, no commentary):
{"analysis": "<1-2 sentence explanation>", "stagnation_confidence": <0.0-1.0>}`;

// ─── Types ────────────────────────────────────────────────────

/** Minimal tool-call shape consumed by the detector. */
export interface StagnationToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

/** Configuration passed from TaskLoop at construction time. */
export interface StagnationDetectorOptions {
  readonly bus: EventBus;
  readonly sessionState: SessionState | null;
}

// ─── StagnationDetector ──────────────────────────────────────

export class StagnationDetector {
  private readonly bus: EventBus;
  private readonly sessionState: SessionState | null;

  // Doom loop state
  private recentToolCalls: Array<{ name: string; argsKey: string }> = [];
  private doomLoopWarned = false;

  // Tool fatigue state
  private toolFailureCounts = new Map<string, number>();
  private toolFatigueWarned = new Set<string>();

  // LLM-as-judge stagnation state
  private lastJudgeIteration = 0;
  private judgeInterval = DEFAULT_JUDGE_INTERVAL;

  // Per-file read stagnation
  private perFileReadCount = new Map<string, number>();

  // Post-compaction re-read storm
  private lastCompactionIteration = 0;
  private postCompactionRereadCount = 0;

  // Error classification rate limiting
  private lastErrorClassificationIteration = 0;

  constructor(options: StagnationDetectorOptions) {
    this.bus = options.bus;
    this.sessionState = options.sessionState;
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

    const recovery = getDoomLoopRecovery(toolName);
    return `WARNING: You have called "${toolName}" ${DOOM_LOOP_THRESHOLD} times with the exact same arguments, and it keeps failing. This approach is not working.\n\n${recovery}\n\nDo NOT repeat the same failing call.`;
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

        const recovery = getToolFatigueRecovery(tc.name);
        return `ESCALATED WARNING: The tool "${tc.name}" has failed ${count} consecutive times, even with different arguments. This tool is not working for the current task.\n\n${recovery}\n\nDo NOT call "${tc.name}" again unless you have resolved the underlying issue.`;
      }
    }
    return null;
  }

  /**
   * LLM-as-judge stagnation detection. Periodically reviews recent conversation
   * history + session state and returns a confidence score for stagnation.
   * Returns a nudge string to inject, or null if no stagnation detected.
   */
  async checkStagnationWithLLM(
    provider: LLMProvider,
    messages: ReadonlyArray<Message>,
    iteration: number,
  ): Promise<string | null> {
    if (iteration < MIN_JUDGE_ITERATION) return null;
    if (iteration - this.lastJudgeIteration < this.judgeInterval) return null;

    this.lastJudgeIteration = iteration;

    try {
      // Extract original user request (first USER-role message)
      const userRequest = messages.find((m) => m.role === MessageRole.USER);
      const originalRequest = userRequest?.content ?? "(unknown request)";

      // Slice last JUDGE_HISTORY_COUNT messages
      const recentMessages = messages.slice(-JUDGE_HISTORY_COUNT);

      // Build session state context
      const stateContext = buildSessionStateContext(this.sessionState);

      // Build judge messages with rich tool call context
      const formattedHistory = recentMessages.map((m) => formatMessageForJudge(m)).join("\n\n");

      const judgeMessages: Message[] = [
        { role: MessageRole.SYSTEM, content: STAGNATION_JUDGE_SYSTEM_PROMPT },
        {
          role: MessageRole.USER,
          content: [
            `## Original user request\n${originalRequest}`,
            `## Session state\n${stateContext}`,
            `## Current iteration: ${iteration}`,
            `## Recent conversation (last ${recentMessages.length} messages)\n${formattedHistory}`,
            `\nAssess whether the assistant is stuck in a stagnation loop. Respond with JSON only.`,
          ].join("\n\n"),
        },
      ];

      const responseText = await collectStreamText(provider, judgeMessages);

      // Parse JSON response — strip markdown fences if present
      const parsed = parseJudgeResponse<{
        analysis: string;
        stagnation_confidence: number;
      }>(responseText);
      const confidence = parsed.stagnation_confidence;

      // Adapt interval based on confidence
      if (confidence >= 0.7) {
        this.judgeInterval = MIN_JUDGE_INTERVAL;
      } else if (confidence <= 0.3) {
        this.judgeInterval = MAX_JUDGE_INTERVAL;
      } else {
        this.judgeInterval = DEFAULT_JUDGE_INTERVAL;
      }

      if (confidence >= JUDGE_CONFIDENCE_THRESHOLD) {
        this.bus.emit("error", {
          message: `LLM stagnation judge detected stagnation (confidence: ${confidence.toFixed(2)}): ${parsed.analysis}`,
          code: "LLM_STAGNATION_DETECTED",
          fatal: false,
        });

        return `STAGNATION DETECTED (confidence: ${confidence.toFixed(2)}): ${parsed.analysis}\n\nYou are stuck in an unproductive loop. Stop repeating readonly operations. Either:\n1. Make file edits to advance the task\n2. Persist your findings with save_finding\n3. Finalize your response with what you know\n4. Ask the user for guidance if you are blocked`;
      }

      return null;
    } catch {
      // On any error (parse failure, provider error): return null gracefully
      return null;
    }
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
    this.lastJudgeIteration = 0;
    this.judgeInterval = DEFAULT_JUDGE_INTERVAL;
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

  /**
   * Classify a tool error using the LLM judge to guide recovery strategy.
   * Rate-limited to 1 classification per 3 iterations. Skips short errors.
   */
  async classifyToolError(
    provider: LLMProvider,
    toolName: string,
    args: Record<string, unknown>,
    errorMessage: string,
    recentMessages: ReadonlyArray<Message>,
    iteration: number,
  ): Promise<string | null> {
    if (errorMessage.length < 50) return null;
    if (iteration - this.lastErrorClassificationIteration < 3) return null;

    this.lastErrorClassificationIteration = iteration;
    const recentContext = recentMessages
      .slice(-3)
      .map(formatMessageForJudge)
      .join("\n\n");

    const result = await classifyError(
      provider, toolName, args, errorMessage, recentContext,
    );
    if (!result) return null;

    return `[Error classification: ${result.category}/${result.severity}] ${result.recovery_hint}`;
  }
}

// ─── Recovery hints ─────────────────────────────────────────

const TOOL_RECOVERY: Record<string, { doom: string; fatigue: string }> = {
  replace_in_file: {
    doom: "Re-read the file with read_file to get the exact current content, then use that exact text as the search parameter. If the file has changed since your last read, your search text is stale.",
    fatigue: "The file content does not match your expectations. Re-read the file, verify the exact text you want to replace, and use a broader anchored replacement. If the file has been heavily modified, consider a single large block replacement.",
  },
  execute_tool_script: {
    doom: "Break the script into individual tool calls instead of batching. Check that tool names are bare canonical names (e.g. read_file, not functions.read_file) and args are valid JSON strings.",
    fatigue: "Stop using execute_tool_script for this task. Use individual tool calls (read_file, search_files, find_files) directly — they give clearer error messages and are easier to debug.",
  },
  run_command: {
    doom: "The command keeps failing with the same error. Fix the underlying code or configuration that the command is testing, or try a more targeted command (e.g. run a single test file instead of the full suite).",
    fatigue: "Shell commands are consistently failing. Check if the project builds first (run the build command), then try a targeted test or lint command. If infrastructure is the issue (missing deps, wrong env), ask the user.",
  },
  search_files: {
    doom: "The search pattern is not matching anything. Try a different pattern, broader file_pattern, or use find_files to verify the file structure first.",
    fatigue: "Search is not finding what you need. Try find_files to map the project structure, then read_file on likely candidates. Consider that the code you're looking for may not exist or may use different naming.",
  },
  read_file: {
    doom: "The file path does not exist. Use find_files to discover the correct path, or check if the file is in a different directory.",
    fatigue: "Multiple file reads are failing. Use find_files to verify the project structure and file locations before attempting to read.",
  },
};

const DEFAULT_RECOVERY = {
  doom: "Try a completely different strategy: use a different tool, change your approach, or ask the user for guidance.",
  fatigue: "You MUST try a fundamentally different approach: use a different tool, break the problem into smaller steps, or ask the user for guidance.",
};

function getDoomLoopRecovery(toolName: string): string {
  return TOOL_RECOVERY[toolName]?.doom ?? DEFAULT_RECOVERY.doom;
}

function getToolFatigueRecovery(toolName: string): string {
  return TOOL_RECOVERY[toolName]?.fatigue ?? DEFAULT_RECOVERY.fatigue;
}

// ─── Helpers ────────────────────────────────────────────────

function normalizeRepoPath(filePath: string): string {
  const normalized = filePath.trim().replaceAll("\\", "/");
  return normalized.startsWith("./") ? normalized.slice(2) : normalized;
}
