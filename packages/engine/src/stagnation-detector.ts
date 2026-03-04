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

import type { EventBus, LLMProvider, Message } from "@devagent/core";
import { MessageRole } from "@devagent/core";
import type { SessionState } from "./session-state.js";

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
      const stateContext = this.buildSessionStateContext();

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
      const cleaned = responseText.replace(/```json\s*|```\s*/g, "").trim();
      const parsed = JSON.parse(cleaned) as {
        analysis: string;
        stagnation_confidence: number;
      };
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

  // ─── Private ─────────────────────────────────────────────────

  private buildSessionStateContext(): string {
    if (!this.sessionState) return "No session state available.";

    const lines: string[] = [];

    const plan = this.sessionState.getPlan();
    if (plan && plan.length > 0) {
      const completed = this.sessionState.getPlanCompletedCount();
      const total = this.sessionState.getTotalPlanCount();
      lines.push(`Plan progress: ${completed}/${total} steps completed`);
      for (const step of plan) {
        lines.push(`  [${step.status}] ${step.description}`);
      }
    } else {
      lines.push("No plan set.");
    }

    const modifiedFiles = this.sessionState.getModifiedFiles();
    lines.push(`Modified files: ${modifiedFiles.length}`);

    const findings = this.sessionState.getFindings();
    lines.push(`Findings: ${findings.length}`);

    return lines.join("\n");
  }
}

// ─── Helpers ────────────────────────────────────────────────

async function collectStreamText(
  provider: LLMProvider,
  messages: ReadonlyArray<Message>,
): Promise<string> {
  const chunks: string[] = [];
  for await (const chunk of provider.chat(messages)) {
    if (chunk.type === "text") {
      chunks.push(chunk.content);
    }
  }
  return chunks.join("");
}

/** Max characters for a single tool argument value in the judge context. */
const JUDGE_ARG_MAX_CHARS = 200;
/** Max characters for tool result content in the judge context. */
const JUDGE_RESULT_MAX_CHARS = 300;

/**
 * Format a conversation message for the stagnation judge, preserving
 * tool call names, arguments, and result context that the judge needs
 * to distinguish productive batch work from stagnation loops.
 */
function formatMessageForJudge(m: Message): string {
  const parts: string[] = [`[${m.role}]`];

  // For ASSISTANT messages: include tool call details (names + arguments)
  if (m.toolCalls && m.toolCalls.length > 0) {
    for (const tc of m.toolCalls) {
      const argsStr = formatToolArgs(tc.arguments);
      parts.push(`  tool_call: ${tc.name}(${argsStr})`);
    }
  }

  // For TOOL messages: include the tool call ID reference and result content
  if (m.role === MessageRole.TOOL && m.toolCallId) {
    parts.push(`  tool_result [${m.toolCallId}]`);
  }

  // Include text content if present (truncated)
  if (m.content) {
    const truncated = m.content.length > JUDGE_RESULT_MAX_CHARS
      ? m.content.slice(0, JUDGE_RESULT_MAX_CHARS) + "..."
      : m.content;
    parts.push(`  ${truncated}`);
  }

  return parts.join("\n");
}

/**
 * Format tool call arguments for judge context. Shows key=value pairs
 * with values truncated, focusing on the arguments that matter for
 * distinguishing stagnation (file paths, search patterns, etc.).
 */
function formatToolArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  return entries
    .map(([key, val]) => {
      const str = typeof val === "string" ? val : JSON.stringify(val);
      const truncated = str.length > JUDGE_ARG_MAX_CHARS
        ? str.slice(0, JUDGE_ARG_MAX_CHARS) + "..."
        : str;
      return `${key}=${truncated}`;
    })
    .join(", ");
}

function normalizeRepoPath(filePath: string): string {
  const normalized = filePath.trim().replaceAll("\\", "/");
  return normalized.startsWith("./") ? normalized.slice(2) : normalized;
}
