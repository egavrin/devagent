/**
 * Periodic tool-use summary — generates a compact summary of recent
 * tool activity and injects it as a system message. Helps the LLM
 * maintain awareness of what it has done across long sessions.
 *
 * Inspired by claude-code-src toolUseSummary pattern.
 */

import type { SessionState } from "./session-state.js";
import type { Message } from "../core/index.js";
import { MessageRole } from "../core/index.js";

// ─── Constants ──────────────────────────────────────────────

/** Default iterations between tool-use summaries. */
export const DEFAULT_TOOL_USE_SUMMARY_INTERVAL = 10;

/** Marker prefix for tool-use summary system messages. */
export const TOOL_USE_SUMMARY_MARKER = "[TOOL USE SUMMARY";

// ─── Types ──────────────────────────────────────────────────

interface ToolUseSummaryOptions {
  /** Iterations between summaries (0 = disabled). */
  readonly interval: number;
}

// ─── ToolUseSummaryGenerator ────────────────────────────────

export class ToolUseSummaryGenerator {
  private lastSummaryIteration = 0;
  private readonly interval: number;

  constructor(options?: Partial<ToolUseSummaryOptions>) {
    this.interval = options?.interval ?? DEFAULT_TOOL_USE_SUMMARY_INTERVAL;
  }

  /**
   * Check if a summary should be generated at the current iteration.
   * Returns a system message to inject, or null if not due.
   */
  maybeSummarize(
    iteration: number,
    messages: ReadonlyArray<Message>,
    sessionState: SessionState | null,
  ): Message | null {
    if (this.interval <= 0) return null;
    if (iteration - this.lastSummaryIteration < this.interval) return null;

    const summary = this.buildSummary(messages, sessionState, iteration);
    if (!summary) return null;

    this.lastSummaryIteration = iteration;
    return {
      role: MessageRole.SYSTEM,
      content: `${TOOL_USE_SUMMARY_MARKER} — iterations ${iteration - this.interval + 1}–${iteration}]\n\n${summary}`,
    };
  }

  /**
   * Reset state (e.g., after compaction).
   */
  reset(): void {
    this.lastSummaryIteration = 0;
  }

  // ─── Private ────────────────────────────────────────────────

  private buildSummary(
    messages: ReadonlyArray<Message>,
    sessionState: SessionState | null,
    currentIteration: number,
  ): string | null {
    const sections: string[] = [];

    // Collect tool activity from recent messages (since last summary)
    const toolCounts = new Map<string, number>();
    const toolErrors = new Map<string, number>();
    const filesModified = new Set<string>();
    const filesRead = new Set<string>();

    // Walk messages backwards to find recent tool activity
    let toolMessagesFound = 0;
    for (let i = messages.length - 1; i >= 0; i--) {
      const msg = messages[i]!;

      // Stop at previous summary marker
      if (msg.role === MessageRole.SYSTEM && msg.content?.startsWith(TOOL_USE_SUMMARY_MARKER)) {
        break;
      }

      // Count tool calls from assistant messages
      if (msg.role === MessageRole.ASSISTANT && msg.toolCalls) {
        for (const tc of msg.toolCalls) {
          toolCounts.set(tc.name, (toolCounts.get(tc.name) ?? 0) + 1);

          // Track file paths
          const path = tc.arguments["path"];
          if (typeof path === "string" && path.trim().length > 0) {
            const isMutating = tc.name === "write_file" || tc.name === "replace_in_file";
            if (isMutating) {
              filesModified.add(path);
            } else {
              filesRead.add(path);
            }
          }
        }
      }

      // Count tool errors
      if (msg.role === MessageRole.TOOL && msg.content) {
        toolMessagesFound++;
        if (msg.content.startsWith("Error:")) {
          // Find the tool name from the assistant message's toolCalls
          const toolCallId = msg.toolCallId;
          if (toolCallId) {
            for (let j = i - 1; j >= 0; j--) {
              const prev = messages[j]!;
              if (prev.role === MessageRole.ASSISTANT && prev.toolCalls) {
                const tc = prev.toolCalls.find((c) => c.callId === toolCallId);
                if (tc) {
                  toolErrors.set(tc.name, (toolErrors.get(tc.name) ?? 0) + 1);
                  break;
                }
              }
            }
          }
        }
      }

      // Limit backward scan
      if (toolMessagesFound > 100) break;
    }

    if (toolCounts.size === 0) return null;

    // Tool usage summary
    const toolLines: string[] = [];
    for (const [name, count] of [...toolCounts.entries()].sort((a, b) => b[1] - a[1])) {
      const errors = toolErrors.get(name) ?? 0;
      const errorSuffix = errors > 0 ? ` (${errors} errors)` : "";
      toolLines.push(`- ${name}: ${count} calls${errorSuffix}`);
    }
    sections.push(`## Tool usage\n${toolLines.join("\n")}`);

    // Files touched
    if (filesModified.size > 0) {
      const lines = [...filesModified].slice(0, 10).map((f) => `- ${f}`);
      if (filesModified.size > 10) lines.push(`- ... (+${filesModified.size - 10} more)`);
      sections.push(`## Files modified\n${lines.join("\n")}`);
    }

    if (filesRead.size > 0 && filesRead.size <= 15) {
      const lines = [...filesRead].map((f) => `- ${f}`);
      sections.push(`## Files read\n${lines.join("\n")}`);
    } else if (filesRead.size > 15) {
      sections.push(`## Files read\n${filesRead.size} files examined`);
    }

    // Plan progress from session state
    if (sessionState) {
      const plan = sessionState.getPlan();
      if (plan && plan.length > 0) {
        const completed = plan.filter((s) => s.status === "completed").length;
        const inProgress = plan.filter((s) => s.status === "in_progress").length;
        const pending = plan.filter((s) => s.status === "pending").length;
        sections.push(`## Plan progress\n${completed}/${plan.length} completed, ${inProgress} in progress, ${pending} pending`);
      }
    }

    return sections.join("\n\n");
  }
}
