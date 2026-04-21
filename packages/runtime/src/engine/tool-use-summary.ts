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

interface ToolUseActivity {
  readonly toolCounts: Map<string, number>;
  readonly toolErrors: Map<string, number>;
  readonly filesModified: Set<string>;
  readonly filesRead: Set<string>;
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
    _currentIteration: number,
  ): string | null {
    const sections: string[] = [];
    const activity = collectRecentToolActivity(messages);

    if (activity.toolCounts.size === 0) return null;

    // Tool usage summary
    sections.push(formatToolUsageSection(activity));

    // Files touched
    sections.push(...formatFileActivitySections(activity));

    // Plan progress from session state
    const planSection = formatPlanProgressSection(sessionState);
    if (planSection) sections.push(planSection);

    return sections.join("\n\n");
  }
}

function collectRecentToolActivity(messages: ReadonlyArray<Message>): ToolUseActivity {
  const activity: ToolUseActivity = {
    toolCounts: new Map(),
    toolErrors: new Map(),
    filesModified: new Set(),
    filesRead: new Set(),
  };
  let toolMessagesFound = 0;
  for (let index = messages.length - 1; index >= 0; index--) {
    const message = messages[index]!;
    if (isPriorToolUseSummary(message)) break;
    collectAssistantToolCalls(message, activity);
    toolMessagesFound += collectToolResultError(messages, index, activity);
    if (toolMessagesFound > 100) break;
  }
  return activity;
}

function isPriorToolUseSummary(message: Message): boolean {
  return message.role === MessageRole.SYSTEM && Boolean(message.content?.startsWith(TOOL_USE_SUMMARY_MARKER));
}

function collectAssistantToolCalls(message: Message, activity: ToolUseActivity): void {
  if (message.role !== MessageRole.ASSISTANT) return;
  for (const toolCall of message.toolCalls ?? []) {
    incrementMap(activity.toolCounts, toolCall.name);
    collectToolPath(toolCall, activity);
  }
}

function collectToolPath(
  toolCall: NonNullable<Message["toolCalls"]>[number],
  activity: ToolUseActivity,
): void {
  const path = toolCall.arguments["path"];
  if (typeof path !== "string" || path.trim().length === 0) return;
  const target = toolCall.name === "write_file" || toolCall.name === "replace_in_file"
    ? activity.filesModified
    : activity.filesRead;
  target.add(path);
}

function collectToolResultError(
  messages: ReadonlyArray<Message>,
  index: number,
  activity: ToolUseActivity,
): number {
  const message = messages[index]!;
  if (message.role !== MessageRole.TOOL || !message.content) return 0;
  if (message.content.startsWith("Error:") && message.toolCallId) {
    const toolName = findToolNameForCallId(messages, index, message.toolCallId);
    if (toolName) incrementMap(activity.toolErrors, toolName);
  }
  return 1;
}

function findToolNameForCallId(
  messages: ReadonlyArray<Message>,
  beforeIndex: number,
  toolCallId: string,
): string | null {
  for (let index = beforeIndex - 1; index >= 0; index--) {
    const match = messages[index]?.toolCalls?.find((toolCall) => toolCall.callId === toolCallId);
    if (match) return match.name;
  }
  return null;
}

function incrementMap(counts: Map<string, number>, key: string): void {
  counts.set(key, (counts.get(key) ?? 0) + 1);
}

function formatToolUsageSection(activity: ToolUseActivity): string {
  const lines = [...activity.toolCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => {
      const errors = activity.toolErrors.get(name) ?? 0;
      const errorSuffix = errors > 0 ? ` (${errors} errors)` : "";
      return `- ${name}: ${count} calls${errorSuffix}`;
    });
  return `## Tool usage\n${lines.join("\n")}`;
}

function formatFileActivitySections(activity: ToolUseActivity): string[] {
  return [
    formatFilesModifiedSection(activity.filesModified),
    formatFilesReadSection(activity.filesRead),
  ].filter((section): section is string => section !== null);
}

function formatFilesModifiedSection(filesModified: ReadonlySet<string>): string | null {
  if (filesModified.size === 0) return null;
  const lines = [...filesModified].slice(0, 10).map((file) => `- ${file}`);
  if (filesModified.size > 10) lines.push(`- ... (+${filesModified.size - 10} more)`);
  return `## Files modified\n${lines.join("\n")}`;
}

function formatFilesReadSection(filesRead: ReadonlySet<string>): string | null {
  if (filesRead.size === 0) return null;
  if (filesRead.size > 15) return `## Files read\n${filesRead.size} files examined`;
  return `## Files read\n${[...filesRead].map((file) => `- ${file}`).join("\n")}`;
}

function formatPlanProgressSection(sessionState: SessionState | null): string | null {
  const plan = sessionState?.getPlan();
  if (!plan || plan.length === 0) return null;
  const completed = plan.filter((step) => step.status === "completed").length;
  const inProgress = plan.filter((step) => step.status === "in_progress").length;
  const pending = plan.filter((step) => step.status === "pending").length;
  return `## Plan progress\n${completed}/${plan.length} completed, ${inProgress} in progress, ${pending} pending`;
}
