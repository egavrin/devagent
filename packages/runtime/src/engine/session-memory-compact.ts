/**
 * Session memory compaction — builds a compacted message array from
 * SessionState instead of calling the LLM for summarization.
 *
 * This is cheaper and faster than LLM-based compaction (~$0 vs ~$0.02-0.05).
 * Falls back to LLM compaction when session state is too sparse.
 */

import type { SessionState } from "./session-state.js";
import type { Message } from "../core/index.js";
import { MessageRole, estimateMessageTokens } from "../core/index.js";

// ─── Constants ──────────────────────────────────────────────

/** Minimum knowledge + findings entries to consider session memory sufficient. */
const MIN_ENTRIES_FOR_COMPACT = 3;

/** Max tokens for the session memory summary message. */
const MAX_SUMMARY_TOKENS = 8_000;

// ─── Types ──────────────────────────────────────────────────

export interface SessionMemoryCompactResult {
  readonly messages: Message[];
  readonly success: boolean;
}

// ─── Session Memory Compaction ──────────────────────────────

/**
 * Attempt to compact messages using SessionState content instead of LLM.
 *
 * Strategy:
 * 1. Keep system prompt (message 0) and first user message
 * 2. Build a summary from SessionState (knowledge, findings, plan, files, summaries)
 * 3. Keep the last N messages that fit in the token budget
 * 4. If the result fits → success, skip LLM compaction
 */
export function trySessionMemoryCompact(
  messages: ReadonlyArray<Message>,
  sessionState: SessionState,
  maxTokens: number,
): SessionMemoryCompactResult {
  if (!hasEnoughSessionMemory(sessionState)) {
    return { messages: [], success: false };
  }

  // Build the summary message from session state
  const summaryContent = buildSessionMemorySummary(sessionState);
  const summaryTokens = estimateMessageTokens([{
    role: MessageRole.SYSTEM,
    content: summaryContent,
  }]);

  // If the summary itself exceeds budget, fall back
  if (summaryTokens > MAX_SUMMARY_TOKENS) {
    return { messages: [], success: false };
  }

  const { preserved, preservedTokens } = collectPreservedMessages(messages, summaryContent, summaryTokens);
  const firstUserIdx = messages.findIndex((m) => m.role === MessageRole.USER);
  const budgetForRecent = maxTokens - preservedTokens;
  if (budgetForRecent <= 0) {
    return { messages: [], success: false };
  }

  const { recentMessages, recentTokens } = collectRecentMessages(messages, firstUserIdx, budgetForRecent);

  const result = [...preserved, ...recentMessages];
  const totalTokens = preservedTokens + recentTokens;

  if (totalTokens > maxTokens) {
    return { messages: [], success: false };
  }

  return { messages: result, success: true };
}

// ─── Helpers ────────────────────────────────────────────────

function hasEnoughSessionMemory(sessionState: SessionState) {
  const totalEntries =
    sessionState.getKnowledge().length +
    sessionState.getFindingsCount() +
    sessionState.getToolSummaries().length;
  return totalEntries >= MIN_ENTRIES_FOR_COMPACT;
}

function collectPreservedMessages(
  messages: ReadonlyArray<Message>,
  summaryContent: string,
  summaryTokens: number,
) {
  const preserved: Message[] = [];
  let preservedTokens = 0;

  const systemMessage = messages[0];
  if (systemMessage?.role === MessageRole.SYSTEM) {
    preserved.push(systemMessage);
    preservedTokens += estimateMessageTokens([systemMessage]);
  }

  const firstUserIdx = messages.findIndex((m) => m.role === MessageRole.USER);
  if (firstUserIdx > 0) {
    preserved.push(messages[firstUserIdx]!);
    preservedTokens += estimateMessageTokens([messages[firstUserIdx]!]);
  }

  preserved.push(createSessionMemorySummaryMessage(summaryContent));
  return { preserved, preservedTokens: preservedTokens + summaryTokens };
}

function createSessionMemorySummaryMessage(summaryContent: string): Message {
  return {
    role: MessageRole.ASSISTANT,
    content: `[Session memory summary — compacted from conversation history]\n\n${summaryContent}`,
  };
}

function collectRecentMessages(
  messages: ReadonlyArray<Message>,
  firstUserIdx: number,
  budgetForRecent: number,
) {
  const recentMessages: Message[] = [];
  let recentTokens = 0;

  for (let i = messages.length - 1; i >= 0; i--) {
    const msg = messages[i]!;
    if (isPreservedMessageIndex(i, firstUserIdx)) continue;
    if (isOldSessionStateMarker(msg)) continue;

    const msgTokens = estimateMessageTokens([msg]);
    if (recentTokens + msgTokens > budgetForRecent) break;

    recentMessages.unshift(msg);
    recentTokens += msgTokens;
  }

  return { recentMessages, recentTokens };
}

function isPreservedMessageIndex(index: number, firstUserIdx: number) {
  return index === 0 || index === firstUserIdx;
}

function isOldSessionStateMarker(msg: Message) {
  return msg.role === MessageRole.SYSTEM && msg.content?.startsWith("[SESSION STATE");
}

function buildSessionMemorySummary(sessionState: SessionState): string {
  const sections: string[] = [];

  // Plan progress
  const plan = sessionState.getPlan();
  if (plan && plan.length > 0) {
    const completed = plan.filter((s) => s.status === "completed").length;
    const lines = plan.map((s) => `- [${s.status}] ${s.description}`);
    sections.push(`## Plan (${completed}/${plan.length} completed)\n${lines.join("\n")}`);
  }

  // Knowledge (most valuable — domain understanding)
  const knowledge = sessionState.getKnowledge();
  if (knowledge.length > 0) {
    const lines = knowledge.map((k) => `- **${k.key}**: ${k.content}`);
    sections.push(`## Accumulated Knowledge\n${lines.join("\n")}`);
  }

  // Findings (analysis conclusions)
  const findings = sessionState.getFindings();
  if (findings.length > 0) {
    const lines = findings.map((f) => `- **${f.title}**: ${f.detail}`);
    sections.push(`## Findings\n${lines.join("\n")}`);
  }

  // Modified files
  const modifiedFiles = sessionState.getModifiedFiles();
  if (modifiedFiles.length > 0) {
    const lines = modifiedFiles.map((f) => `- ${f}`);
    sections.push(`## Modified Files\n${lines.join("\n")}`);
  }

  // Recent tool summaries (last 15)
  const summaries = sessionState.getToolSummaries();
  if (summaries.length > 0) {
    const recent = summaries.slice(-15);
    const lines = recent.map((s) => `- [iter ${s.iteration}] ${s.tool}(${s.target}): ${s.summary.replace(/\n/g, " | ").slice(0, 120)}`);
    sections.push(`## Recent Activity\n${lines.join("\n")}`);
  }

  // Environment facts
  const envFacts = sessionState.getEnvFacts();
  if (envFacts.length > 0) {
    const lines = envFacts.map((f) => `- ${f}`);
    sections.push(`## Environment\n${lines.join("\n")}`);
  }

  return sections.join("\n\n");
}
