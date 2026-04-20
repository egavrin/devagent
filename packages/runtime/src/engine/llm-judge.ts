/**
 * Shared LLM-as-judge infrastructure.
 *
 * Provides reusable utilities for all LLM judge modules:
 * stream collection, JSON parsing, message formatting, session state context.
 *
 * Extracted from stagnation-detector.ts to avoid duplication across
 * compaction-judge, plan-judge, subagent-judge, and error-judge.
 */

import type { SessionState } from "./session-state.js";
import type { LLMProvider, Message } from "../core/index.js";
import { MessageRole } from "../core/index.js";

// ─── Constants ────────────────────────────────────────────────

/** Max characters for a single tool argument value in the judge context. */
export const JUDGE_ARG_MAX_CHARS = 200;
/** Max characters for tool result content in the judge context. */
export const JUDGE_RESULT_MAX_CHARS = 300;

// ─── Stream collection ───────────────────────────────────────

/**
 * Collect all text chunks from a provider chat stream into a single string.
 * Ignores non-text chunks (thinking, tool_call, done, error).
 */
export async function collectStreamText(
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

// ─── JSON parsing ────────────────────────────────────────────

/**
 * Parse a judge LLM response: strip markdown fences, trim, parse JSON.
 * Throws on invalid JSON (callers should catch for graceful degradation).
 */
export function parseJudgeResponse<T>(raw: string): T {
  const cleaned = raw.replace(/```json\s*|```\s*/g, "").trim();
  return JSON.parse(cleaned) as T;
}

// ─── Message formatting ──────────────────────────────────────

/**
 * Format a conversation message for an LLM judge, preserving
 * tool call names, arguments, and result context.
 */
export function formatMessageForJudge(m: Message): string {
  const parts: string[] = [`[${m.role}]`];

  if (m.toolCalls && m.toolCalls.length > 0) {
    for (const tc of m.toolCalls) {
      const argsStr = formatToolArgs(tc.arguments);
      parts.push(`  tool_call: ${tc.name}(${argsStr})`);
    }
  }

  if (m.role === MessageRole.TOOL && m.toolCallId) {
    parts.push(`  tool_result [${m.toolCallId}]`);
  }

  if (m.content) {
    const truncated =
      m.content.length > JUDGE_RESULT_MAX_CHARS
        ? m.content.slice(0, JUDGE_RESULT_MAX_CHARS) + "..."
        : m.content;
    parts.push(`  ${truncated}`);
  }

  return parts.join("\n");
}

/**
 * Format tool call arguments for judge context. Shows key=value pairs
 * with values truncated.
 */
export function formatToolArgs(args: Record<string, unknown>): string {
  const entries = Object.entries(args);
  if (entries.length === 0) return "";
  return entries
    .map(([key, val]) => {
      const str = typeof val === "string" ? val : JSON.stringify(val);
      const truncated =
        str.length > JUDGE_ARG_MAX_CHARS
          ? str.slice(0, JUDGE_ARG_MAX_CHARS) + "..."
          : str;
      return `${key}=${truncated}`;
    })
    .join(", ");
}

// ─── Session state context ───────────────────────────────────

/**
 * Build a text summary of session state for LLM judge context.
 * Includes plan progress, modified files count, and findings count.
 */
export function buildSessionStateContext(
  sessionState: SessionState | null,
): string {
  if (!sessionState) return "No session state available.";

  const lines: string[] = [];

  const plan = sessionState.getPlan();
  if (plan && plan.length > 0) {
    const completed = sessionState.getPlanCompletedCount();
    const total = sessionState.getTotalPlanCount();
    lines.push(`Plan progress: ${completed}/${total} steps completed`);
    for (const step of plan) {
      lines.push(`  [${step.status}] ${step.description}`);
    }
  } else {
    lines.push("No plan set.");
  }

  const modifiedFiles = sessionState.getModifiedFiles();
  lines.push(`Modified files: ${modifiedFiles.length}`);

  const findings = sessionState.getFindings();
  lines.push(`Findings: ${findings.length}`);

  const knowledge = sessionState.getKnowledge();
  if (knowledge.length > 0) {
    lines.push(`Knowledge entries: ${knowledge.length}`);
    for (const k of knowledge) {
      lines.push(`  [${k.key}] ${k.content.slice(0, 200)}`);
    }
  }

  return lines.join("\n");
}
