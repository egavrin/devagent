/**
 * Error Recovery Classification Judge — classifies tool errors
 * to guide the LLM's recovery strategy.
 *
 * Categories: code_error (fixable), infrastructure (retry/ask user),
 * permission (ask user), tool_misuse (change approach).
 */

import type { LLMProvider } from "../core/index.js";
import { MessageRole } from "../core/index.js";
import {
  collectStreamText,
  parseJudgeResponse,
  formatToolArgs,
} from "./llm-judge.js";

// ─── Types ───────────────────────────────────────────────────

export interface ErrorClassification {
  category: "code_error" | "infrastructure" | "permission" | "tool_misuse";
  severity: "low" | "medium" | "high";
  recovery_hint: string;
}

// ─── System prompt ───────────────────────────────────────────

export const ERROR_JUDGE_SYSTEM_PROMPT = `You classify tool errors to guide recovery strategy for an AI coding assistant.

Categories:
- code_error: Syntax error, type error, test failure, lint error — fixable by editing code
- infrastructure: Network timeout, disk full, process killed, service unavailable — retry or ask user
- permission: Access denied, auth required, read-only filesystem — ask user to resolve
- tool_misuse: Wrong arguments, non-existent file path, invalid command — change approach

Respond ONLY with valid JSON (no markdown fences, no commentary):
{"category": "code_error|infrastructure|permission|tool_misuse", "severity": "low|medium|high", "recovery_hint": "actionable guidance"}`;

// ─── Judge function ──────────────────────────────────────────

export async function classifyError(
  provider: LLMProvider,
  toolName: string,
  args: Record<string, unknown>,
  errorMessage: string,
  recentContext: string,
): Promise<ErrorClassification | null> {
  try {
    const truncatedError = errorMessage.length > 500
      ? errorMessage.slice(0, 500) + "..."
      : errorMessage;

    const parts: string[] = [
      `## Failed tool: ${toolName}`,
      `## Arguments: ${formatToolArgs(args)}`,
      `## Error message\n${truncatedError}`,
      `## Recent conversation context\n${recentContext}`,
      "\nClassify this error. Respond with JSON only.",
    ];

    const messages = [
      { role: MessageRole.SYSTEM as const, content: ERROR_JUDGE_SYSTEM_PROMPT },
      { role: MessageRole.USER as const, content: parts.join("\n\n") },
    ];

    const responseText = await collectStreamText(provider, messages);
    return parseJudgeResponse<ErrorClassification>(responseText);
  } catch {
    return null;
  }
}
