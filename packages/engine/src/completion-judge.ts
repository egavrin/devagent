/**
 * Completion Judge — determines whether an LLM text-only response
 * is a final answer or a progress update.
 *
 * Prevents the task loop from exiting prematurely when the LLM
 * produces text-only "progress update" responses instead of final answers.
 * A progress update means the LLM intends to continue working, so the
 * loop should keep iterating rather than returning the partial result.
 */

import type { LLMProvider } from "@devagent/core";
import { MessageRole } from "@devagent/core";
import type { SessionState } from "./session-state.js";
import {
  collectStreamText,
  parseJudgeResponse,
  buildSessionStateContext,
} from "./llm-judge.js";

// ─── Types ───────────────────────────────────────────────────

export interface CompletionJudgeResult {
  is_final: boolean;
  confidence: number;
  reason: string;
}

// ─── System prompt ───────────────────────────────────────────

export const COMPLETION_JUDGE_SYSTEM_PROMPT = `You determine whether an AI coding assistant's text response is a FINAL ANSWER or a PROGRESS UPDATE.

## Final answer signals
- Summarizes findings, conclusions, or deliverables
- Directly addresses the original user request
- Uses past tense about completed work
- No forward-looking action language
- Phrases like "Here are the results", "I found that...", "All set.", "Done."
- Asking the user a question (waiting for input counts as final)

## Progress update signals
- States intent to continue: "Let me...", "Now I'll...", "Next I need to..."
- Partial results with pending action stated
- Does not answer the original request yet
- Narrates intermediate status

## Edge cases
- "I've completed X and here are the results" = final
- Lists findings AND says "let me also check Y" = NOT final (still working)
- "Done." or "All set." = final
- Asking user a question = final (waiting for input)

## Context signals
- "Had tool calls this turn: yes" means the assistant already used tools this session — a text-only response between tool phases is likely a progress update, not a final answer
- "Had tool calls this turn: no" means no tools were used — the response is more likely a direct answer to a simple question

Respond ONLY with valid JSON (no markdown fences, no commentary):
{"is_final": true|false, "confidence": 0.0-1.0, "reason": "brief explanation"}`;

// ─── Judge function ──────────────────────────────────────────

export async function judgeCompletion(
  provider: LLMProvider,
  textResponse: string,
  originalRequest: string,
  sessionState: SessionState | null,
  iteration: number,
  hadToolCalls: boolean,
): Promise<CompletionJudgeResult | null> {
  try {
    const truncatedResponse = textResponse.length > 2000
      ? textResponse.slice(0, 2000) + "..."
      : textResponse;

    const truncatedRequest = originalRequest.length > 1000
      ? originalRequest.slice(0, 1000) + "..."
      : originalRequest;

    const parts: string[] = [
      `## Original user request\n${truncatedRequest}`,
      `## Assistant's text response\n${truncatedResponse}`,
      `## Session context\n${buildSessionStateContext(sessionState)}`,
      `## Current iteration: ${iteration}`,
      `## Had tool calls this turn: ${hadToolCalls ? "yes" : "no"}`,
      "\nIs this a final answer or a progress update? Respond with JSON only.",
    ];

    const messages = [
      { role: MessageRole.SYSTEM as const, content: COMPLETION_JUDGE_SYSTEM_PROMPT },
      { role: MessageRole.USER as const, content: parts.join("\n\n") },
    ];

    const responseText = await collectStreamText(provider, messages);
    return parseJudgeResponse<CompletionJudgeResult>(responseText);
  } catch {
    return null;
  }
}
