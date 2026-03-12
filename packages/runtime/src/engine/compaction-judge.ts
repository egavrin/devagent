/**
 * Compaction Quality Judge — assesses whether context compaction
 * preserved critical information for the AI coding assistant.
 *
 * Detects quality loss (missing plan context, file locations, findings)
 * and provides recommendations for gap compensation.
 */

import type { LLMProvider, Message } from "../core/index.js";
import { MessageRole } from "../core/index.js";
import type { SessionState } from "./session-state.js";
import {
  collectStreamText,
  parseJudgeResponse,
  formatMessageForJudge,
  buildSessionStateContext,
} from "./llm-judge.js";

// ─── Types ───────────────────────────────────────────────────

export interface CompactionJudgeResult {
  quality_loss: number;
  missing_context: string[];
  recommendation: string;
}

// ─── System prompt ───────────────────────────────────────────

export const COMPACTION_JUDGE_SYSTEM_PROMPT = `You assess whether context compaction preserved critical information for an AI coding assistant.

Evaluate whether the assistant can continue its current task from the compacted context.

What IS a quality loss:
- Losing which files were modified and why
- Losing the current plan step context
- Losing error messages the assistant was debugging
- Losing user requirements or constraints

What is NOT a quality loss:
- Losing verbose tool output that's summarized in session state
- Losing early exploration messages when the plan is established
- Losing redundant file reads (same file read multiple times)

Respond ONLY with valid JSON (no markdown fences, no commentary):
{"quality_loss": 0.0-1.0, "missing_context": ["critical item lost", ...], "recommendation": "what to inject to compensate"}`;

// ─── Judge function ──────────────────────────────────────────

export async function judgeCompactionQuality(
  provider: LLMProvider,
  preCompactionSummary: string,
  postCompactionMessages: Message[],
  sessionState: SessionState | null,
): Promise<CompactionJudgeResult | null> {
  try {
    const recentPost = postCompactionMessages.slice(-10);
    const formattedPost = recentPost
      .map((m) => formatMessageForJudge(m))
      .join("\n\n");

    const parts: string[] = [
      `## Pre-compaction context summary\n${preCompactionSummary}`,
      `## Post-compaction messages (last ${recentPost.length})\n${formattedPost}`,
      `## Session state sidecar\n${buildSessionStateContext(sessionState)}`,
      "\nAssess the compaction quality. Respond with JSON only.",
    ];

    const messages = [
      { role: MessageRole.SYSTEM as const, content: COMPACTION_JUDGE_SYSTEM_PROMPT },
      { role: MessageRole.USER as const, content: parts.join("\n\n") },
    ];

    const responseText = await collectStreamText(provider, messages);
    return parseJudgeResponse<CompactionJudgeResult>(responseText);
  } catch {
    return null;
  }
}

// ─── Pre-compaction summary builder ──────────────────────────

/**
 * Build a compact summary of context state before compaction.
 * Captures plan progress, modified files, recent activity, and user context.
 */
export function buildPreCompactionSummary(
  sessionState: SessionState | null,
  messages: ReadonlyArray<Message>,
  iteration: number,
): string {
  const parts: string[] = [`Iteration: ${iteration}`];

  if (sessionState) {
    const plan = sessionState.getPlan();
    if (plan && plan.length > 0) {
      const inProgress = plan.find((s) => s.status === "in_progress");
      if (inProgress) {
        parts.push(`Active plan step: ${inProgress.description}`);
      }
      const completed = sessionState.getPlanCompletedCount();
      const total = sessionState.getTotalPlanCount();
      parts.push(`Plan progress: ${completed}/${total}`);
    }

    const modifiedFiles = sessionState.getModifiedFiles();
    if (modifiedFiles.length > 0) {
      parts.push(`Modified files: ${modifiedFiles.join(", ")}`);
    }

    const findings = sessionState.getFindings();
    if (findings.length > 0) {
      parts.push(`Findings: ${findings.length}`);
    }

    const summaries = sessionState.getToolSummaries();
    const recentSummaries = summaries.slice(-3);
    if (recentSummaries.length > 0) {
      parts.push("Recent tool activity:");
      for (const s of recentSummaries) {
        parts.push(`  ${s.tool}: ${s.summary.slice(0, 100)}`);
      }
    }
  }

  // Last user message
  const userMessages = messages.filter((m) => m.role === MessageRole.USER && m.content);
  const lastUser = userMessages[userMessages.length - 1];
  if (lastUser?.content) {
    const truncated = lastUser.content.length > 500
      ? lastUser.content.slice(0, 500) + "..."
      : lastUser.content;
    parts.push(`Last user message: ${truncated}`);
  }

  return parts.join("\n");
}
