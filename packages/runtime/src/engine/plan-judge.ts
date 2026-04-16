/**
 * Plan Quality Judge — assesses semantic quality of structured task plans.
 *
 * Evaluates: relevance to user request, completeness, step granularity,
 * drift from previous plans, dependency ordering.
 */

import type { LLMProvider } from "../core/index.js";
import { MessageRole } from "../core/index.js";
import type { SessionState } from "./session-state.js";
import type { PlanStep } from "./plan-tool.js";
import {
  collectStreamText,
  parseJudgeResponse,
  buildSessionStateContext,
} from "./llm-judge.js";

// Re-export isStructuralChange from plan-tool to avoid circular dependency
// (it lives in plan-tool.ts since plan-tool needs it directly)
export { isStructuralChange } from "./plan-tool.js";

// ─── Types ───────────────────────────────────────────────────

interface PlanJudgeResult {
  quality_score: number;
  issues: string[];
  suggestion: string | null;
}

// ─── System prompt ───────────────────────────────────────────

const PLAN_JUDGE_SYSTEM_PROMPT = `You assess the quality of a structured task plan created by an AI coding assistant.

Evaluate the plan against the original user request and these quality criteria:

- **Relevance**: Do all steps contribute to the user's request? Are there unrelated steps?
- **Completeness**: Are there obvious gaps? (e.g., no testing step for code changes, no verification step)
- **Granularity**: Are steps appropriately sized? "Implement everything" is too broad. "Add import statement" is too narrow.
- **Drift**: If a previous plan exists, has the new plan diverged from the original goal?
- **Ordering**: Are dependencies respected? (e.g., don't test before implementing)

NOT a quality issue: Minor wording differences between old/new plans. Adding verification steps. Reordering steps if dependencies are maintained.

Respond ONLY with valid JSON (no markdown fences, no commentary):
{"quality_score": 0.0-1.0, "issues": ["specific problem found", ...], "suggestion": "how to improve or null"}`;

// ─── Judge function ──────────────────────────────────────────

export async function judgePlanQuality(
  provider: LLMProvider,
  originalRequest: string,
  currentPlan: ReadonlyArray<PlanStep>,
  previousPlan: ReadonlyArray<PlanStep> | null,
  sessionState: SessionState | null,
  iteration: number,
): Promise<PlanJudgeResult | null> {
  try {
    const formatPlan = (plan: ReadonlyArray<PlanStep>) =>
      plan.map((s) => `[${s.status}] ${s.description}`).join("\n");

    const parts: string[] = [
      `## Original user request\n${originalRequest.slice(0, 1000)}`,
      `## Current plan\n${formatPlan(currentPlan)}`,
    ];

    if (previousPlan) {
      parts.push(`## Previous plan\n${formatPlan(previousPlan)}`);
    }

    parts.push(`## Session context\n${buildSessionStateContext(sessionState)}`);
    parts.push(`## Current iteration: ${iteration}`);
    parts.push(
      "\nAssess the plan quality. Respond with JSON only.",
    );

    const messages = [
      { role: MessageRole.SYSTEM as const, content: PLAN_JUDGE_SYSTEM_PROMPT },
      { role: MessageRole.USER as const, content: parts.join("\n\n") },
    ];

    const responseText = await collectStreamText(provider, messages);
    return parseJudgeResponse<PlanJudgeResult>(responseText);
  } catch {
    return null;
  }
}
