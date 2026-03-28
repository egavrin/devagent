/**
 * Sub-Agent Validation Judge — assesses quality of subagent output
 * before returning it to the parent agent.
 *
 * Detects empty, off-topic, or incomplete subagent results
 * and provides a quality signal to the parent agent.
 */

import type { LLMProvider } from "../core/index.js";
import { AgentType, MessageRole } from "../core/index.js";
import { parseAgentType } from "./agent-type.js";
import { collectStreamText, parseJudgeResponse } from "./llm-judge.js";

// ─── Types ───────────────────────────────────────────────────

export interface SubagentJudgeResult {
  quality_score: number;
  completeness: "complete" | "partial" | "off_topic" | "empty";
  note: string;
}

// ─── System prompt ───────────────────────────────────────────

const BASE_SUBAGENT_JUDGE_SYSTEM_PROMPT = `You assess whether a subagent completed its assigned task.

Evaluate the output against the task description:
- Does the output answer/address the task?
- Is the output actionable and specific?
- Was the effort proportionate? (30 iterations for a simple question = problem)

NOT a quality issue: Verbose but complete output. Agent using many iterations for genuinely complex task.
IS a quality issue: Empty or generic output. Output doesn't address the task. Agent hit max iterations without conclusion.

Respond ONLY with valid JSON (no markdown fences, no commentary):
{"quality_score": 0.0-1.0, "completeness": "complete|partial|off_topic|empty", "note": "brief assessment"}`;

const TYPE_SPECIFIC_GUIDANCE: Record<AgentType, string> = {
  [AgentType.EXPLORE]: "Explore agents must provide a direct answer plus concrete evidence such as file paths or line references.",
  [AgentType.REVIEWER]: "Reviewer agents must provide concrete findings or explicitly state that no issues were found.",
  [AgentType.ARCHITECT]: "Architect agents must provide actionable implementation steps, risks, or assumptions rather than vague advice.",
  [AgentType.GENERAL]: "General agents must provide a clear completion summary, touched files or checks run, and any unresolved work.",
};

// ─── Judge function ──────────────────────────────────────────

export async function judgeSubagentOutput(
  provider: LLMProvider,
  task: string,
  agentType: AgentType | string,
  output: string,
  iterationsUsed: number,
  maxIterations: number,
): Promise<SubagentJudgeResult | null> {
  // Gating: skip trivial tasks
  if (iterationsUsed < 5) return null;

  try {
    const truncatedOutput = output.length > 2000
      ? output.slice(0, 2000) + "..."
      : output;

    const parts: string[] = [
      `## Task assigned to subagent\n${task}`,
      `## Agent type: ${agentType}`,
      `## Iterations used: ${iterationsUsed}/${maxIterations}`,
      `## Subagent output\n${truncatedOutput}`,
      "\nAssess the subagent output quality. Respond with JSON only.",
    ];

    const messages = [
      {
        role: MessageRole.SYSTEM as const,
        content: `${BASE_SUBAGENT_JUDGE_SYSTEM_PROMPT}\n\n${TYPE_SPECIFIC_GUIDANCE[parseAgentType(agentType) ?? AgentType.GENERAL]}`,
      },
      { role: MessageRole.USER as const, content: parts.join("\n\n") },
    ];

    const responseText = await collectStreamText(provider, messages);
    return parseJudgeResponse<SubagentJudgeResult>(responseText);
  } catch {
    return null;
  }
}
