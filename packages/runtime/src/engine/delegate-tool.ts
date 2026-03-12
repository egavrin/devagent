/**
 * delegate — Spawns a subagent to handle a subtask.
 * Category: workflow.
 *
 * This tool lives in the engine package (not tools) to avoid circular
 * dependencies: it needs access to runAgent which depends on TaskLoop.
 */

import type { ToolSpec, LLMProvider, DevAgentConfig } from "../core/index.js";
import { AgentType, EventBus, ApprovalGate , extractErrorMessage } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";
import { AgentRegistry, runAgent } from "./agents.js";
import type { SessionState } from "./session-state.js";
import { judgeSubagentOutput } from "./subagent-judge.js";

/** Hard cap on subagent iterations to prevent runaway loops. */
export const SUBAGENT_MAX_ITERATIONS = 30;

/** Per-agent-type iteration caps. Explore is faster — lower budget. */
const AGENT_ITERATION_CAPS: Readonly<Record<string, number>> = {
  explore: 15,
};

// ─── Types ──────────────────────────────────────────────────

export interface DelegateToolContext {
  readonly provider: LLMProvider;
  readonly tools: ToolRegistry;
  readonly bus: EventBus;
  readonly approvalGate: ApprovalGate;
  readonly config: DevAgentConfig;
  readonly repoRoot: string;
  readonly agentRegistry: AgentRegistry;
  readonly parentAgentId: string;
  /** Parent's session state — resolved lazily so resume can swap the instance. */
  readonly getParentSessionState?: () => SessionState | undefined;
}

// ─── Factory ────────────────────────────────────────────────

/**
 * Create the delegate tool bound to a specific execution context.
 * Must be called per-session since it needs the provider and tools.
 */
export function createDelegateTool(ctx: DelegateToolContext): ToolSpec {
  let subagentCounter = 0;

  return {
    name: "delegate",
    description:
      "Spawn a subagent to handle a subtask. Choose the agent type based on the task: 'explore' for codebase search, 'general' for implementation, 'reviewer' for code review, 'architect' for design/planning.",
    category: "workflow",
    errorGuidance: {
      common: "Verify the agent type is valid (explore, general, reviewer, architect). Ensure the task description is specific and actionable.",
    },
    paramSchema: {
      type: "object",
      properties: {
        agent_type: {
          type: "string",
          description:
            "Agent type: 'explore', 'general', 'reviewer', or 'architect'",
        },
        task: {
          type: "string",
          description: "The task description for the subagent",
        },
      },
      required: ["agent_type", "task"],
    },
    resultSchema: {
      type: "object",
      properties: {
        output: { type: "string" },
        iterations: { type: "number" },
        cost: { type: "object" },
      },
    },
    handler: async (params) => {
      const agentTypeStr = params["agent_type"] as string;
      const task = params["task"] as string;

      // Map string to AgentType enum
      const agentType = parseAgentType(agentTypeStr);
      if (!agentType) {
        return {
          success: false,
          output: "",
          error: `Invalid agent type: "${agentTypeStr}". Use 'explore', 'general', 'reviewer', or 'architect'.`,
          artifacts: [],
        };
      }

      subagentCounter++;
      const agentId = `${ctx.parentAgentId}-sub-${subagentCounter}`;

      // Cap subagent iterations: use the smaller of parent's budget and the hard cap.
      // When parent has maxIterations: 0 (unlimited), use the hard cap.
      const agentCap = AGENT_ITERATION_CAPS[agentTypeStr.toLowerCase()] ?? SUBAGENT_MAX_ITERATIONS;
      const parentMax = ctx.config.budget.maxIterations;
      const cappedMax = parentMax > 0
        ? Math.min(parentMax, agentCap)
        : agentCap;
      const subagentConfig: DevAgentConfig = {
        ...ctx.config,
        budget: { ...ctx.config.budget, maxIterations: cappedMax },
      };

      try {
        const result = await runAgent(
          agentType,
          task,
          {
            provider: ctx.provider,
            tools: ctx.tools,
            bus: ctx.bus,
            approvalGate: ctx.approvalGate,
            config: subagentConfig,
            repoRoot: ctx.repoRoot,
            parentId: ctx.parentAgentId,
            agentId,
            parentSessionState: ctx.getParentSessionState?.(),
          },
          ctx.agentRegistry,
        );

        // Extract the final assistant message
        const assistantMessages = result.result.messages.filter(
          (m) => m.content && m.role === "assistant",
        );
        const finalMessage =
          assistantMessages[assistantMessages.length - 1]?.content ??
          "(no output)";

        const costSummary = `[${result.result.iterations} iterations, ${result.cost.inputTokens}+${result.cost.outputTokens} tokens]`;

        // Validate subagent output quality for non-trivial tasks
        let qualityNote = "";
        if (result.result.iterations >= 5) {
          try {
            const judgeResult = await judgeSubagentOutput(
              ctx.provider, task, agentTypeStr, finalMessage,
              result.result.iterations, cappedMax,
            );
            if (judgeResult && judgeResult.quality_score < 0.4) {
              qualityNote = `[Subagent quality: ${judgeResult.completeness}] ${judgeResult.note}\n\n`;
            }
          } catch {
            // Judge failure is non-fatal
          }
        }

        return {
          success: true,
          output: `${qualityNote}Subagent (${agentTypeStr}) completed ${costSummary}:\n\n${finalMessage}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        const message = extractErrorMessage(err);
        return {
          success: false,
          output: "",
          error: `Subagent (${agentTypeStr}) failed: ${message}`,
          artifacts: [],
        };
      }
    },
  };
}

// ─── Helpers ────────────────────────────────────────────────

function parseAgentType(str: string): AgentType | null {
  switch (str.toLowerCase()) {
    case "general":
      return AgentType.GENERAL;
    case "reviewer":
      return AgentType.REVIEWER;
    case "architect":
      return AgentType.ARCHITECT;
    case "explore":
      return AgentType.EXPLORE;
    default:
      return null;
  }
}
