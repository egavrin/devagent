/**
 * delegate — Spawns a subagent to handle a subtask.
 * Category: workflow.
 *
 * This tool lives in the engine package (not tools) to avoid circular
 * dependencies: it needs access to runAgent which depends on TaskLoop.
 */

import type { ToolSpec, LLMProvider, DevAgentConfig } from "@devagent/core";
import { AgentType, EventBus, ApprovalGate } from "@devagent/core";
import type { ToolRegistry } from "@devagent/tools";
import { AgentRegistry, runAgent } from "./agents.js";

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
      "Spawn a subagent to handle a subtask. Choose the agent type based on the task: 'general' for implementation, 'reviewer' for code review, 'architect' for design/planning.",
    category: "workflow",
    paramSchema: {
      type: "object",
      properties: {
        agent_type: {
          type: "string",
          description:
            "Agent type: 'general', 'reviewer', or 'architect'",
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
          error: `Invalid agent type: "${agentTypeStr}". Use 'general', 'reviewer', or 'architect'.`,
          artifacts: [],
        };
      }

      subagentCounter++;
      const agentId = `${ctx.parentAgentId}-sub-${subagentCounter}`;

      try {
        const result = await runAgent(
          agentType,
          task,
          {
            provider: ctx.provider,
            tools: ctx.tools,
            bus: ctx.bus,
            approvalGate: ctx.approvalGate,
            config: ctx.config,
            repoRoot: ctx.repoRoot,
            parentId: ctx.parentAgentId,
            agentId,
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

        return {
          success: true,
          output: `Subagent (${agentTypeStr}) completed ${costSummary}:\n\n${finalMessage}`,
          error: null,
          artifacts: [],
        };
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
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
    default:
      return null;
  }
}
