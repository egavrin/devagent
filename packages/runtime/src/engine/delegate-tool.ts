/**
 * delegate — Spawns a subagent to handle a subtask.
 * Category: workflow.
 *
 * This tool lives in the engine package (not tools) to avoid circular
 * dependencies: it needs access to runAgent which depends on TaskLoop.
 */

import type { ToolSpec, LLMProvider, DevAgentConfig, Message } from "../core/index.js";
import { AgentType, EventBus, ApprovalGate, extractErrorMessage } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";
import { AgentRegistry, runAgent, runForkedAgent } from "./agents.js";
import type { AgentAmbientContext } from "./agents.js";
import { parseAgentType } from "./agent-type.js";
import type { SessionState } from "./session-state.js";
import { judgeSubagentOutput } from "./subagent-judge.js";
import {
  buildDelegationQuery,
  normalizeDelegationRequest,
} from "./subagent-contract.js";

/** Hard cap on subagent iterations to prevent runaway loops. */
export const SUBAGENT_MAX_ITERATIONS = 30;
export const MAX_DELEGATION_DEPTH = 1;

/** Default per-agent-type iteration caps. */
const DEFAULT_AGENT_ITERATION_CAPS: Readonly<Partial<Record<AgentType, number>>> = {
  [AgentType.GENERAL]: 30,
  [AgentType.REVIEWER]: 18,
  [AgentType.ARCHITECT]: 20,
  [AgentType.EXPLORE]: 15,
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
  /** Parent's current messages — needed for fork mode (prompt cache sharing). */
  readonly getParentMessages?: () => ReadonlyArray<Message>;
  /** Parent's system prompt — needed for fork mode (cache prefix alignment). */
  readonly parentSystemPrompt?: string;
  readonly depth?: number;
  readonly parentAgentType?: AgentType;
  readonly ambient?: AgentAmbientContext;
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
        parallel_safe: {
          type: "boolean",
          description:
            "Set to true when this subagent's task is independent (no shared file mutations) and can run concurrently with other parallel_safe delegates.",
        },
        fork: {
          type: "boolean",
          description:
            "Set to true to fork the subagent with the parent's full conversation context. Enables prompt cache sharing for efficiency. Cannot be nested (forked children cannot fork again).",
        },
        request: {
          type: "object",
          description:
            "Structured delegation request with objective, optional scope, constraints, successCriteria, and parentContext.",
          properties: {
            objective: {
              type: "string",
              description: "Primary objective for the child agent.",
            },
            laneLabel: {
              type: ["string", "null"],
              description: "Optional short label describing the evidence lane for this child agent.",
            },
            scope: {
              type: ["string", "null"],
              description: "Optional scope boundary for the child agent.",
            },
            constraints: {
              type: ["array", "null"],
              description: "Optional list of constraints the child agent must follow.",
              items: {
                type: "string",
              },
            },
            exclusions: {
              type: ["array", "null"],
              description: "Optional list of concerns that are explicitly out of scope for the child agent.",
              items: {
                type: "string",
              },
            },
            successCriteria: {
              type: ["array", "null"],
              description: "Optional success criteria for the child agent result.",
              items: {
                type: "string",
              },
            },
            parentContext: {
              type: ["string", "null"],
              description: "Optional parent context explaining why delegation is needed.",
            },
          },
          required: ["objective"],
          additionalProperties: false,
        },
      },
      // Keep request non-null in strict provider schemas. OpenAI strict mode
      // rejects nullable object params unless the object variant itself is a
      // fully strict schema, so request must stay a plain object here.
      required: ["agent_type", "request"],
    },
    resultSchema: {
      type: "object",
      properties: {
        output: { type: "string" },
        iterations: { type: "number" },
        cost: { type: "object" },
      },
    },
    handler: async (params, toolContext) => {
      const agentTypeStr = params["agent_type"] as string;
      const forkMode = params["fork"] === true;
      const request = normalizeDelegationRequest(params["request"], params["task"]);

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
      if (!request) {
        return {
          success: false,
          output: "",
          error: "Missing delegation request. Provide `task` or `request.objective`.",
          artifacts: [],
        };
      }
      if (!isAllowedChildAgent(ctx.parentAgentType ?? AgentType.GENERAL, agentType, ctx.config)) {
        return {
          success: false,
          output: "",
          error: `Subagent (${agentTypeStr}) is not allowed from parent ${String(ctx.parentAgentType ?? AgentType.GENERAL)}.`,
          artifacts: [],
        };
      }

      subagentCounter++;
      const agentId = `${ctx.parentAgentId}-sub-${subagentCounter}`;

      const agentCap = resolveAgentIterationCap(ctx.config, agentType, request);
      const parentMax = ctx.config.budget.maxIterations;
      const cappedMax = parentMax > 0
        ? Math.min(parentMax, agentCap)
        : agentCap;
      const modelOverride = ctx.config.agentModelOverrides?.[agentType];
      const reasoningOverride = ctx.config.agentReasoningOverrides?.[agentType];
      const subagentConfig: DevAgentConfig = {
        ...ctx.config,
        ...(modelOverride ? { model: modelOverride } : {}),
        budget: { ...ctx.config.budget, maxIterations: cappedMax },
      };
      const resolvedReasoningEffort = reasoningOverride
        ?? subagentConfig.providers[subagentConfig.provider]?.reasoningEffort;
      const batchId = toolContext.batchId;
      const batchSize = toolContext.batchSize;
      const startedAt = Date.now();
      const childDepth = (ctx.depth ?? 0) + 1;

      try {
        ctx.bus.emit("subagent:start", {
          agentId,
          parentAgentId: ctx.parentAgentId,
          depth: childDepth,
          agentType,
          laneLabel: request.laneLabel ?? null,
          objective: request.objective,
          model: subagentConfig.model,
          reasoningEffort: resolvedReasoningEffort,
          status: "running",
          batchId,
          batchSize,
        });
        const query = buildDelegationQuery(request, cappedMax);

        // Fork mode: inherit parent's conversation context for prompt cache sharing
        if (forkMode && ctx.getParentMessages && ctx.parentSystemPrompt) {
          const forkResult = await runForkedAgent(
            query,
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
              depth: childDepth,
              ambient: ctx.ambient,
              parentMessages: ctx.getParentMessages(),
              parentSystemPrompt: ctx.parentSystemPrompt,
              laneLabel: request.laneLabel ?? null,
              batchId,
              batchSize,
            },
            ctx.agentRegistry,
          );
          const durationMs = Date.now() - startedAt;
          ctx.bus.emit("subagent:end", {
            agentId,
            parentAgentId: ctx.parentAgentId,
            depth: childDepth,
            agentType,
            laneLabel: request.laneLabel ?? null,
            objective: request.objective,
            model: subagentConfig.model,
            reasoningEffort: resolvedReasoningEffort,
            status: "completed",
            durationMs,
            iterations: forkResult.result.iterations,
            cost: forkResult.cost,
            parsedOutputKeys: [],
            batchId,
            batchSize,
          });
          const costSummary = `[${forkResult.result.iterations} iterations, fork mode]`;
          return {
            success: true,
            output: `Forked subagent completed ${costSummary}:\n\n${forkResult.finalMessage}`,
            error: null,
            artifacts: [],
            metadata: {
              agentMeta: forkResult.agentMeta,
              childSessionState: forkResult.childSessionState,
              delegateSummary: {
                agentId,
                agentType: agentTypeStr,
                laneLabel: request.laneLabel ?? null,
                durationMs,
                iterations: forkResult.result.iterations,
              },
            },
          };
        }

        const result = await runAgentWithQualityRetry(
          agentType,
          query,
          request.objective,
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
            depth: childDepth,
            ambient: {
              ...ctx.ambient,
              providerLabel: modelOverride
                ? `${subagentConfig.provider} / ${subagentConfig.model}`
                : ctx.ambient?.providerLabel,
            },
            createDelegateTool,
            laneLabel: request.laneLabel ?? null,
            batchId,
            batchSize,
          },
          ctx.agentRegistry,
          cappedMax,
          ctx.provider,
        );
        const durationMs = Date.now() - startedAt;
        const qualitySummary = result.judgeResult
          ? {
              score: result.judgeResult.quality_score,
              completeness: result.judgeResult.completeness,
              note: result.judgeResult.note,
            }
          : undefined;
        ctx.bus.emit("subagent:end", {
          agentId,
          parentAgentId: ctx.parentAgentId,
          depth: childDepth,
          agentType,
          laneLabel: request.laneLabel ?? null,
          objective: request.objective,
          model: subagentConfig.model,
          reasoningEffort: resolvedReasoningEffort,
          status: "completed",
          durationMs,
          iterations: result.run.result.iterations,
          cost: result.run.cost,
          parsedOutputKeys: result.run.parsedOutput ? Object.keys(result.run.parsedOutput) : [],
          quality: qualitySummary,
          batchId,
          batchSize,
        });

        const costSummary = `[${result.run.result.iterations} iterations, ${result.run.cost.inputTokens}+${result.run.cost.outputTokens} tokens]`;
        const qualityNote = result.qualityNote ? `${result.qualityNote}\n\n` : "";

        return {
          success: true,
          output: `${qualityNote}Subagent (${agentTypeStr}) completed ${costSummary}:\n\n${result.run.finalMessage}`,
          error: null,
          artifacts: [],
          metadata: {
            agentMeta: result.run.agentMeta,
            parsedOutput: result.run.parsedOutput,
            childSessionState: result.run.childSessionState,
            quality: qualitySummary,
            delegateSummary: {
              agentId,
              agentType: agentTypeStr,
              laneLabel: request.laneLabel ?? null,
              durationMs,
              iterations: result.run.result.iterations,
              quality: qualitySummary,
            },
          },
        };
      } catch (err) {
        const message = extractErrorMessage(err);
        ctx.bus.emit("subagent:error", {
          agentId,
          parentAgentId: ctx.parentAgentId,
          depth: childDepth,
          agentType,
          laneLabel: request.laneLabel ?? null,
          objective: request.objective,
          model: subagentConfig.model,
          reasoningEffort: resolvedReasoningEffort,
          status: "error",
          durationMs: Date.now() - startedAt,
          error: message,
          batchId,
          batchSize,
        });
        return {
          success: false,
          output: "",
          error: `Subagent (${agentTypeStr}) failed: ${message}`,
          artifacts: [],
          metadata: {
            agentMeta: {
              agentId,
              parentId: ctx.parentAgentId,
              depth: childDepth,
              agentType,
            },
          },
        };
      }
    },
  };
}

// ─── Helpers ────────────────────────────────────────────────

function resolveAgentIterationCap(
  config: DevAgentConfig,
  agentType: AgentType,
  request: { objective: string; scope?: string; constraints?: ReadonlyArray<string> },
): number {
  const override = config.agentIterationCaps?.[agentType];
  if (override !== undefined) return override;

  const objective = request.objective.toLowerCase();
  const shortLookup = objective.length < 80 &&
    /(find|locate|where|search|which file|what calls)/.test(objective);
  const broadAnalysis = objective.length > 160 ||
    objective.includes("dependency graph") ||
    objective.includes("across the codebase") ||
    (request.constraints?.length ?? 0) > 2;

  if (agentType === AgentType.EXPLORE) {
    return shortLookup ? 8 : broadAnalysis ? 20 : DEFAULT_AGENT_ITERATION_CAPS[AgentType.EXPLORE]!;
  }
  if (agentType === AgentType.REVIEWER || agentType === AgentType.ARCHITECT) {
    return broadAnalysis ? 24 : DEFAULT_AGENT_ITERATION_CAPS[agentType]!;
  }
  return broadAnalysis ? SUBAGENT_MAX_ITERATIONS : 18;
}

function isAllowedChildAgent(
  parentAgentType: AgentType,
  childAgentType: AgentType,
  config: DevAgentConfig,
): boolean {
  const allowed = config.allowedChildAgents?.[parentAgentType];
  return allowed === undefined || allowed.includes(childAgentType);
}

async function runAgentWithQualityRetry(
  agentType: AgentType,
  query: string,
  task: string,
  options: Parameters<typeof runAgent>[2],
  registry: AgentRegistry,
  cappedMax: number,
  judgeProvider: LLMProvider,
): Promise<{
  run: Awaited<ReturnType<typeof runAgent>>;
  judgeResult: Awaited<ReturnType<typeof judgeSubagentOutput>>;
  qualityNote: string;
}> {
  const firstRun = await runAgent(agentType, query, options, registry);
  const firstJudge = await maybeJudgeSubagent(judgeProvider, task, agentType, firstRun.finalMessage, firstRun.result.iterations, cappedMax);
  if (!firstJudge || firstJudge.quality_score >= judgeThreshold(agentType)) {
    // Secondary check: retry if completeness is "partial" even with high score,
    // but only when score is below the partial-retry threshold (0.75).
    // A score of 0.81 + "partial" is acceptable; 0.6 + "partial" warrants retry.
    const shouldRetryPartial = firstJudge &&
      firstJudge.completeness === "partial" &&
      firstJudge.quality_score < 0.75;
    if (!shouldRetryPartial) {
      return { run: firstRun, judgeResult: firstJudge, qualityNote: "" };
    }
  }

  const retryQuery = `${query}\n\nRetry note: the previous attempt was judged ${firstJudge.completeness}. Fix this issue explicitly: ${firstJudge.note}`;
  const secondRun = await runAgent(agentType, retryQuery, options, registry);
  const secondJudge = await maybeJudgeSubagent(judgeProvider, task, agentType, secondRun.finalMessage, secondRun.result.iterations, cappedMax);
  if (!secondJudge || secondJudge.quality_score >= judgeThreshold(agentType)) {
    return {
      run: secondRun,
      judgeResult: secondJudge,
      qualityNote: `[Subagent quality retry] Improved after retry.`,
    };
  }

  return {
    run: secondRun,
    judgeResult: secondJudge,
    qualityNote: `[Subagent escalation: ${secondJudge.completeness}] ${secondJudge.note}`,
  };
}

async function maybeJudgeSubagent(
  provider: LLMProvider,
  task: string,
  agentType: AgentType,
  output: string,
  iterationsUsed: number,
  maxIterations: number,
) {
  if (iterationsUsed < 5) return null;
  try {
    return await judgeSubagentOutput(
      provider,
      task,
      agentType,
      output,
      iterationsUsed,
      maxIterations,
    );
  } catch {
    return null;
  }
}

function judgeThreshold(agentType: AgentType): number {
  switch (agentType) {
    case AgentType.EXPLORE:
    case AgentType.REVIEWER:
      return 0.55;
    case AgentType.ARCHITECT:
      return 0.5;
    case AgentType.GENERAL:
    default:
      return 0.45;
  }
}
