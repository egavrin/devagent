/**
 * delegate — Spawns a subagent to handle a subtask.
 * Category: workflow.
 *
 * This tool lives in the engine package (not tools) to avoid circular
 * dependencies: it needs access to runAgent which depends on TaskLoop.
 */

import { parseAgentType } from "./agent-type.js";
import { runAgent, runForkedAgent } from "./agents.js";
import type { AgentAmbientContext , AgentRegistry} from "./agents.js";
import type { SessionState } from "./session-state.js";
import {
  buildDelegationQuery,
  normalizeDelegationRequest,
} from "./subagent-contract.js";
import { judgeSubagentOutput } from "./subagent-judge.js";
import type {
  ApprovalGate,
  CostRecord,
  DevAgentConfig,
  EventBus,
  LLMProvider,
  LSPDocumentSync,
  Message,
  ReasoningEffort,
  ToolResult,
  ToolSpec,
} from "../core/index.js";
import { AgentType, extractErrorMessage } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";

/** Hard cap on subagent iterations to prevent runaway loops. */
const SUBAGENT_MAX_ITERATIONS = 30;

/** Default per-agent-type iteration caps. */
const DEFAULT_AGENT_ITERATION_CAPS: Readonly<Partial<Record<AgentType, number>>> = {
  [AgentType.GENERAL]: 30,
  [AgentType.REVIEWER]: 18,
  [AgentType.ARCHITECT]: 20,
  [AgentType.EXPLORE]: 15,
};

interface QualityRetryOptions {
  readonly agentType: AgentType;
  readonly query: string;
  readonly task: string;
  readonly options: Parameters<typeof runAgent>[2];
  readonly registry: AgentRegistry;
  readonly cappedMax: number;
  readonly judgeProvider: LLMProvider;
}

interface SubagentJudgeOptions {
  readonly provider: LLMProvider;
  readonly task: string;
  readonly agentType: AgentType;
  readonly output: string;
  readonly iterationsUsed: number;
  readonly maxIterations: number;
}

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
  readonly lspSync?: LSPDocumentSync;
  readonly depth?: number;
  readonly parentAgentType?: AgentType;
  readonly ambient?: AgentAmbientContext;
}

const DELEGATE_PARAM_SCHEMA = {
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
    };

const DELEGATE_RESULT_SCHEMA = {
      type: "object",
      properties: {
        output: { type: "string" },
        iterations: { type: "number" },
        cost: { type: "object" },
      },
    };

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
    paramSchema: DELEGATE_PARAM_SCHEMA,
    resultSchema: DELEGATE_RESULT_SCHEMA,
    handler: async (params, toolContext) => runDelegateTool(ctx, () => ++subagentCounter, params, toolContext),
  };
}


interface DelegateRunState {
  readonly agentTypeStr: string;
  readonly agentType: AgentType;
  readonly request: NonNullable<ReturnType<typeof normalizeDelegationRequest>>;
  readonly forkMode: boolean;
  readonly agentId: string;
  readonly cappedMax: number;
  readonly modelOverride: string | undefined;
  readonly subagentConfig: DevAgentConfig;
  readonly resolvedReasoningEffort: ReasoningEffort | undefined;
  readonly batchId: string | undefined;
  readonly batchSize: number | undefined;
  readonly startedAt: number;
  readonly lspSync: LSPDocumentSync | undefined;
  readonly childDepth: number;
  readonly query: string;
}

async function runDelegateTool(
  ctx: DelegateToolContext,
  nextSubagentNumber: () => number,
  params: Record<string, unknown>,
  toolContext: import("../core/index.js").ToolContext,
) {
  const prepared = prepareDelegateRun(ctx, nextSubagentNumber, params, toolContext);
  if ("failure" in prepared) return prepared.failure;

  const state = prepared.state;
  try {
    emitSubagentStart(ctx, state);
    if (state.forkMode && ctx.getParentMessages && ctx.parentSystemPrompt) {
      return await runForkDelegate(ctx, state);
    }
    return await runStandardDelegate(ctx, state);
  } catch (err) {
    return handleDelegateError(ctx, state, err);
  }
}

function prepareDelegateRun(
  ctx: DelegateToolContext,
  nextSubagentNumber: () => number,
  params: Record<string, unknown>,
  toolContext: import("../core/index.js").ToolContext,
): { state: DelegateRunState } | { failure: ToolResult } {
  const agentTypeStr = params["agent_type"] as string;
  const request = normalizeDelegationRequest(params["request"], params["task"]);
  const agentType = parseAgentType(agentTypeStr);
  const failure = validateDelegateRequest(ctx, agentTypeStr, agentType, request);
  if (failure) return { failure };

  const agentId = `${ctx.parentAgentId}-sub-${nextSubagentNumber()}`;
  const cappedMax = resolveCappedAgentIterations(ctx.config, agentType!, request!);
  const subagentConfig = buildSubagentConfig(ctx.config, agentType!, cappedMax);
  return {
    state: {
      agentTypeStr,
      agentType: agentType!,
      request: request!,
      forkMode: params["fork"] === true,
      agentId,
      cappedMax,
      modelOverride: ctx.config.agentModelOverrides?.[agentType!],
      subagentConfig,
      resolvedReasoningEffort: resolveSubagentReasoning(ctx.config, subagentConfig, agentType!),
      batchId: toolContext.batchId,
      batchSize: toolContext.batchSize,
      startedAt: Date.now(),
      lspSync: toolContext.lspSync ?? ctx.lspSync,
      childDepth: (ctx.depth ?? 0) + 1,
      query: buildDelegationQuery(request!, cappedMax),
    },
  };
}

function validateDelegateRequest(
  ctx: DelegateToolContext,
  agentTypeStr: string,
  agentType: AgentType | null | undefined,
  request: ReturnType<typeof normalizeDelegationRequest>,
): ToolResult | null {
  if (!agentType) {
    return toolFailure(`Invalid agent type: "${agentTypeStr}". Use 'explore', 'general', 'reviewer', or 'architect'.`);
  }
  if (!request) return toolFailure("Missing delegation request. Provide `task` or `request.objective`.");
  if (isAllowedChildAgent(ctx.parentAgentType ?? AgentType.GENERAL, agentType, ctx.config)) return null;
  return toolFailure(`Subagent (${agentTypeStr}) is not allowed from parent ${String(ctx.parentAgentType ?? AgentType.GENERAL)}.`);
}

function toolFailure(error: string): ToolResult {
  return { success: false, output: "", error, artifacts: [] };
}

function resolveCappedAgentIterations(
  config: DevAgentConfig,
  agentType: AgentType,
  request: NonNullable<ReturnType<typeof normalizeDelegationRequest>>,
) {
  const agentCap = resolveAgentIterationCap(config, agentType, request);
  return config.budget.maxIterations > 0 ? Math.min(config.budget.maxIterations, agentCap) : agentCap;
}

function buildSubagentConfig(config: DevAgentConfig, agentType: AgentType, cappedMax: number): DevAgentConfig {
  const modelOverride = config.agentModelOverrides?.[agentType];
  return {
    ...config,
    ...(modelOverride ? { model: modelOverride } : {}),
    budget: { ...config.budget, maxIterations: cappedMax },
  };
}

function resolveSubagentReasoning(
  config: DevAgentConfig,
  subagentConfig: DevAgentConfig,
  agentType: AgentType,
): ReasoningEffort | undefined {
  return config.agentReasoningOverrides?.[agentType]
    ?? subagentConfig.providers[subagentConfig.provider]?.reasoningEffort;
}

function emitSubagentStart(ctx: DelegateToolContext, state: DelegateRunState) {
  ctx.bus.emit("subagent:start", {
    agentId: state.agentId,
    parentAgentId: ctx.parentAgentId,
    depth: state.childDepth,
    agentType: state.agentType,
    laneLabel: state.request.laneLabel ?? null,
    objective: state.request.objective,
    model: state.subagentConfig.model,
    reasoningEffort: state.resolvedReasoningEffort,
    status: "running",
    batchId: state.batchId,
    batchSize: state.batchSize,
  });
}

async function runForkDelegate(ctx: DelegateToolContext, state: DelegateRunState): Promise<ToolResult> {
  const forkResult = await runForkedAgent(state.query, buildForkAgentOptions(ctx, state), ctx.agentRegistry);
  const durationMs = Date.now() - state.startedAt;
  emitSubagentEnd(ctx, state, {
    durationMs,
    iterations: forkResult.result.iterations,
    cost: forkResult.cost,
    parsedOutputKeys: [],
  });
  return {
    success: true,
    output: `Forked subagent completed [${forkResult.result.iterations} iterations, fork mode]:\n\n${forkResult.finalMessage}`,
    error: null,
    artifacts: [],
    metadata: {
      agentMeta: forkResult.agentMeta,
      childSessionState: forkResult.childSessionState,
      delegateSummary: buildDelegateSummary(state, durationMs, forkResult.result.iterations),
    },
  };
}

function buildForkAgentOptions(ctx: DelegateToolContext, state: DelegateRunState): Parameters<typeof runForkedAgent>[1] {
  return {
    provider: ctx.provider,
    tools: ctx.tools,
    bus: ctx.bus,
    approvalGate: ctx.approvalGate,
    config: state.subagentConfig,
    repoRoot: ctx.repoRoot,
    parentId: ctx.parentAgentId,
    agentId: state.agentId,
    parentSessionState: ctx.getParentSessionState?.(),
    lspSync: state.lspSync,
    depth: state.childDepth,
    ambient: ctx.ambient,
    parentMessages: ctx.getParentMessages!(),
    parentSystemPrompt: ctx.parentSystemPrompt!,
    laneLabel: state.request.laneLabel ?? null,
    batchId: state.batchId,
    batchSize: state.batchSize,
  };
}

async function runStandardDelegate(ctx: DelegateToolContext, state: DelegateRunState): Promise<ToolResult> {
  const result = await runAgentWithQualityRetry({
    agentType: state.agentType,
    query: state.query,
    task: state.request.objective,
    options: buildStandardAgentOptions(ctx, state),
    registry: ctx.agentRegistry,
    cappedMax: state.cappedMax,
    judgeProvider: ctx.provider,
  });
  const durationMs = Date.now() - state.startedAt;
  const qualitySummary = formatQualitySummary(result.judgeResult);
  emitSubagentEnd(ctx, state, {
    durationMs,
    iterations: result.run.result.iterations,
    cost: result.run.cost,
    parsedOutputKeys: result.run.parsedOutput ? Object.keys(result.run.parsedOutput) : [],
    quality: qualitySummary,
  });
  return buildStandardDelegateResult(state, result, durationMs, qualitySummary);
}

function buildStandardAgentOptions(ctx: DelegateToolContext, state: DelegateRunState): Parameters<typeof runAgent>[2] {
  return {
    provider: ctx.provider,
    tools: ctx.tools,
    bus: ctx.bus,
    approvalGate: ctx.approvalGate,
    config: state.subagentConfig,
    repoRoot: ctx.repoRoot,
    parentId: ctx.parentAgentId,
    agentId: state.agentId,
    parentSessionState: ctx.getParentSessionState?.(),
    lspSync: state.lspSync,
    depth: state.childDepth,
    ambient: {
      ...ctx.ambient,
      providerLabel: state.modelOverride
        ? `${state.subagentConfig.provider} / ${state.subagentConfig.model}`
        : ctx.ambient?.providerLabel,
    },
    createDelegateTool,
    laneLabel: state.request.laneLabel ?? null,
    batchId: state.batchId,
    batchSize: state.batchSize,
  };
}

function buildStandardDelegateResult(
  state: DelegateRunState,
  result: Awaited<ReturnType<typeof runAgentWithQualityRetry>>,
  durationMs: number,
  qualitySummary: ReturnType<typeof formatQualitySummary>,
): ToolResult {
  const costSummary = `[${result.run.result.iterations} iterations, ${result.run.cost.inputTokens}+${result.run.cost.outputTokens} tokens]`;
  const qualityNote = result.qualityNote ? `${result.qualityNote}\n\n` : "";
  return {
    success: true,
    output: `${qualityNote}Subagent (${state.agentTypeStr}) completed ${costSummary}:\n\n${result.run.finalMessage}`,
    error: null,
    artifacts: [],
    metadata: {
      agentMeta: result.run.agentMeta,
      parsedOutput: result.run.parsedOutput,
      childSessionState: result.run.childSessionState,
      quality: qualitySummary,
      delegateSummary: {
        ...buildDelegateSummary(state, durationMs, result.run.result.iterations),
        quality: qualitySummary,
      },
    },
  };
}

function formatQualitySummary(judgeResult: Awaited<ReturnType<typeof judgeSubagentOutput>>) {
  return judgeResult
    ? {
        score: judgeResult.quality_score,
        completeness: judgeResult.completeness,
        note: judgeResult.note,
      }
    : undefined;
}

function emitSubagentEnd(
  ctx: DelegateToolContext,
  state: DelegateRunState,
  result: {
    readonly durationMs: number;
    readonly iterations: number;
    readonly cost: CostRecord;
    readonly parsedOutputKeys: ReadonlyArray<string>;
    readonly quality?: ReturnType<typeof formatQualitySummary>;
  },
) {
  ctx.bus.emit("subagent:end", {
    agentId: state.agentId,
    parentAgentId: ctx.parentAgentId,
    depth: state.childDepth,
    agentType: state.agentType,
    laneLabel: state.request.laneLabel ?? null,
    objective: state.request.objective,
    model: state.subagentConfig.model,
    reasoningEffort: state.resolvedReasoningEffort,
    status: "completed",
    durationMs: result.durationMs,
    iterations: result.iterations,
    cost: result.cost,
    parsedOutputKeys: result.parsedOutputKeys,
    ...(result.quality ? { quality: result.quality } : {}),
    batchId: state.batchId,
    batchSize: state.batchSize,
  });
}

function buildDelegateSummary(state: DelegateRunState, durationMs: number, iterations: number) {
  return {
    agentId: state.agentId,
    agentType: state.agentTypeStr,
    laneLabel: state.request.laneLabel ?? null,
    durationMs,
    iterations,
  };
}

function handleDelegateError(ctx: DelegateToolContext, state: DelegateRunState, err: unknown): ToolResult {
  const message = extractErrorMessage(err);
  ctx.bus.emit("subagent:error", {
    agentId: state.agentId,
    parentAgentId: ctx.parentAgentId,
    depth: state.childDepth,
    agentType: state.agentType,
    laneLabel: state.request.laneLabel ?? null,
    objective: state.request.objective,
    model: state.subagentConfig.model,
    reasoningEffort: state.resolvedReasoningEffort,
    status: "error",
    durationMs: Date.now() - state.startedAt,
    error: message,
    batchId: state.batchId,
    batchSize: state.batchSize,
  });
  return {
    success: false,
    output: "",
    error: `Subagent (${state.agentTypeStr}) failed: ${message}`,
    artifacts: [],
    metadata: {
      agentMeta: {
        agentId: state.agentId,
        parentId: ctx.parentAgentId,
        depth: state.childDepth,
        agentType: state.agentType,
      },
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

  const broadAnalysis = isBroadDelegationRequest(request);
  if (agentType === AgentType.EXPLORE) return resolveExploreIterationCap(request, broadAnalysis);
  if (agentType === AgentType.REVIEWER || agentType === AgentType.ARCHITECT) {
    return broadAnalysis ? 24 : DEFAULT_AGENT_ITERATION_CAPS[agentType]!;
  }
  return broadAnalysis ? SUBAGENT_MAX_ITERATIONS : 18;
}

function resolveExploreIterationCap(
  request: { objective: string; scope?: string; constraints?: ReadonlyArray<string> },
  broadAnalysis: boolean,
) {
  if (isShortLookupRequest(request)) return 8;
  return broadAnalysis ? 20 : DEFAULT_AGENT_ITERATION_CAPS[AgentType.EXPLORE]!;
}

function isShortLookupRequest(request: { objective: string }) {
  const objective = request.objective.toLowerCase();
  return objective.length < 80 && /(find|locate|where|search|which file|what calls)/.test(objective);
}

function isBroadDelegationRequest(
  request: { objective: string; scope?: string; constraints?: ReadonlyArray<string> },
) {
  const objective = request.objective.toLowerCase();
  return objective.length > 160 ||
    objective.includes("dependency graph") ||
    objective.includes("across the codebase") ||
    (request.constraints?.length ?? 0) > 2;
}

function isAllowedChildAgent(
  parentAgentType: AgentType,
  childAgentType: AgentType,
  config: DevAgentConfig,
): boolean {
  const allowed = config.allowedChildAgents?.[parentAgentType];
  return allowed === undefined || allowed.includes(childAgentType);
}
async function runAgentWithQualityRetry(params: QualityRetryOptions): Promise<{
  run: Awaited<ReturnType<typeof runAgent>>;
  judgeResult: Awaited<ReturnType<typeof judgeSubagentOutput>>;
  qualityNote: string;
}> {
  const firstRun = await runAgent(params.agentType, params.query, params.options, params.registry);
  const firstJudge = await maybeJudgeSubagent(buildSubagentJudgeOptions(params, firstRun));
  if (!firstJudge || firstJudge.quality_score >= judgeThreshold(params.agentType)) {
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

  const retryQuery = `${params.query}\n\nRetry note: the previous attempt was judged ${firstJudge.completeness}. Fix this issue explicitly: ${firstJudge.note}`;
  const secondRun = await runAgent(params.agentType, retryQuery, params.options, params.registry);
  const secondJudge = await maybeJudgeSubagent(buildSubagentJudgeOptions(params, secondRun));
  if (!secondJudge || secondJudge.quality_score >= judgeThreshold(params.agentType)) {
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

function buildSubagentJudgeOptions(
  retry: QualityRetryOptions,
  run: Awaited<ReturnType<typeof runAgent>>,
): SubagentJudgeOptions {
  return {
    provider: retry.judgeProvider,
    task: retry.task,
    agentType: retry.agentType,
    output: run.finalMessage,
    iterationsUsed: run.result.iterations,
    maxIterations: retry.cappedMax,
  };
}

async function maybeJudgeSubagent(options: SubagentJudgeOptions) {
  if (options.iterationsUsed < 5) return null;
  try {
    return await judgeSubagentOutput(options);
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
