/**
 * Lightweight engine setup for headless workflow execution.
 * Reuses DevAgent's core engine (provider, tools, TaskLoop) but
 * skips human-CLI-only systems and keeps bootstrap minimal.
 */

import { createDefaultRegistry } from "@devagent/providers";
import {
  AgentRegistry,
  AgentType,
  ApprovalGate,
  ApprovalMode,
  ContextManager,
  DEFAULT_BUDGET,
  DEFAULT_DOUBLE_CHECK_OPTIONS,
  DoubleCheck,
  EventBus,
  SessionState,
  TaskLoop,
  createDelegateTool,
  createFindingTool,
  createPlanTool,
  createSkillTool,
  createToolScriptTool,
  extractErrorMessage,
  loadConfig,
  loadModelRegistry,
  lookupModelEntry,
  resolveProviderCredentials,
} from "@devagent/runtime";
import { appendFileSync } from "node:fs";
import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

import { createShellTestRunner } from "./double-check-wiring.js";
import { resolveBundledModelsDir } from "./model-registry-path.js";
import { assembleSystemPrompt } from "./prompts/index.js";
import { buildProviderConfig } from "./provider-config.js";
import {
  formatProviderModelCompatibilityError,
  formatProviderModelCompatibilityHint,
  getProviderModelCompatibilityIssue,
} from "./provider-model-compat.js";
import { buildContextPrompt, loadRepoContext } from "./repo-context.js";
import { createSkillInfrastructure } from "./skill-setup.js";
import { detectProjectTestCommand } from "./test-command-detect.js";
import type {
  ApprovalPolicy,
  DevAgentConfig,
  FinalTextValidator,
  LLMProvider,
  Message,
  SessionStateJSON,
  SkillRegistry,
  ToolRegistry,
} from "@devagent/runtime";
import type { ContinuationSession } from "@devagent-sdk/types";

interface WorkflowQueryOptions {
  query: string;
  taskType?: string;
  repoPath: string;
  provider?: string;
  model?: string;
  maxIterations?: number;
  approvalMode: string;
  reasoning?: string;
  eventsPath: string;
  requestedSkills?: string[];
  continuation?: {
    session?: ContinuationSession;
    mode?: "fresh" | "resume";
    reason?: string;
    instructions?: string;
  };
}

interface WorkflowQueryResult {
  success: boolean;
  responseText: string;
  iterations: number;
  session?: ContinuationSession;
  outcome?: "completed" | "no_progress";
  outcomeReason?: "no_code" | "iteration_limit" | "empty_artifact" | "no_repo_changes";
}

type HeadlessSessionPayload = {
  version: 1;
  messages: Message[];
  sessionState?: SessionStateJSON;
};

interface ResolvedWorkflowContinuation {
  readonly query: string;
  readonly initialMessages: Message[] | undefined;
  readonly sessionState: SessionStateJSON | undefined;
}

interface WorkflowConfigResult {
  readonly config: DevAgentConfig;
  readonly approvalMode: ApprovalMode;
}

interface WorkflowToolSetup {
  readonly skills: SkillRegistry;
  readonly toolRegistry: ToolRegistry;
}

interface WorkflowLoopSetup {
  readonly bus: EventBus;
  readonly config: DevAgentConfig;
  readonly contextManager: ContextManager;
  readonly doubleCheck: DoubleCheck;
  readonly gate: ApprovalGate;
  readonly provider: LLMProvider;
  readonly sessionState: SessionState;
  readonly skills: SkillRegistry;
  readonly systemPrompt: string;
  readonly toolRegistry: ToolRegistry;
}

const PLAN_FREE_WORKFLOW_TASK_TYPES = new Set([
  "task-intake",
  "design",
  "breakdown",
  "issue-generation",
  "triage",
  "plan",
  "test-plan",
  "completion",
]);

const STRICT_STRUCTURED_WORKFLOW_TASK_TYPES = new Set([
  "breakdown",
  "issue-generation",
]);

const READONLY_WORKFLOW_TASK_TYPES = new Set([
  "task-intake",
  "design",
  "breakdown",
  "issue-generation",
  "triage",
  "plan",
  "test-plan",
  "review",
  "completion",
]);

const EXTRA_DEFERRED_READONLY_WORKFLOW_TOOLS = new Set([
  "write_file",
  "replace_in_file",
  "run_command",
]);

export function shouldEnableWorkflowPlanTool(taskType: string | undefined): boolean {
  return !taskType || !PLAN_FREE_WORKFLOW_TASK_TYPES.has(taskType);
}

export function createWorkflowFinalTextValidator(
  taskType: string | undefined,
): FinalTextValidator | undefined {
  if (!taskType || !STRICT_STRUCTURED_WORKFLOW_TASK_TYPES.has(taskType)) {
    return undefined;
  }

  return (candidate) => {
    const trimmed = candidate.trim();
    if (!trimmed.startsWith("{") || !trimmed.endsWith("}")) {
      return {
        valid: false,
        retryMessage: buildStrictWorkflowRetryMessage(taskType),
      };
    }

    try {
      const parsed = JSON.parse(trimmed);
      if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) {
        return {
          valid: false,
          retryMessage: buildStrictWorkflowRetryMessage(taskType),
        };
      }

      const envelope = parsed as Record<string, unknown>;
      if (!("structured" in envelope) || typeof envelope["rendered"] !== "string") {
        return {
          valid: false,
          retryMessage: buildStrictWorkflowRetryMessage(taskType),
        };
      }

      return { valid: true };
    } catch {
      return {
        valid: false,
        retryMessage: buildStrictWorkflowRetryMessage(taskType),
      };
    }
  };
}

function buildStrictWorkflowRetryMessage(taskType: string): string {
  return [
    `Your previous reply was a progress update or other non-final response for the ${taskType} stage.`,
    "This stage must end with a single JSON object now.",
    'Return exactly one top-level object containing "structured" and "rendered".',
    "Do not include commentary, code fences, progress sections, or any text before or after the JSON object.",
  ].join(" ");
}

function loadContinuationPayload(session: ContinuationSession | undefined): HeadlessSessionPayload | null {
  if (!session || session.kind !== "devagent-headless-v1") {
    return null;
  }
  const rawMessages = Array.isArray(session.payload.messages) ? session.payload.messages : null;
  if (!rawMessages) {
    return null;
  }
  return {
    version: 1,
    messages: rawMessages as Message[],
    sessionState: session.payload.sessionState as SessionStateJSON | undefined,
  };
}
export function resolveWorkflowContinuation(
  query: string,
  options: WorkflowQueryOptions["continuation"],
): ResolvedWorkflowContinuation {
  const previousSession = loadContinuationPayload(options?.session);
  const isResume = options?.mode === "resume";
  const parts = buildContinuationQueryParts(query, options, isResume);

  return {
    query: parts.length === 1 ? query : parts.join("\n\n"),
    initialMessages: isResume ? previousSession?.messages : undefined,
    sessionState: isResume ? previousSession?.sessionState : undefined,
  };
}

function buildContinuationQueryParts(
  query: string,
  options: WorkflowQueryOptions["continuation"],
  isResume: boolean,
): string[] {
  return [
    isResume ? "Continue the prior session using the preserved context below." : "",
    options?.reason ? `Continuation reason: ${options.reason}` : "",
    options?.instructions ? `Continuation instructions:\n${options.instructions}` : "",
    query,
  ].filter(Boolean);
}

async function loadWorkflowConfig(options: WorkflowQueryOptions): Promise<WorkflowConfigResult> {
  const approvalMode = resolveWorkflowApprovalMode(options.approvalMode);
  const configOverrides: Partial<DevAgentConfig> = {
    ...(options.provider ? { provider: options.provider } : {}),
    ...(options.model ? { model: options.model } : {}),
    approval: { mode: approvalMode } as ApprovalPolicy,
  };
  let config = await resolveProviderCredentials(loadConfig(options.repoPath, configOverrides));
  if (options.maxIterations) {
    config = { ...config, budget: { ...config.budget, maxIterations: options.maxIterations } };
  }
  loadBundledModelRegistry(options.repoPath);
  assertWorkflowProviderModel(config);
  return { config: applyModelRegistryBudget(config), approvalMode };
}

function resolveWorkflowApprovalMode(value: string): ApprovalMode {
  return (
    { "suggest": ApprovalMode.SUGGEST, "auto-edit": ApprovalMode.AUTO_EDIT, "full-auto": ApprovalMode.FULL_AUTO }
  )[value] ?? ApprovalMode.FULL_AUTO;
}

function loadBundledModelRegistry(projectRoot: string): void {
  const cliDir = dirname(fileURLToPath(import.meta.url));
  loadModelRegistry(projectRoot, [resolveBundledModelsDir(cliDir)]);
}

function assertWorkflowProviderModel(config: DevAgentConfig): void {
  const providerModelIssue = getProviderModelCompatibilityIssue(config.provider, config.model);
  if (!providerModelIssue) {
    return;
  }
  const hint = formatProviderModelCompatibilityHint(providerModelIssue);
  const message = hint
    ? `${formatProviderModelCompatibilityError(providerModelIssue)} ${hint}`
    : formatProviderModelCompatibilityError(providerModelIssue);
  throw new Error(message);
}

function applyModelRegistryBudget(config: DevAgentConfig): DevAgentConfig {
  const registryEntry = lookupModelEntry(config.model, config.provider);
  if (!registryEntry || config.budget.maxContextTokens !== DEFAULT_BUDGET.maxContextTokens) {
    return config;
  }
  return {
    ...config,
    budget: {
      ...config.budget,
      maxContextTokens: registryEntry.contextWindow,
      responseHeadroom: registryEntry.responseHeadroom,
    },
  };
}

function createWorkflowSessionState(
  continuation: ResolvedWorkflowContinuation,
  config: DevAgentConfig,
): SessionState {
  return continuation.sessionState
    ? SessionState.fromJSON(continuation.sessionState, config.sessionState)
    : new SessionState(config.sessionState);
}

function setupWorkflowTools(
  options: WorkflowQueryOptions,
  setup: Pick<WorkflowLoopSetup, "bus" | "config" | "gate" | "provider" | "sessionState">,
): WorkflowToolSetup {
  const { skills, skillResolver, skillAccess, toolRegistry } = createSkillInfrastructure(
    options.repoPath,
    setup.sessionState,
    READONLY_WORKFLOW_TASK_TYPES.has(options.taskType ?? "")
      ? { additionalDeferredToolNames: EXTRA_DEFERRED_READONLY_WORKFLOW_TOOLS }
      : undefined,
  );
  if (shouldEnableWorkflowPlanTool(options.taskType)) {
    toolRegistry.register(createPlanTool(setup.bus, () => setup.sessionState, () => 0, async () => null));
  }
  toolRegistry.register(createFindingTool(() => setup.sessionState, () => 0));
  toolRegistry.register(createSkillTool(skills, skillResolver, { skillAccess }));
  registerWorkflowDelegateTool(options, { ...setup, skills, toolRegistry });
  return { skills, toolRegistry };
}

function registerWorkflowDelegateTool(
  options: WorkflowQueryOptions,
  setup: Pick<WorkflowLoopSetup, "bus" | "config" | "gate" | "provider" | "sessionState" | "skills" | "toolRegistry">,
): void {
  setup.toolRegistry.register(createToolScriptTool({ registry: setup.toolRegistry, bus: setup.bus }));
  const providerRegistry = createDefaultRegistry();
  setup.toolRegistry.register(createDelegateTool({
    provider: setup.provider,
    tools: setup.toolRegistry,
    bus: setup.bus,
    approvalGate: setup.gate,
    config: setup.config,
    repoRoot: options.repoPath,
    agentRegistry: new AgentRegistry(),
    parentAgentId: "workflow",
    getParentSessionState: () => setup.sessionState,
    depth: 0,
    parentAgentType: AgentType.GENERAL,
    ambient: {
      skills: setup.skills,
      approvalMode: options.approvalMode,
      providerLabel: `${setup.config.provider} / ${setup.config.model}`,
      providerFactory: (agentConfig, agentType) => providerRegistry.get(
        agentConfig.provider,
        buildProviderConfig(agentConfig, options.reasoning as "low" | "medium" | "high" | "xhigh" | undefined, agentType),
      ),
    },
  }));
}

function createWorkflowDoubleCheck(config: DevAgentConfig, bus: EventBus, projectRoot: string): DoubleCheck {
  const dcEnabled = config.doubleCheck?.enabled ?? config.approval.mode === ApprovalMode.FULL_AUTO;
  const autoTestCommand = resolveAutoTestCommand(config, dcEnabled, projectRoot);
  const doubleCheck = new DoubleCheck({
    ...DEFAULT_DOUBLE_CHECK_OPTIONS,
    ...config.doubleCheck,
    enabled: dcEnabled,
    runTests: config.doubleCheck?.runTests ?? (autoTestCommand !== null),
    testCommand: config.doubleCheck?.testCommand ?? autoTestCommand,
  }, bus);
  if (autoTestCommand) {
    doubleCheck.setTestRunner(createShellTestRunner(projectRoot));
  }
  return doubleCheck;
}

function resolveAutoTestCommand(config: DevAgentConfig, enabled: boolean, projectRoot: string): string | null {
  if (!enabled || config.doubleCheck?.testCommand) {
    return null;
  }
  return detectProjectTestCommand(projectRoot);
}

function appendWorkflowToolEvents(bus: EventBus, eventsPath: string): void {
  bus.on("tool:after", (event) => {
    const line = JSON.stringify({
      type: "tool_call",
      tool: event.name,
      callId: event.callId,
      ...(event.batchId ? { batchId: event.batchId } : {}),
      ...(typeof event.batchSize === "number" ? { batchSize: event.batchSize } : {}),
      timestamp: new Date().toISOString(),
    }) + "\n";
    try { appendFileSync(eventsPath, line); } catch { /* best-effort */ }
  });
}

function buildWorkflowSystemPrompt(options: WorkflowQueryOptions, setup: WorkflowLoopSetup): string {
  const baseSystemPrompt = assembleSystemPrompt({
    mode: "act",
    skills: setup.skills,
    repoRoot: options.repoPath,
    availableTools: setup.toolRegistry.getLoaded(),
    deferredTools: setup.toolRegistry.getDeferred(),
    approvalMode: options.approvalMode,
    provider: options.provider,
    model: options.model,
    agentModelOverrides: setup.config.agentModelOverrides,
    agentReasoningOverrides: setup.config.agentReasoningOverrides,
  });
  const repoContextPrompt = buildContextPrompt(loadRepoContext(options.repoPath));
  const requestedSkillsPrompt = options.requestedSkills?.length
    ? `## Requested Skills\n\nThese skills were explicitly requested for this task: ${options.requestedSkills.join(", ")}.`
    : "";
  return [baseSystemPrompt, repoContextPrompt, requestedSkillsPrompt].filter(Boolean).join("\n\n");
}

function createWorkflowTaskLoop(
  options: WorkflowQueryOptions,
  continuation: ResolvedWorkflowContinuation,
  setup: WorkflowLoopSetup,
): TaskLoop {
  return new TaskLoop({
    provider: setup.provider,
    tools: setup.toolRegistry,
    bus: setup.bus,
    approvalGate: setup.gate,
    config: setup.config,
    systemPrompt: setup.systemPrompt,
    repoRoot: options.repoPath,
    mode: "act",
    contextManager: setup.contextManager,
    doubleCheck: setup.doubleCheck,
    sessionState: setup.sessionState,
    initialMessages: continuation.initialMessages,
    finalTextValidator: createWorkflowFinalTextValidator(options.taskType),
  });
}

function classifyWorkflowRunStatus(status: string): Pick<WorkflowQueryResult, "outcome" | "outcomeReason" | "success"> {
  if (status === "budget_exceeded") {
    return { outcome: "no_progress", outcomeReason: "iteration_limit", success: false };
  }
  if (status === "empty_response" || status === "aborted") {
    return { outcome: "no_progress", outcomeReason: "no_code", success: false };
  }
  return { outcome: "completed", success: true };
}
export async function setupAndRunWorkflowQuery(
  options: WorkflowQueryOptions,
): Promise<WorkflowQueryResult> {
  const { config } = await loadWorkflowConfig(options);
  const providerRegistry = createDefaultRegistry();
  const provider = providerRegistry.get(config.provider, buildProviderConfig(config, options.reasoning as "low" | "medium" | "high" | "xhigh" | undefined));
  const bus = new EventBus();
  const gate = new ApprovalGate(config.approval, bus);
  const continuation = resolveWorkflowContinuation(options.query, options.continuation);
  const sessionState = createWorkflowSessionState(continuation, config);
  const contextManager = new ContextManager(config.context);
  const doubleCheck = createWorkflowDoubleCheck(config, bus, options.repoPath);
  const tools = setupWorkflowTools(options, { bus, config, gate, provider, sessionState });
  appendWorkflowToolEvents(bus, options.eventsPath);
  const setup = {
    bus, config, contextManager, doubleCheck, gate, provider, sessionState,
    skills: tools.skills, systemPrompt: "", toolRegistry: tools.toolRegistry,
  } satisfies WorkflowLoopSetup;
  const loopSetup = { ...setup, systemPrompt: buildWorkflowSystemPrompt(options, setup) };
  const taskLoop = createWorkflowTaskLoop(options, continuation, loopSetup);

  let responseText = "";
  let iterations = 0;
  let success = true;
  let resultSession: ContinuationSession | undefined;
  let outcome: WorkflowQueryResult["outcome"] = "completed";
  let outcomeReason: WorkflowQueryResult["outcomeReason"] | undefined;
  const keepalive = setInterval(() => {
    // Keep Bun's event loop alive while the headless workflow request is pending.
  }, 1_000);

  try {
    const result = await taskLoop.run(continuation.query);
    responseText = result.lastText ?? "";
    iterations = result.iterations;
    success = !result.aborted;
    resultSession = {
      kind: "devagent-headless-v1",
      payload: {
        version: 1,
        messages: result.messages,
        sessionState: sessionState.toJSON(),
      },
    };
    ({ outcome, outcomeReason, success } = classifyWorkflowRunStatus(result.status));
  } catch (err) {
    responseText = extractErrorMessage(err);
    success = false;
    outcome = "no_progress";
  } finally {
    clearInterval(keepalive);
  }

  return { success, responseText, iterations, session: resultSession, outcome, outcomeReason };
}
