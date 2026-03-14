/**
 * Lightweight engine setup for headless workflow execution.
 * Reuses DevAgent's core engine (provider, tools, TaskLoop) but
 * skips human-CLI-only systems and keeps bootstrap minimal.
 */

import { dirname } from "node:path";
import { fileURLToPath } from "node:url";
import { appendFileSync } from "node:fs";
import {
  EventBus,
  ApprovalGate,
  SkillLoader,
  SkillRegistry,
  SkillResolver,
  ContextManager,
  loadConfig,
  resolveProviderCredentials,
  loadModelRegistry,
  lookupModelCapabilities,
  lookupModelEntry,
  DEFAULT_BUDGET,
  ApprovalMode,
  extractErrorMessage,
} from "@devagent/runtime";
import type { DevAgentConfig, ApprovalPolicy } from "@devagent/runtime";
import { createDefaultRegistry } from "@devagent/providers";
import { createDefaultToolRegistry, ToolRegistry } from "@devagent/runtime";
import {
  TaskLoop,
  createPlanTool,
  createFindingTool,
  createToolScriptTool,
  createDelegateTool,
  createSkillTool,
  AgentRegistry,
  DoubleCheck,
  DEFAULT_DOUBLE_CHECK_OPTIONS,
  SessionState,
  type Message,
  type SessionStateJSON,
} from "@devagent/runtime";
import type { ContinuationSession } from "@devagent-sdk/types";
import { assembleSystemPrompt } from "./prompts/index.js";
import { detectProjectTestCommand } from "./test-command-detect.js";
import { loadRepoContext, buildContextPrompt } from "./repo-context.js";
import { resolveBundledModelsDir } from "./model-registry-path.js";
import { createShellTestRunner } from "./double-check-wiring.js";

export interface WorkflowQueryOptions {
  query: string;
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

export interface WorkflowQueryResult {
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

function buildContinuationQuery(query: string, options: WorkflowQueryOptions["continuation"]): string {
  if (!options || options.mode !== "resume") {
    return query;
  }
  const parts = [
    "Continue the prior session using the preserved context below.",
    options.reason ? `Continuation reason: ${options.reason}` : "",
    options.instructions ? `Continuation instructions:\n${options.instructions}` : "",
    query,
  ];
  return parts.filter(Boolean).join("\n\n");
}

export async function setupAndRunWorkflowQuery(
  options: WorkflowQueryOptions,
): Promise<WorkflowQueryResult> {
  const projectRoot = options.repoPath;

  // 1. Load config with CLI overrides
  const approvalMode = (
    { "suggest": ApprovalMode.SUGGEST, "auto-edit": ApprovalMode.AUTO_EDIT, "full-auto": ApprovalMode.FULL_AUTO }
  )[options.approvalMode] ?? ApprovalMode.FULL_AUTO;

  const configOverrides: Partial<DevAgentConfig> = {
    ...(options.provider ? { provider: options.provider } : {}),
    ...(options.model ? { model: options.model } : {}),
    approval: { mode: approvalMode } as ApprovalPolicy,
  };

  let config = loadConfig(projectRoot, configOverrides);
  config = await resolveProviderCredentials(config);

  if (options.maxIterations) {
    config = { ...config, budget: { ...config.budget, maxIterations: options.maxIterations } };
  }

  // Load model registry
  const cliDir = dirname(fileURLToPath(import.meta.url));
  const devagentModelsDir = resolveBundledModelsDir(cliDir);
  loadModelRegistry(projectRoot, [devagentModelsDir]);

  // Auto-size context budget from model registry
  const registryEntry = lookupModelEntry(config.model);
  if (registryEntry && config.budget.maxContextTokens === DEFAULT_BUDGET.maxContextTokens) {
    config = {
      ...config,
      budget: {
        ...config.budget,
        maxContextTokens: registryEntry.contextWindow,
        responseHeadroom: registryEntry.responseHeadroom,
      },
    };
  }

  // 2. Create provider
  const providerRegistry = createDefaultRegistry();
  const baseProviderConfig = config.providers[config.provider] ?? {
    model: config.model,
    apiKey: process.env["DEVAGENT_API_KEY"],
  };
  const registryCaps = lookupModelCapabilities(config.model);
  const providerConfig = {
    ...baseProviderConfig,
    ...(options.reasoning ? { reasoningEffort: options.reasoning as "low" | "medium" | "high" | "xhigh" } : {}),
    ...(!baseProviderConfig.capabilities && registryCaps ? { capabilities: registryCaps } : {}),
  };

  const provider = providerRegistry.get(config.provider, providerConfig);

  // 3. Set up tools (lightweight shared runtime bootstrap)
  const toolRegistry = createDefaultToolRegistry();
  const bus = new EventBus();
  const gate = new ApprovalGate(config.approval, bus);

  const previousSession = loadContinuationPayload(options.continuation?.session);
  const sessionState = previousSession?.sessionState
    ? SessionState.fromJSON(previousSession.sessionState, config.sessionState)
    : new SessionState(config.sessionState);

  // Plan tool (minimal — no quality judgment in headless mode)
  toolRegistry.register(createPlanTool(
    bus,
    () => sessionState,
    () => 0,
    async () => null,
  ));
  toolRegistry.register(createFindingTool(() => sessionState, () => 0));

  // Skills
  const skillLoader = new SkillLoader();
  const skillMetadata = skillLoader.discover({ repoRoot: projectRoot });
  const skills = new SkillRegistry();
  skills.register(skillMetadata);
  const skillResolver = new SkillResolver();
  toolRegistry.register(createSkillTool(skills, skillResolver));

  // Tool scripts + delegate
  toolRegistry.register(createToolScriptTool({ registry: toolRegistry, bus }));
  const agentRegistry = new AgentRegistry();
  toolRegistry.register(createDelegateTool({
    provider,
    tools: toolRegistry,
    bus,
    approvalGate: gate,
    config,
    repoRoot: projectRoot,
    agentRegistry,
    parentAgentId: "workflow",
    getParentSessionState: () => sessionState,
  }));

  // DoubleCheck (auto-enable in full-auto)
  const isFullAuto = config.approval.mode === ApprovalMode.FULL_AUTO;
  const dcEnabled = config.doubleCheck?.enabled ?? isFullAuto;
  const autoTestCommand = dcEnabled && !config.doubleCheck?.testCommand
    ? detectProjectTestCommand(projectRoot)
    : null;
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

  // Context
  const contextManager = new ContextManager(config.context);

  // 4. Log events to JSONL file
  bus.on("tool:after", (event) => {
    const line = JSON.stringify({
      type: "tool_call",
      tool: event.name,
      timestamp: new Date().toISOString(),
    }) + "\n";
    try { appendFileSync(options.eventsPath, line); } catch { /* best-effort */ }
  });

  // 5. Load repo context (WORKFLOW.md, AGENTS.md, instructions)
  const repoContext = loadRepoContext(projectRoot);
  const repoContextPrompt = buildContextPrompt(repoContext);

  // 6. Assemble system prompt
  const baseSystemPrompt = assembleSystemPrompt({
    mode: "act",
    skills,
    repoRoot: projectRoot,
    approvalMode: options.approvalMode,
    provider: options.provider,
    model: options.model,
  });

  const requestedSkillsPrompt = options.requestedSkills?.length
    ? `## Requested Skills\n\nThese skills were explicitly requested for this task: ${options.requestedSkills.join(", ")}.`
    : "";

  // Append repo context after the base system prompt
  const contextSections = [baseSystemPrompt, repoContextPrompt, requestedSkillsPrompt].filter(Boolean);
  const systemPrompt = contextSections.join("\n\n");

  // 6. Run TaskLoop
  const taskLoop = new TaskLoop({
    provider,
    tools: toolRegistry,
    bus,
    approvalGate: gate,
    config,
    systemPrompt,
    repoRoot: projectRoot,
    mode: "act",
    contextManager,
    doubleCheck,
    sessionState,
    initialMessages: options.continuation?.mode === "resume" ? previousSession?.messages : undefined,
  });

  let responseText = "";
  let iterations = 0;
  let success = true;
  let resultSession: ContinuationSession | undefined;
  let outcome: WorkflowQueryResult["outcome"] = "completed";
  let outcomeReason: WorkflowQueryResult["outcomeReason"] | undefined;

  try {
    const result = await taskLoop.run(buildContinuationQuery(options.query, options.continuation));
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
    if (result.status === "budget_exceeded") {
      outcome = "no_progress";
      outcomeReason = "iteration_limit";
      success = false;
    } else if (result.status === "empty_response") {
      outcome = "no_progress";
      outcomeReason = "no_code";
      success = false;
    } else if (result.status === "aborted") {
      outcome = "no_progress";
      outcomeReason = "no_code";
      success = false;
    }
  } catch (err) {
    responseText = extractErrorMessage(err);
    success = false;
    outcome = "no_progress";
  }

  return { success, responseText, iterations, session: resultSession, outcome, outcomeReason };
}
