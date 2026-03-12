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
} from "@devagent/runtime";
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
}

export interface WorkflowQueryResult {
  success: boolean;
  responseText: string;
  iterations: number;
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

  const sessionState = new SessionState(config.sessionState);

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
  });

  let responseText = "";
  let iterations = 0;
  let success = true;

  try {
    const result = await taskLoop.run(options.query);
    responseText = result.lastText ?? "";
    iterations = result.iterations;
    success = !result.aborted;
  } catch (err) {
    responseText = extractErrorMessage(err);
    success = false;
  }

  return { success, responseText, iterations };
}
