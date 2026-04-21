import {
  AgentRegistry,
  AgentType,
  ApprovalGate,
  ContextManager,
  DEFAULT_DOUBLE_CHECK_OPTIONS,
  DoubleCheck,
  EventBus,
  SessionState,
  createDelegateTool,
  createFindingTool,
  createPlanTool,
  createSkillTool,
  createToolScriptTool,
} from "@devagent/runtime";

import { createShellTestRunner } from "./double-check-wiring.js";
import { buildVerbosityConfig, dim } from "./format.js";
import { setupEventHandlers } from "./main-event-handlers.js";
import { getInteractiveSafetyMode } from "./main-safety.js";
import { outputState } from "./main-state.js";
import type { CliArgs, ToolsSetupResult } from "./main-types.js";
import { buildProviderConfig } from "./provider-config.js";
import { createSkillInfrastructure } from "./skill-setup.js";
import { detectProjectTestCommand } from "./test-command-detect.js";
import type { createDefaultRegistry } from "@devagent/providers";
import type { DevAgentConfig, LLMProvider, SessionState as RuntimeSessionState, VerbosityConfig } from "@devagent/runtime";

interface ToolRenderSetup {
  readonly lspToolCounts: Map<string, number>;
  readonly statusLine: import("./status-line.js").StatusLine | null;
  readonly verbosityConfig: VerbosityConfig;
  readonly willUseTui: boolean;
}

interface SkillToolSetup {
  readonly skills: ReturnType<typeof createSkillInfrastructure>["skills"];
  readonly skillResolver: ReturnType<typeof createSkillInfrastructure>["skillResolver"];
  readonly skillAccess: ReturnType<typeof createSkillInfrastructure>["skillAccess"];
  readonly toolRegistry: ReturnType<typeof createSkillInfrastructure>["toolRegistry"];
}

function setupToolRendering(bus: EventBus, config: DevAgentConfig, cliArgs: CliArgs): ToolRenderSetup {
  const verbosityConfig = buildVerbosityConfig(cliArgs.verbosity, cliArgs.verboseCategories);
  const willUseTui = (process.stderr.isTTY ?? false) && cliArgs.verbosity !== "quiet";
  const rendered = willUseTui
    ? { lspToolCounts: new Map<string, number>(), statusLine: null }
    : setupEventHandlers(bus, config, cliArgs.verbosity, verbosityConfig, outputState);
  if (willUseTui) registerTuiOutputTracking(bus);
  return { ...rendered, verbosityConfig, willUseTui };
}

function registerTuiOutputTracking(bus: EventBus): void {
  bus.on("cost:update", (event) => {
    outputState.sessionTotalInputTokens += event.inputTokens;
    outputState.sessionTotalOutputTokens += event.outputTokens;
    outputState.sessionTotalCost += event.totalCost;
    outputState.turnInputTokens += event.inputTokens;
    outputState.turnCostDelta += event.totalCost;
  });
  bus.on("iteration:start", (event) => {
    if (event.agentId) return;
    outputState.currentIteration = event.iteration;
    outputState.currentTokens = event.estimatedTokens;
    outputState.maxContextTokens = event.maxContextTokens;
    outputState.sessionTotalIterations++;
  });
  bus.on("tool:before", (event) => {
    if (event.name.startsWith("audit:")) return;
    outputState.sessionTotalToolCalls++;
    outputState.turnToolCallCount++;
    outputState.hadToolCalls = true;
    outputState.sessionToolUsage.set(event.name, (outputState.sessionToolUsage.get(event.name) ?? 0) + 1);
  });
  bus.on("message:assistant", (event) => {
    if (!event.agentId && event.partial) outputState.textBuffer += event.content;
  });
}

function createLspDiagnosticsTracker(lspToolCounts: Map<string, number>): () => void {
  return () => {
    const key = "diagnostics(double-check)";
    lspToolCounts.set(key, (lspToolCounts.get(key) ?? 0) + 1);
  };
}

function setupSkillTools(
  bus: EventBus,
  projectRoot: string,
  sessionStateRef: { current: RuntimeSessionState },
  cliArgs: CliArgs,
  willUseTui: boolean,
): SkillToolSetup {
  const setup = createSkillInfrastructure(projectRoot, sessionStateRef.current);
  setup.toolRegistry.register(createPlanTool(bus, () => sessionStateRef.current, () => outputState.currentIteration, async () => null));
  let findingToolCallCount = 0;
  bus.on("tool:after", () => { findingToolCallCount++; });
  setup.toolRegistry.register(createFindingTool(() => sessionStateRef.current, () => findingToolCallCount));
  if (setup.skills.size > 0 && cliArgs.verbosity !== "quiet" && !willUseTui) {
    process.stderr.write(dim(`[skills] Discovered ${setup.skills.size} skill(s)`) + "\n");
  }
  setup.toolRegistry.register(createSkillTool(setup.skills, setup.skillResolver, { skillAccess: setup.skillAccess }));
  setup.toolRegistry.register(createToolScriptTool({ registry: setup.toolRegistry, bus }));
  return setup;
}

function registerDelegateTool(options: {
  readonly bus: EventBus;
  readonly cliArgs: CliArgs;
  readonly config: DevAgentConfig;
  readonly gate: ApprovalGate;
  readonly projectRoot: string;
  readonly provider: LLMProvider;
  readonly providerRegistry: ReturnType<typeof createDefaultRegistry>;
  readonly sessionStateRef: { current: RuntimeSessionState };
  readonly skills: SkillToolSetup["skills"];
  readonly toolRegistry: SkillToolSetup["toolRegistry"];
}): { approvalMode: string } {
  const delegateAmbientContext = {
    skills: options.skills,
    approvalMode: getInteractiveSafetyMode(options.config),
    providerLabel: `${options.config.provider} / ${options.config.model}`,
    providerFactory: (agentConfig: DevAgentConfig, agentType: AgentType) => options.providerRegistry.get(
      agentConfig.provider,
      buildProviderConfig(agentConfig, options.cliArgs.reasoning ?? undefined, agentType),
    ),
  };
  options.toolRegistry.register(createDelegateTool({
    provider: options.provider,
    tools: options.toolRegistry,
    bus: options.bus,
    approvalGate: options.gate,
    config: options.config,
    repoRoot: options.projectRoot,
    agentRegistry: new AgentRegistry(),
    parentAgentId: "root",
    getParentSessionState: () => options.sessionStateRef.current,
    depth: 0,
    parentAgentType: AgentType.GENERAL,
    ambient: delegateAmbientContext,
  }));
  return delegateAmbientContext;
}

function setupDoubleCheck(config: DevAgentConfig, bus: EventBus, cliArgs: CliArgs, projectRoot: string, willUseTui: boolean): DoubleCheck {
  const effectiveDoubleCheck = buildEffectiveDoubleCheckOptions(config, projectRoot);
  const doubleCheck = new DoubleCheck(effectiveDoubleCheck, bus);
  if (shouldLogSetup(cliArgs, willUseTui) && effectiveDoubleCheck.enabled) process.stderr.write(dim("[double-check] Validation enabled") + "\n");
  if (effectiveDoubleCheck.testCommand) {
    doubleCheck.setTestRunner(createShellTestRunner(projectRoot));
    logAutoDetectedTestCommand(cliArgs, willUseTui, config, effectiveDoubleCheck.testCommand);
  }
  return doubleCheck;
}

function shouldLogSetup(cliArgs: CliArgs, willUseTui: boolean): boolean {
  return cliArgs.verbosity !== "quiet" && !willUseTui;
}

function buildEffectiveDoubleCheckOptions(config: DevAgentConfig, projectRoot: string): typeof DEFAULT_DOUBLE_CHECK_OPTIONS {
  const enabled = config.doubleCheck?.enabled ?? getInteractiveSafetyMode(config) === "autopilot";
  const autoTestCommand = enabled && !config.doubleCheck?.testCommand ? detectProjectTestCommand(projectRoot) : null;
  return {
    ...DEFAULT_DOUBLE_CHECK_OPTIONS,
    ...config.doubleCheck,
    enabled,
    runTests: config.doubleCheck?.runTests ?? (autoTestCommand !== null),
    testCommand: config.doubleCheck?.testCommand ?? autoTestCommand,
  };
}

function logAutoDetectedTestCommand(cliArgs: CliArgs, willUseTui: boolean, config: DevAgentConfig, testCommand: string): void {
  if (!shouldLogSetup(cliArgs, willUseTui) || config.doubleCheck?.testCommand) return;
  process.stderr.write(dim(`[double-check] Auto-detected test command: ${testCommand}`) + "\n");
}

export function setupTools(
  config: DevAgentConfig,
  cliArgs: CliArgs,
  projectRoot: string,
  provider: LLMProvider,
  providerRegistry: ReturnType<typeof createDefaultRegistry>,
): ToolsSetupResult {
  const bus = new EventBus();
  const gate = new ApprovalGate(config.approval, bus);
  const renderSetup = setupToolRendering(bus, config, cliArgs);
  const sessionStateRef = { current: new SessionState(config.sessionState) };
  const skillSetup = setupSkillTools(bus, projectRoot, sessionStateRef, cliArgs, renderSetup.willUseTui);
  const delegateAmbientContext = registerDelegateTool({
    bus, cliArgs, config, gate, projectRoot, provider, providerRegistry,
    sessionStateRef, skills: skillSetup.skills, toolRegistry: skillSetup.toolRegistry,
  });
  const doubleCheck = setupDoubleCheck(config, bus, cliArgs, projectRoot, renderSetup.willUseTui);
  return {
    toolRegistry: skillSetup.toolRegistry,
    bus,
    gate,
    verbosityConfig: renderSetup.verbosityConfig,
    lspToolCounts: renderSetup.lspToolCounts,
    statusLine: renderSetup.statusLine,
    trackInternalLSPDiagnostics: createLspDiagnosticsTracker(renderSetup.lspToolCounts),
    get sessionState() { return sessionStateRef.current; },
    set sessionState(value: RuntimeSessionState) { sessionStateRef.current = value; },
    skills: skillSetup.skills,
    skillResolver: skillSetup.skillResolver,
    doubleCheck,
    contextManager: new ContextManager(config.context),
    delegateAmbientContext,
  };
}
