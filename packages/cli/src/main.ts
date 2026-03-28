/**
 * CLI main entry point — parses arguments, wires up engine, runs queries.
 * Integrates: skills, session persistence, and task execution.
 */

import { execSync } from "node:child_process";
import { readFileSync as nodeReadFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import {
  EventBus,
  ApprovalGate,
  SkillRegistry,
  SkillResolver,
  ContextManager,
  SessionStore,
  loadConfig,
  resolveProviderCredentials,
  findProjectRoot,
  loadModelRegistry,
  lookupModelEntry,
  DEFAULT_BUDGET,
  DEFAULT_CONTEXT,
  EventLogger,
  loggedSubagentRunFromEvent,
} from "@devagent/runtime";
import type { DevAgentConfig, ApprovalPolicy, LLMProvider, Message, VerbosityConfig } from "@devagent/runtime";
import { AgentType, ApprovalMode, MessageRole , extractErrorMessage } from "@devagent/runtime";
import { createDefaultRegistry, validateOllamaModel } from "@devagent/providers";
import {
  createRoutingLSPTools,
  ToolRegistry,
} from "@devagent/runtime";
import {
  TaskLoop,
  truncateToolOutput,
  createPlanTool,
  createFindingTool,
  createToolScriptTool,
  createDelegateTool,
  createSkillTool,
  AgentRegistry,
  DoubleCheck,
  DEFAULT_DOUBLE_CHECK_OPTIONS,
  synthesizeBriefing,
  findLastUserContent,
  SessionState,
} from "@devagent/runtime";
import type { TaskMode, TaskLoopResult, TurnBriefing, MidpointCallback, SessionStatePersistence, SessionStateJSON } from "@devagent/runtime";
import { LSPRouter, createRoutingDiagnosticProvider, createCompilerFallbackProvider, createShellTestRunner, lazyUpgradeLSP } from "./double-check-wiring.js";
import { createArkTSDiagnosticProvider } from "@devagent/arkts";
import { assembleSystemPrompt } from "./prompts/index.js";
import { detectProjectTestCommand } from "./test-command-detect.js";
import {
  Spinner,
  dim, red, cyan, green, yellow, bold,
  formatToolStart,
  formatToolEnd,
  formatToolGroupStart,
  formatToolGroupEnd,
  formatSubagentBatchLaunch,
  formatSubagentStart,
  formatSubagentError,
  summarizeSubagentUpdate,
  SubagentPanelRenderer,
  summarizeToolParams,
  formatPlan,
  formatError,
  isCategoryEnabled,
  buildVerbosityConfig,
  formatContextGauge,
  formatEnrichedError,
  inferErrorSuggestion,
  formatTurnSummary,
  formatSessionSummary,
  formatCompactionResult,
  formatReasoning,
} from "./format.js";
import { resolveBundledModelsDir } from "./model-registry-path.js";
import { OutputState } from "./output-state.js";
import type { SubagentDisplayState } from "./output-state.js";
import { buildProviderConfig } from "./provider-config.js";
import { createSkillInfrastructure } from "./skill-setup.js";

// ─── Argument Parsing ────────────────────────────────────────

type Verbosity = "quiet" | "normal" | "verbose";

// ─── Run Options ────────────────────────────────────────────

interface RunOptions {
  readonly provider: LLMProvider;
  readonly toolRegistry: ToolRegistry;
  readonly bus: EventBus;
  readonly gate: ApprovalGate;
  readonly config: DevAgentConfig;
  readonly repoRoot: string;
  readonly mode: TaskMode;
  readonly skills: SkillRegistry;
  readonly contextManager: ContextManager;
  readonly doubleCheck: DoubleCheck;
  readonly initialMessages: Message[] | undefined;
  readonly verbosity: Verbosity;
  readonly verbosityConfig: VerbosityConfig;
  readonly sessionState: SessionState;
  readonly briefing?: TurnBriefing;
}

interface RunSingleQueryOptions extends RunOptions {
  readonly query: string;
}

interface ReviewArgs {
  patchFile: string;
  ruleFile: string | null;
  jsonOutput: boolean;
}

interface CliArgs {
  query: string | null;
  file: string | null;
  provider: string | null;
  model: string | null;
  maxIterations: number | null;
  reasoning: "low" | "medium" | "high" | null;
  verbosity: Verbosity;
  verboseCategories: string | undefined;
  authCommand: string | null;
  resume: string | null;
  continue_: boolean;
  review: ReviewArgs | null;
}

export function loadQueryFromFile(
  path: string,
  readFileSync: (path: string, encoding: "utf-8") => string = nodeReadFileSync,
  inlineQuery: string | null = null,
): string {
  if (inlineQuery) {
    throw new Error("Cannot specify both --file and an inline query");
  }

  let raw: string;
  try {
    raw = readFileSync(path, "utf-8");
  } catch (error) {
    const message = extractErrorMessage(error);
    if (message.includes("ENOENT")) {
      throw new Error(`Input file not found: ${path}`);
    }
    throw error;
  }

  const query = raw.trim();
  if (query.length === 0) {
    throw new Error(`Input file is empty: ${path}`);
  }

  return query;
}

interface CrashSessionReporter {
  printSessionId: () => void;
  dispose: () => void;
}

interface CrashSessionReporterProcess {
  stderr: { write: (chunk: string) => boolean };
  once: (event: "SIGINT" | "uncaughtException" | "unhandledRejection", listener: (...args: any[]) => void) => void;
  off: (event: "SIGINT" | "uncaughtException" | "unhandledRejection", listener: (...args: any[]) => void) => void;
  exit: (code?: number) => never;
}

export function createCrashSessionReporter(
  sessionId: string,
  verbosity: Verbosity,
  proc: CrashSessionReporterProcess = process,
): CrashSessionReporter {
  let printed = false;

  const printSessionId = (): void => {
    if (printed || verbosity === "quiet") return;
    proc.stderr.write(dim(`[session] ${sessionId}`) + "\n");
    printed = true;
  };

  const onSigint = (): void => {
    printSessionId();
    proc.exit(130);
  };

  const onUncaughtException = (err: unknown): void => {
    proc.stderr.write(formatError(`Uncaught exception: ${extractErrorMessage(err)}`) + "\n");
    printSessionId();
    proc.exit(1);
  };

  const onUnhandledRejection = (reason: unknown): void => {
    proc.stderr.write(formatError(`Unhandled rejection: ${extractErrorMessage(reason)}`) + "\n");
    printSessionId();
    proc.exit(1);
  };

  proc.once("SIGINT", onSigint);
  proc.once("uncaughtException", onUncaughtException);
  proc.once("unhandledRejection", onUnhandledRejection);

  return {
    printSessionId,
    dispose: (): void => {
      proc.off("SIGINT", onSigint);
      proc.off("uncaughtException", onUncaughtException);
      proc.off("unhandledRejection", onUnhandledRejection);
    },
  };
}

export function parseArgs(argv: string[]): CliArgs {
  const args = argv.slice(2); // Skip bun and script path
  const result: CliArgs = {
    query: null,
    file: null,
    provider: null,
    model: null,
    maxIterations: null,
    reasoning: null,
    verbosity: "normal",
    verboseCategories: undefined,
    authCommand: null,
    resume: null,
    continue_: false,
    review: null,
  };

  for (let i = 0; i < args.length; i++) {
    const arg = args[i]!;

    if (arg === "review") {
      // devagent review <file> [--rule <rule_file>] [--json] [--provider <p>] [--model <m>]
      const reviewArgs: ReviewArgs = { patchFile: "", ruleFile: null, jsonOutput: false };
      i++;
      while (i < args.length) {
        const rarg = args[i]!;
        if (rarg === "--rule" && i + 1 < args.length) {
          reviewArgs.ruleFile = args[++i]!;
        } else if (rarg === "--json") {
          reviewArgs.jsonOutput = true;
        } else if (rarg === "--provider" && i + 1 < args.length) {
          result.provider = args[++i]!;
        } else if (rarg === "--model" && i + 1 < args.length) {
          result.model = args[++i]!;
        } else if (!rarg.startsWith("-")) {
          reviewArgs.patchFile = rarg;
        }
        i++;
      }
      result.review = reviewArgs;
      return result;
    } else if (arg === "auth") {
      result.authCommand = args[i + 1] ?? "login";
      i++;
      return result; // auth is handled before anything else
    } else if (arg === "--suggest") {
      // handled in config override
    } else if (arg === "--auto-edit") {
      // handled in config override
    } else if (arg === "--full-auto") {
      // handled in config override
    } else if (arg === "-q" || arg === "--quiet") {
      result.verbosity = "quiet";
    } else if (arg === "-v" || arg === "--verbose") {
      result.verbosity = "verbose";
    } else if (arg.startsWith("--verbose=")) {
      const cats = arg.slice("--verbose=".length);
      result.verboseCategories = cats;
    } else if ((arg === "--file" || arg === "-f") && i + 1 < args.length) {
      result.file = args[++i]!;
    } else if (arg === "--provider" && i + 1 < args.length) {
      result.provider = args[++i]!;
    } else if (arg === "--model" && i + 1 < args.length) {
      result.model = args[++i]!;
    } else if (arg === "--max-iterations" && i + 1 < args.length) {
      const val = parseInt(args[++i]!, 10);
      result.maxIterations = isNaN(val) ? null : val;
    } else if (arg === "--reasoning" && i + 1 < args.length) {
      const val = args[++i]! as "low" | "medium" | "high";
      if (["low", "medium", "high"].includes(val)) {
        result.reasoning = val;
      }
    } else if (arg === "--resume" && i + 1 < args.length) {
      result.resume = args[++i]!;
    } else if (arg === "--continue") {
      result.continue_ = true;
    } else if (arg === "--help" || arg === "-h") {
      printHelp();
      process.exit(0);
    } else if (!arg.startsWith("-")) {
      result.query = arg;
    }
  }

  return result;
}

function getApprovalMode(argv: string[]): ApprovalMode | null {
  if (argv.includes("--suggest")) return ApprovalMode.SUGGEST;
  if (argv.includes("--auto-edit")) return ApprovalMode.AUTO_EDIT;
  if (argv.includes("--full-auto")) return ApprovalMode.FULL_AUTO;
  return null;
}

export function renderHelpText(): string {
  return `
devagent — AI-powered development agent

Usage:
  devagent "<query>"              Natural language query
  devagent -f <path>               Read query from file
  devagent review <file> --rule <rule_file> [--json]
                                  Rule-based patch review

Auth:
  devagent auth login             Store API key for a provider
  devagent auth status            Show configured credentials
  devagent auth logout            Remove stored credentials

Options:
  -f, --file <path>    Read query from file
  --provider <name>     LLM provider (anthropic, openai, deepseek, openrouter, ollama, chatgpt, github-copilot)
  --model <id>          Model ID
  --max-iterations <n>  Max tool-call iterations (default: 30)
  --reasoning <level>   Reasoning effort: low, medium, high
  --resume <id>         Resume a previous session by ID
  --continue            Resume the most recent session
  --suggest             Suggest mode (show diffs, ask before writing)
  --auto-edit           Auto-edit mode (auto-approve file writes)
  --full-auto           Full-auto mode (auto-approve everything)
  -v, --verbose         Verbose output (show full tool params and results)
  -q, --quiet           Quiet output (errors only)
  -h, --help            Show this help

Environment:
  DEVAGENT_PROVIDER     Default provider
  DEVAGENT_MODEL        Default model
  DEVAGENT_API_KEY      API key for the default provider

Installation:
  bun run build && bun run install-cli    Install as 'devagent' command
`;
}

function printHelp(): void {
  console.log(renderHelpText());
}



// ─── Setup Helpers ─────────────────────────────────────────

/** Result of config setup: resolved config and project root. */
interface ConfigSetupResult {
  readonly config: DevAgentConfig;
  readonly projectRoot: string;
}

/**
 * Parse CLI overrides, load config from disk, resolve OAuth credentials,
 * load the model registry, and auto-size the context budget.
 */
async function setupConfig(cliArgs: CliArgs): Promise<ConfigSetupResult> {
  const projectRoot = findProjectRoot() ?? process.cwd();

  // Load config with CLI overrides
  const approvalMode = getApprovalMode(process.argv);
  const configOverrides: Partial<DevAgentConfig> = {
    ...(cliArgs.provider ? { provider: cliArgs.provider } : {}),
    ...(cliArgs.model ? { model: cliArgs.model } : {}),
    ...(approvalMode ? { approval: { mode: approvalMode } as ApprovalPolicy } : {}),
  };

  let config = loadConfig(projectRoot, configOverrides);

  // Resolve OAuth credentials (refresh expired tokens) — must come before provider setup
  try {
    config = await resolveProviderCredentials(config);
  } catch (err) {
    const msg = extractErrorMessage(err);
    process.stderr.write(formatError(`OAuth credential error: ${msg}`) + "\n");
    process.stderr.write(dim('Run "devagent auth login" to re-authenticate.') + "\n");
    process.exit(1);
  }

  // Apply CLI overrides that loadConfig doesn't handle
  if (cliArgs.maxIterations !== null) {
    config = {
      ...config,
      budget: { ...config.budget, maxIterations: cliArgs.maxIterations },
    };
  }

  // Load model registry (models/*.toml files with per-model capabilities)
  // Search: devagent repo models/ dir, project models/ dir, ~/.config/devagent/models/
  const cliDir = dirname(fileURLToPath(import.meta.url));
  const devagentModelsDir = resolveBundledModelsDir(cliDir);
  loadModelRegistry(projectRoot, [devagentModelsDir]);

  // Auto-size context budget from model registry when user hasn't overridden it.
  // Without this, a model with 192K context gets the default 100K budget.
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

  // Scale keepRecentMessages with context budget when the user hasn't
  // explicitly configured it. A fixed value (40) is too aggressive on
  // large-context models (192K+) and causes unnecessary Phase 2 compaction.
  if (config.context.keepRecentMessages === DEFAULT_CONTEXT.keepRecentMessages) {
    const effectiveBudget = config.budget.maxContextTokens - config.budget.responseHeadroom;
    const scaledKeep = Math.floor(effectiveBudget / 1500);
    if (scaledKeep > DEFAULT_CONTEXT.keepRecentMessages) {
      config = {
        ...config,
        context: {
          ...config.context,
          keepRecentMessages: scaledKeep,
        },
      };
    }
  }

  return { config, projectRoot };
}

/** Result of provider setup: the LLM provider instance. */
interface ProviderSetupResult {
  readonly provider: LLMProvider;
  readonly providerRegistry: ReturnType<typeof createDefaultRegistry>;
}

/**
 * Create the LLM provider from config, resolving capabilities from the model
 * registry and validating API keys.
 */
function setupProvider(config: DevAgentConfig, cliArgs: CliArgs): ProviderSetupResult {
  const providerRegistry = createDefaultRegistry();
  const providerConfig = buildProviderConfig(config, cliArgs.reasoning ?? undefined);

  // Providers that don't require an API key (local endpoints or OAuth-based)
  const noKeyProviders = new Set(["ollama", "chatgpt", "github-copilot"]);
  if (!providerConfig.apiKey && !providerConfig.oauthToken && !noKeyProviders.has(config.provider)) {
    process.stderr.write(
      formatError(`No API key configured for provider "${config.provider}".`) + "\n",
    );
    process.stderr.write(
      dim('Run "devagent auth login" to store a key, or set DEVAGENT_API_KEY') + "\n",
    );
    process.exit(1);
  }

  const provider = providerRegistry.get(config.provider, providerConfig);

  return { provider, providerRegistry };
}

/** Result of tools setup: all registries, bus, gate, and supporting objects. */
interface ToolsSetupResult {
  readonly toolRegistry: ToolRegistry;
  readonly bus: EventBus;
  readonly gate: ApprovalGate;
  readonly verbosityConfig: VerbosityConfig;
  readonly lspToolCounts: Map<string, number>;
  readonly trackInternalLSPDiagnostics: () => void;
  /** Mutable — may be swapped on resume. Access via getter closure. */
  sessionState: SessionState;
  readonly skills: SkillRegistry;
  readonly skillResolver: SkillResolver;
  readonly doubleCheck: DoubleCheck;
  readonly contextManager: ContextManager;
}

/**
 * Wire up the tool registry, event bus, approval gate, session state, skills,
 * delegate tool, double-check, and context manager.
 */
async function setupTools(
  config: DevAgentConfig,
  cliArgs: CliArgs,
  projectRoot: string,
  provider: LLMProvider,
  providerRegistry: ReturnType<typeof createDefaultRegistry>,
): Promise<ToolsSetupResult> {
  const bus = new EventBus();
  const gate = new ApprovalGate(config.approval, bus);
  const verbosityConfig = buildVerbosityConfig(cliArgs.verbosity, cliArgs.verboseCategories);
  const { lspToolCounts } = setupEventHandlers(bus, config, cliArgs.verbosity, verbosityConfig);
  const trackInternalLSPDiagnostics = () => {
    const key = "diagnostics(double-check)";
    lspToolCounts.set(key, (lspToolCounts.get(key) ?? 0) + 1);
  };

  // Session state sidecar — structured facts that survive compaction and turn boundaries.
  // Created early so it can be passed to createPlanTool. Bound to disk persistence
  // later (after session record is created) via sessionState.bind().
  let sessionState = new SessionState(config.sessionState);

  // ─── Skills (Agent Skills standard) ────────────────────────
  const { skills, skillResolver, skillAccess, toolRegistry } = createSkillInfrastructure(
    projectRoot,
    sessionState,
  );

  // Register state tools (getter indirection so resume can swap the instance)
  toolRegistry.register(createPlanTool(
    bus,
    () => sessionState,
    () => outputState.currentIteration,
    async () => null,
  ));
  // Finding tool: tracks iteration via tool:after event count
  let findingToolCallCount = 0;
  bus.on("tool:after", () => { findingToolCallCount++; });
  toolRegistry.register(createFindingTool(() => sessionState, () => findingToolCallCount));
  if (skills.size > 0 && cliArgs.verbosity !== "quiet") {
    process.stderr.write(dim(`[skills] Discovered ${skills.size} skill(s)`) + "\n");
  }

  // Register skill tool (LLM can invoke skills during conversation)
  toolRegistry.register(createSkillTool(skills, skillResolver, { skillAccess }));

  // ─── Batched Readonly Tool Scripts ──────────────────────────
  // Register after all readonly tools so the script engine can access them.
  toolRegistry.register(createToolScriptTool({ registry: toolRegistry, bus }));

  // ─── Delegate (subagent spawning) ─────────────────────────────
  const agentRegistry = new AgentRegistry();
  toolRegistry.register(createDelegateTool({
    provider,
    tools: toolRegistry,
    bus,
    approvalGate: gate,
    config,
    repoRoot: projectRoot,
    agentRegistry,
    parentAgentId: "root",
    getParentSessionState: () => sessionState,
    depth: 0,
    parentAgentType: AgentType.GENERAL,
    ambient: {
      skills,
      approvalMode: config.approval.mode,
      providerLabel: `${config.provider} / ${config.model}`,
      providerFactory: (agentConfig, agentType) => {
        return providerRegistry.get(
          agentConfig.provider,
          buildProviderConfig(agentConfig, cliArgs.reasoning ?? undefined, agentType),
        );
      },
    },
  }));

  // ─── Double-Check ──────────────────────────────────────────
  // Auto-enable DoubleCheck in full-auto mode unless explicitly disabled.
  // When enabled with no explicit test command, auto-detect from package.json
  // and enable test running so the LLM can self-correct from test failures.
  const isFullAuto = config.approval.mode === ApprovalMode.FULL_AUTO;
  const dcEnabled = config.doubleCheck?.enabled ?? isFullAuto;
  const autoTestCommand = dcEnabled && !config.doubleCheck?.testCommand
    ? detectProjectTestCommand(projectRoot)
    : null;

  const effectiveDoubleCheck = {
    ...DEFAULT_DOUBLE_CHECK_OPTIONS,
    ...config.doubleCheck,
    enabled: dcEnabled,
    // Auto-enable test running when a test command is detected
    runTests: config.doubleCheck?.runTests ?? (autoTestCommand !== null),
    testCommand: config.doubleCheck?.testCommand ?? autoTestCommand,
  };

  const doubleCheck = new DoubleCheck(effectiveDoubleCheck, bus);

  if (effectiveDoubleCheck.enabled && cliArgs.verbosity !== "quiet") {
    process.stderr.write(dim("[double-check] Validation enabled") + "\n");
  }

  // Wire test runner (works without LSP — just needs a shell)
  if (effectiveDoubleCheck.testCommand) {
    doubleCheck.setTestRunner(createShellTestRunner(projectRoot));
    if (autoTestCommand && cliArgs.verbosity !== "quiet") {
      process.stderr.write(dim(`[double-check] Auto-detected test command: ${autoTestCommand}`) + "\n");
    }
  }

  // ─── Context Management ────────────────────────────────────
  const contextManager = new ContextManager(config.context);

  return {
    toolRegistry,
    bus,
    gate,
    verbosityConfig,
    lspToolCounts,
    trackInternalLSPDiagnostics,
    sessionState,
    skills,
    skillResolver,
    doubleCheck,
    contextManager,
  };
}

/** Result of LSP setup: the router (for shutdown) and whether diagnostics are available. */
interface LSPSetupResult {
  lspRouter: LSPRouter | null;
  hasLSPDiagnostics: boolean;
}

/**
 * Start LSP servers, register routing tools, wire diagnostic providers
 * (including ArkTS and compiler fallback), and schedule lazy LSP upgrade.
 */
async function setupLSP(
  config: DevAgentConfig,
  cliArgs: CliArgs,
  projectRoot: string,
  toolRegistry: ToolRegistry,
  doubleCheck: DoubleCheck,
  trackInternalLSPDiagnostics: () => void,
): Promise<LSPSetupResult> {
  let lspRouter: LSPRouter | null = null;
  let hasLSPDiagnostics = false;

  if (config.lsp?.servers && config.lsp.servers.length > 0) {
    lspRouter = new LSPRouter(projectRoot);

    const lspStartPromises = config.lsp.servers.map(async (serverConfig) => {
      try {
        await lspRouter!.addServer(serverConfig);
        if (cliArgs.verbosity !== "quiet") {
          process.stderr.write(
            dim(`[lsp] Started: ${serverConfig.command} (${serverConfig.languages.join(", ")})`) + "\n",
          );
        }
      } catch (err) {
        const msg = extractErrorMessage(err);
        process.stderr.write(
          formatError(`LSP start failed for ${serverConfig.command}: ${msg}. Skipping.`) + "\n",
        );
      }
    });
    await Promise.allSettled(lspStartPromises);

    // Register routing LSP tools (routes to correct server by file extension)
    const clients = lspRouter.getClients();
    if (clients.length > 0) {
      hasLSPDiagnostics = true;
      const router = lspRouter;
      const resolver = (filePath: string) => router.getClientForFile(filePath);
      for (const tool of createRoutingLSPTools(resolver)) {
        toolRegistry.register(tool);
      }
    }

    // Wire routing diagnostic provider (routes by file extension)
    let diagnosticProvider = createRoutingDiagnosticProvider(
      lspRouter,
      trackInternalLSPDiagnostics,
    );

    // Compose with ArkTS linter if enabled (adds tslinter checks for .ets files)
    if (config.arkts?.enabled && config.arkts.linterPath) {
      const arktsProvider = createArkTSDiagnosticProvider(config.arkts);
      if (arktsProvider) {
        const lspProvider = diagnosticProvider;
        diagnosticProvider = async (filePath: string) => {
          const [lsp, arkts] = await Promise.all([
            lspProvider(filePath),
            arktsProvider(filePath),
          ]);
          return [...lsp, ...arkts];
        };
        if (cliArgs.verbosity !== "quiet") {
          process.stderr.write(
            dim(`[arkts] ArkTS linter enabled (${config.arkts.linterPath})`) + "\n",
          );
        }
      }
    }

    doubleCheck.setDiagnosticProvider(diagnosticProvider);
  } else if (config.arkts?.enabled && config.arkts.linterPath) {
    // Standalone ArkTS linting (no LSP servers configured)
    const arktsProvider = createArkTSDiagnosticProvider(config.arkts);
    if (arktsProvider) {
      doubleCheck.setDiagnosticProvider(arktsProvider);
      hasLSPDiagnostics = true;
      if (cliArgs.verbosity !== "quiet") {
        process.stderr.write(
          dim(`[arkts] ArkTS linter enabled (standalone, ${config.arkts.linterPath})`) + "\n",
        );
      }
    }
  }

  // Wire compiler fallback when no LSP/linter diagnostics and DoubleCheck is enabled.
  // Then schedule lazy LSP upgrade in the background — when LSP servers are found
  // in PATH, they'll be started and the provider swapped transparently.
  const effectiveEnabled = doubleCheck.isEnabled();
  if (!hasLSPDiagnostics && effectiveEnabled) {
    doubleCheck.setDiagnosticProvider(createCompilerFallbackProvider(projectRoot));

    // Prepare ArkTS provider for composition with lazy LSP (if enabled)
    const arktsProviderForLazy = (config.arkts?.enabled && config.arkts.linterPath)
      ? createArkTSDiagnosticProvider(config.arkts) ?? undefined
      : undefined;

    // Lazy LSP: create a router, detect servers in background, upgrade when ready
    const lazyRouter = new LSPRouter(projectRoot);
    lspRouter = lazyRouter; // So shutdown can clean up

    // Fire-and-forget — runs in background while the session starts
    lazyUpgradeLSP({
      repoRoot: projectRoot,
      doubleCheck,
      lspRouter: lazyRouter,
      arktsProvider: arktsProviderForLazy,
      onLSPDiagnostics: trackInternalLSPDiagnostics,
      onServerStarted: (server) => {
        if (cliArgs.verbosity !== "quiet") {
          spinner.log(
            dim(`[lsp] Auto-detected: ${server.command} (${server.languages.join(", ")})`),
          );
        }
      },
      onUpgradeComplete: (count) => {
        if (count > 0) {
          hasLSPDiagnostics = true;
          // Register routing LSP tools (routes to correct server by file extension)
          const clients = lazyRouter.getClients();
          if (clients.length > 0) {
            const resolver = (filePath: string) => lazyRouter.getClientForFile(filePath);
            for (const tool of createRoutingLSPTools(resolver)) {
              toolRegistry.register(tool);
            }
          }
          if (cliArgs.verbosity !== "quiet") {
            spinner.log(
              dim(`[lsp] Upgraded to LSP diagnostics (${count} server(s))`),
            );
          }
        } else if (cliArgs.verbosity !== "quiet") {
          spinner.log(dim("[double-check] Using compiler fallback diagnostics (no LSP servers in PATH)"));
        }
      },
      onError: (err) => {
        if (cliArgs.verbosity !== "quiet") {
          spinner.log(dim(`[lsp] Lazy detection failed: ${err.message}`));
        }
      },
    }).catch(() => {
      // Silently absorb — compiler fallback remains active
    });
  }

  return { lspRouter, hasLSPDiagnostics };
}

/** Result of session persistence setup. */
interface SessionPersistenceResult {
  readonly sessionStore: SessionStore;
  readonly session: import("@devagent/runtime").Session;
  readonly initialMessages: Message[] | undefined;
  readonly resumeBriefing: TurnBriefing | undefined;
  readonly eventLogger: EventLogger | null;
  /** Possibly updated sessionState (swapped on resume). */
  sessionState: SessionState;
}

/**
 * Create or resume a session: load prior messages/briefing, create session
 * record, bind session state to disk, set up event logger and bus-based
 * message persistence.
 */
async function setupSessionPersistence(
  config: DevAgentConfig,
  cliArgs: CliArgs,
  provider: LLMProvider,
  bus: EventBus,
  sessionState: SessionState,
): Promise<SessionPersistenceResult> {
  const sessionStore = new SessionStore();

  // Resume previous session if requested
  // Turn isolation: synthesize briefing from prior session instead of loading raw messages.
  // This prevents accumulated history from degrading LLM accuracy (Manager-Worker pattern).
  let initialMessages: Message[] | undefined;
  let resumeBriefing: TurnBriefing | undefined;
  let prevSession: import("@devagent/runtime").Session | null = null;
  if (cliArgs.resume || cliArgs.continue_) {
    prevSession = cliArgs.resume
      ? sessionStore.getSession(cliArgs.resume)
      : sessionStore.listSessions(1)[0] ?? null;

    if (prevSession) {
      const useTurnIsolation = config.context.turnIsolation !== false;
      if (useTurnIsolation) {
        // Synthesize briefing from the prior session instead of replaying raw history.
        // This prevents accumulated history from degrading LLM accuracy.
        try {
          resumeBriefing = await synthesizeBriefing(
            prevSession.messages, 1,
            { strategy: config.context.briefingStrategy ?? "auto", provider },
          );
          if (cliArgs.verbosity !== "quiet") {
            process.stderr.write(
              dim(`[session] Resuming ${prevSession.id} via briefing (${prevSession.messages.length} messages → synthesized)`) + "\n",
            );
          }
        } catch {
          // Fallback to raw messages if briefing synthesis fails
          initialMessages = [...prevSession.messages];
          if (cliArgs.verbosity !== "quiet") {
            process.stderr.write(
              dim(`[session] Resuming ${prevSession.id} (${prevSession.messages.length} messages, briefing failed)`) + "\n",
            );
          }
        }
      } else {
        // Turn isolation disabled — raw message resume
        initialMessages = [...prevSession.messages];
        if (cliArgs.verbosity !== "quiet") {
          process.stderr.write(
            dim(`[session] Resuming ${prevSession.id} (${prevSession.messages.length} messages)`) + "\n",
          );
        }
      }
    } else {
      const searchId = cliArgs.resume ?? "most recent";
      process.stderr.write(
        yellow(`[session] No session found: ${searchId}`) + "\n",
      );
    }
  }

  // Create session record for this run
  const session = sessionStore.createSession({
    query: cliArgs.query ?? "(file query)",
    provider: config.provider,
    model: config.model,
    mode: "act",
  });

  // ─── Disk-backed SessionState ────────────────────────────────
  // Adapter: bridge SessionStore's object-typed methods with
  // the typed SessionStatePersistence interface from @devagent/runtime.
  const sessionStatePersistence: SessionStatePersistence = {
    save: (id: string, state: SessionStateJSON) =>
      sessionStore.saveSessionState(id, state),
    load: (id: string) => {
      const raw = sessionStore.loadSessionState(id);
      return raw as SessionStateJSON | null;
    },
  };

  // On resume: load accumulated state from the prior session
  let effectiveSessionState = sessionState;
  if ((cliArgs.resume || cliArgs.continue_) && prevSession) {
    const prevData = sessionStatePersistence.load(prevSession.id);
    if (prevData) {
      effectiveSessionState = SessionState.fromJSON(prevData, config.sessionState);
      if (cliArgs.verbosity !== "quiet") {
        const planLen = prevData.plan?.length ?? 0;
        const fileLen = prevData.modifiedFiles?.length ?? 0;
        process.stderr.write(
          dim(`[session-state] Restored from prior session (${planLen} plan steps, ${fileLen} files)`) + "\n",
        );
      }
    }
  }
  // Bind to current session for ongoing auto-save
  effectiveSessionState.bind(session.id, sessionStatePersistence);

  // ─── Event Logger (JSONL persistence) ─────────────────────
  let eventLogger: EventLogger | null = null;
  const loggingEnabled = config.logging?.enabled !== false;
  if (loggingEnabled) {
    // Rotate old logs (non-fatal — log cleanup should never block the user)
    try {
      const retentionDays = config.logging?.retentionDays ?? 30;
      const deleted = EventLogger.rotate(retentionDays, config.logging?.logDir);
      if (deleted > 0 && cliArgs.verbosity === "verbose") {
        process.stderr.write(dim(`[logging] Rotated ${deleted} old log file(s)`) + "\n");
      }
    } catch {
      // Non-fatal: documented exception to fail-fast
    }

    eventLogger = new EventLogger(session.id, config.logging?.logDir);
    eventLogger.attach(bus);
  }

  // Persist messages incrementally via bus events
  bus.on("message:user", (event) => {
    if (event.agentId) return;
    sessionStore.addMessage(session.id, {
      role: MessageRole.USER,
      content: event.content,
    });
  });
  bus.on("message:assistant", (event) => {
    if (event.agentId) return;
    if (!event.partial) {
      sessionStore.addMessage(session.id, {
        role: MessageRole.ASSISTANT,
        content: event.content,
        toolCalls: event.toolCalls,
      });
    }
  });
  bus.on("message:tool", (event) => {
    if (event.agentId) return;
    sessionStore.addMessage(session.id, {
      role: MessageRole.TOOL,
      content: event.content,
      toolCallId: event.toolCallId,
    });
  });
  bus.on("cost:update", (event) => {
    sessionStore.addCostRecord(session.id, {
      inputTokens: event.inputTokens,
      outputTokens: event.outputTokens,
      cacheReadTokens: 0,
      cacheWriteTokens: 0,
      totalCost: event.totalCost,
    });
  });

  // Persist compaction events for forensic analysis
  bus.on("context:compacted", (event) => {
    sessionStore.saveCompactionEvent(session.id, {
      tokensBefore: event.tokensBefore,
      tokensAfter: event.estimatedTokens,
      removedCount: event.removedCount,
    });
  });

  return {
    sessionStore,
    session,
    initialMessages,
    resumeBriefing,
    eventLogger,
    sessionState: effectiveSessionState,
  };
}

// ─── Main ──────────────────────────────────────────────────

export async function main(): Promise<void> {
  if (process.argv[2] === "execute") {
    const {
      executeTask,
      loadTaskExecutionRequest,
      parseExecuteArgs,
    } = await import("@devagent/executor");
    const executeArgs = parseExecuteArgs(process.argv);
    if (!executeArgs) {
      process.exit(1);
    }
    const request = await loadTaskExecutionRequest(executeArgs.requestPath);
    const emit = (event: Record<string, unknown>): void => {
      process.stdout.write(JSON.stringify(event) + "\n");
    };

    try {
      const { setupAndRunWorkflowQuery } = await import("./workflow-engine.js");
      const result = await executeTask({
        request,
        artifactDir: executeArgs.artifactDir,
        repoRoot: process.cwd(),
        runQuery: setupAndRunWorkflowQuery,
        emit: (event) => {
          emit(event);
        },
      });
      if (result.status !== "success") {
        process.exit(1);
      }
    } catch (error) {
      process.exit(1);
    }
    return;
  }

  const cliArgs = parseArgs(process.argv);

  // Auth commands — handle before config loading (doesn't need provider setup)
  if (cliArgs.authCommand) {
    const { runAuthCommand } = await import("./auth.js");
    await runAuthCommand(cliArgs.authCommand);
    return;
  }

  // ─── 1. Config ─────────────────────────────────────────────
  const { config, projectRoot } = await setupConfig(cliArgs);

  // ─── 2. Provider ───────────────────────────────────────────
  const { provider, providerRegistry } = setupProvider(config, cliArgs);

  // ─── Review Command (needs provider but not full tool setup) ──
  if (cliArgs.review) {
    const { runReviewPipeline } = await import("@devagent/runtime");
    const reviewArgs = cliArgs.review;

    if (!reviewArgs.patchFile) {
      process.stderr.write(formatError("Usage: devagent review <file> [--rule <rule_file>] [--json]") + "\n");
      process.exit(1);
    }

    if (!reviewArgs.ruleFile) {
      process.stderr.write(formatError("Rule file required: devagent review <file> --rule <rule_file>") + "\n");
      process.exit(1);
    }

    try {
      const result = await runReviewPipeline(
        { provider, workspaceRoot: projectRoot },
        { patchFile: reviewArgs.patchFile, ruleFile: reviewArgs.ruleFile },
      );

      if (reviewArgs.jsonOutput) {
        process.stdout.write(JSON.stringify(result, null, 2) + "\n");
      } else {
        // Human-readable output
        const { violations, summary } = result;
        if (violations.length === 0) {
          console.log(green("No violations found."));
        } else {
          console.log(bold(`Found ${violations.length} violation(s) in ${summary.filesReviewed} file(s):\n`));
          for (const v of violations) {
            const sevColor = v.severity === "error" ? red : v.severity === "warning" ? yellow : dim;
            console.log(`  ${sevColor(v.severity.toUpperCase().padEnd(7))} ${dim(v.file)}:${v.line}`);
            console.log(`           ${v.message}`);
            if (v.codeSnippet) {
              console.log(`           ${dim(v.codeSnippet)}`);
            }
            console.log();
          }
        }
        console.log(dim(`Rule: ${summary.ruleName} | Files: ${summary.filesReviewed} | Violations: ${summary.totalViolations}`));
      }
    } catch (err) {
      const msg = extractErrorMessage(err);
      process.stderr.write(formatError(`Review failed: ${msg}`) + "\n");
      process.exit(1);
    }

    return;
  }

  // Pre-flight: validate Ollama model availability before session setup
  if (config.provider === "ollama") {
    try {
      const ollamaConfig = config.providers[config.provider];
      const ollamaBaseUrl = ollamaConfig?.baseUrl ?? "http://localhost:11434/v1";
      await validateOllamaModel(config.model, ollamaBaseUrl);
    } catch (err) {
      const msg = extractErrorMessage(err);
      process.stderr.write(formatError(msg) + "\n");
      process.exit(1);
    }
  }

  // ─── 3. Tools, bus, gate, skills, delegate, double-check ──
  const tools = await setupTools(config, cliArgs, projectRoot, provider, providerRegistry);

  // ─── 4. LSP ────────────────────────────────────────────────
  const lsp = await setupLSP(
    config, cliArgs, projectRoot,
    tools.toolRegistry, tools.doubleCheck, tools.trackInternalLSPDiagnostics,
  );

  // ─── 5. Session persistence ────────────────────────────────
  const persistence = await setupSessionPersistence(
    config, cliArgs, provider, tools.bus, tools.sessionState,
  );
  // If session state was swapped on resume, propagate back so closures see it
  tools.sessionState = persistence.sessionState;

  const crashSessionReporter = createCrashSessionReporter(
    persistence.session.id,
    cliArgs.verbosity,
  );

  const sessionStartTime = Date.now();
  try {
    const commonOptions = {
      provider,
      toolRegistry: tools.toolRegistry,
      bus: tools.bus,
      gate: tools.gate,
      config,
      repoRoot: projectRoot,
      mode: "act" as TaskMode,
      skills: tools.skills,
      contextManager: tools.contextManager,
      doubleCheck: tools.doubleCheck,
      initialMessages: persistence.initialMessages,
      verbosity: cliArgs.verbosity,
      verbosityConfig: tools.verbosityConfig,
      sessionState: tools.sessionState,
      briefing: persistence.resumeBriefing,
    };

    const query = cliArgs.file
      ? loadQueryFromFile(cliArgs.file, nodeReadFileSync, cliArgs.query)
      : cliArgs.query;
    if (!query) {
      process.stderr.write(formatError("Query required. Use `devagent \"<query>\"`, `-f <path>`, or `devagent review ...`.") + "\n");
      process.exit(1);
    }
    await runSingleQuery({
      ...commonOptions,
      query,
    });
  } finally {
    crashSessionReporter.dispose();

    // Print LSP tool usage summary (for measuring value)
    if (tools.lspToolCounts.size > 0 && cliArgs.verbosity !== "quiet") {
      const parts = [...tools.lspToolCounts.entries()]
        .map(([name, count]) => `${name}=${count}`)
        .join(", ");
      process.stderr.write(dim(`[lsp-usage] ${parts}`) + "\n");
    }

    // Session summary
    const delegatedWork = outputState.buildDelegatedWorkSummary();
    persistence.sessionStore.updateSessionMetadata(persistence.session.id, {
      delegatedWork,
    });
    if (cliArgs.verbosity !== "quiet" && isCategoryEnabled("session", tools.verbosityConfig)) {
      const planSteps = tools.sessionState.getPlan();
      process.stderr.write(formatSessionSummary({
        sessionId: persistence.session.id,
        totalIterations: outputState.sessionTotalIterations,
        totalToolCalls: outputState.sessionTotalToolCalls,
        toolUsage: outputState.sessionToolUsage,
        filesChanged: tools.sessionState.getModifiedFiles(),
        planSteps: planSteps
          ? planSteps.map((s) => ({ description: s.description, status: s.status }))
          : undefined,
        totalCost: outputState.sessionTotalCost,
        totalInputTokens: outputState.sessionTotalInputTokens,
        totalOutputTokens: outputState.sessionTotalOutputTokens,
        elapsedMs: Date.now() - sessionStartTime,
        completionReason: "completed",
        delegatedWork,
      }) + "\n");
    }

    // Print session ID for future resume
    if (cliArgs.verbosity !== "quiet") {
      process.stderr.write(dim(`[session] ${persistence.session.id}`) + "\n");
    }

    // Cleanup
    if (lsp.lspRouter) {
      try {
        await lsp.lspRouter.stopAll();
      } catch {
        // Servers might already be dead
      }
    }
    persistence.eventLogger?.close();
    persistence.sessionStore.close();
  }
}

// ─── Event Handlers ──────────────────────────────────────────

/** Shared spinner instance — started during LLM thinking, stopped on tool/text events. */
const spinner = new Spinner();

/** Centralized mutable output state — replaces former module-level `let` declarations. */
const outputState = new OutputState();

function setupEventHandlers(
  bus: EventBus,
  config: DevAgentConfig,
  verbosity: Verbosity,
  verbosityConfig?: VerbosityConfig,
  os: OutputState = outputState,
): { lspToolCounts: Map<string, number> } {
  const maxIter = config.budget.maxIterations;
  const vc = verbosityConfig ?? buildVerbosityConfig(verbosity, undefined);
  const toolParamsCache = new Map<string, Record<string, unknown>>();
  const subagentRenderer = new SubagentPanelRenderer(
    verbosity !== "quiet" || isCategoryEnabled("tools", vc),
  );

  function hasRunningSubagents(): boolean {
    for (const panel of os.subagentDisplay.values()) {
      if (panel.status === "running") return true;
    }
    return false;
  }

  function withSubagentPanelsHidden(action: () => void): void {
    if (!subagentRenderer.active) {
      action();
      return;
    }
    subagentRenderer.suspend();
    action();
    if (hasRunningSubagents()) {
      subagentRenderer.resume();
      subagentRenderer.setPanels([...os.subagentDisplay.values()]);
      return;
    }
    subagentRenderer.setPanels([]);
  }

  function writeUi(text: string): void {
    withSubagentPanelsHidden(() => {
      process.stderr.write(text);
    });
  }

  function writeUiLine(line: string): void {
    writeUi(line + "\n");
  }

  function pushSubagentActivity(state: SubagentDisplayState, activity: string): SubagentDisplayState {
    const nextActivity = activity.trim();
    if (nextActivity.length === 0) return state;
    const prior = nextActivity === state.currentActivity
      ? state.recentActivity
      : [nextActivity, ...state.recentActivity.filter((entry) => entry !== nextActivity)].slice(0, 2);
    return {
      ...state,
      currentActivity: nextActivity,
      recentActivity: prior,
    };
  }

  function flushToolGroup(): void {
    const group = os.pendingToolGroup;
    if (!group || group.count <= 1) {
      os.pendingToolGroup = null;
      return;
    }
    os.pendingToolGroup = null;

    withSubagentPanelsHidden(() => {
      spinner.log(formatToolGroupStart(group.name, group.count, group.params, group.iteration, group.maxIter));
      spinner.log(formatToolGroupEnd(group.name, group.count, group.lastSuccess, group.totalDurationMs, group.lastError));
    });
  }

  // ─── Iteration tracking ─────────────────────────────────────

  bus.on("iteration:start", (event) => {
    if (event.agentId) return;
    os.currentIteration = event.iteration;
    os.currentTokens = event.estimatedTokens;
    os.maxContextTokens = event.maxContextTokens;
  });

  // ─── Tool events ──────────────────────────────────────────

  bus.on("tool:before", (event) => {
    if (event.name.startsWith("audit:")) return;

    // Stop spinner — a tool call arrived
    spinner.stop();

    // Show buffered thinking text as dimmed reasoning, then clear
    if (os.textBuffer.trim() && verbosity !== "quiet") {
      const reasoningLine = formatReasoning(os.textBuffer);
      if (reasoningLine) {
        spinner.log(reasoningLine);
      }
    }
    os.textBuffer = "";

    os.hadToolCalls = true;
    os.turnToolCallCount++;
    toolParamsCache.set(`${event.agentId ?? "root"}:${event.callId}`, event.params);

    if (event.agentId) {
      const existing = os.sessionSubagents.get(event.agentId);
      if (existing) {
        os.sessionSubagents.set(event.agentId, {
          ...existing,
          toolCalls: existing.toolCalls + 1,
        });
      }
      return;
    }

    if (event.name === "delegate") return;

    if (verbosity === "quiet" && !isCategoryEnabled("tools", vc)) return;

    const gauge = isCategoryEnabled("context", vc)
      ? formatContextGauge(os.currentTokens, os.maxContextTokens)
      : undefined;

    // Verbose mode: no grouping, show every call individually
    if (isCategoryEnabled("tools", vc)) {
      flushToolGroup();
      const line = formatToolStart(event.name, event.params, os.currentIteration, maxIter, gauge);
      writeUiLine(line);
      writeUiLine(dim(`  params: ${JSON.stringify(event.params, null, 2)}`));
      return;
    }

    if (verbosity === "quiet") return;

    const summary = summarizeToolParams(event.name, event.params);

    // Grouping: check if same tool as pending group
    if (os.pendingToolGroup && os.pendingToolGroup.name === event.name) {
      os.pendingToolGroup.count++;
      if (summary) os.pendingToolGroup.params.push(summary);
      return;  // Don't print individual start line
    }

    // Different tool or no group — flush previous group and start new one
    flushToolGroup();
    os.pendingToolGroup = {
      name: event.name,
      count: 1,
      params: summary ? [summary] : [],
      totalDurationMs: 0,
      lastSuccess: true,
      lastError: undefined,
      iteration: os.currentIteration,
      maxIter,
    };
    // Show the first call immediately (it might end up being a single call)
    writeUiLine(
      formatToolStart(event.name, event.params, os.currentIteration, maxIter, gauge),
    );
  });

  bus.on("tool:after", (event) => {
    if (event.name.startsWith("audit:")) return;

    // Stop spinner before writing
    spinner.stop();

    const cacheKey = `${event.agentId ?? "root"}:${event.callId}`;
    const cachedParams = toolParamsCache.get(cacheKey);
    toolParamsCache.delete(cacheKey);

    if (event.agentId) {
      return;
    }

    if (event.name === "delegate" && event.result.metadata?.["agentMeta"]) {
      if (verbosity !== "quiet" && !hasRunningSubagents()) {
        spinner.start();
      }
      return;
    }

    if (verbosity === "quiet" && !isCategoryEnabled("tools", vc) && event.result.success) return;

    // Accumulate into group if applicable
    if (os.pendingToolGroup && os.pendingToolGroup.name === event.name && !isCategoryEnabled("tools", vc)) {
      os.pendingToolGroup.totalDurationMs += event.durationMs;
      if (!event.result.success) {
        os.pendingToolGroup.lastSuccess = false;
        os.pendingToolGroup.lastError = event.result.error ?? undefined;
      }

      if (os.pendingToolGroup.count === 1) {
        // Single call so far — show individual end line
        writeUiLine(formatToolEnd(
          event.name,
          event.result.success,
          event.durationMs,
          event.result.error ?? undefined,
          cachedParams,
        ));
      }
      // If count > 1, end line will be shown when group flushes
    } else {
      // Not in a group (or verbose mode) — show individual end line
      writeUiLine(formatToolEnd(
        event.name,
        event.result.success,
        event.durationMs,
        event.result.error ?? undefined,
        cachedParams,
      ));
    }

    if (isCategoryEnabled("tools", vc) && event.result.output) {
      const output = event.result.output.length > 500
        ? event.result.output.substring(0, 500) + "…"
        : event.result.output;
      writeUiLine(dim(`  output: ${output}`));
    }

    // Restart spinner after tool completes
    if (verbosity !== "quiet" && !hasRunningSubagents()) {
      spinner.start();
    }
  });

  // ─── LSP tool usage tracking ────────────────────────────────

  const lspToolNames = new Set(["diagnostics", "definitions", "references", "symbols", "definition_by_name", "references_by_name"]);
  const lspToolCounts = new Map<string, number>();
  bus.on("tool:after", (event) => {
    if (lspToolNames.has(event.name)) {
      lspToolCounts.set(event.name, (lspToolCounts.get(event.name) ?? 0) + 1);
    }
  });

  // ─── Plan updates ──────────────────────────────────────────

  bus.on("plan:updated", (event) => {
    if (verbosity === "quiet" && !isCategoryEnabled("plan", vc)) return;

    spinner.stop();
    writeUi("\n" + dim("── Plan ──") + "\n");
    writeUi(formatPlan(event.steps) + "\n\n");
  });

  // ─── Subagent lifecycle ───────────────────────────────────

  bus.on("subagent:start", (event) => {
    spinner.stop();
    flushToolGroup();
    os.sessionSubagents.set(
      event.agentId,
      loggedSubagentRunFromEvent(event, os.sessionSubagents.get(event.agentId)),
    );
    os.subagentDisplay.set(event.agentId, {
      agentId: event.agentId,
      agentType: event.agentType,
      laneLabel: event.laneLabel,
      model: event.model,
      reasoningEffort: event.reasoningEffort,
      status: "running",
      currentIteration: 0,
      startedAtMs: Date.now(),
      currentActivity: "Waiting for first action",
      recentActivity: [],
    });

    if (verbosity === "quiet" && !isCategoryEnabled("tools", vc)) return;
    if (event.batchId && (event.batchSize ?? 0) > 1 && !os.announcedSubagentBatches.has(event.batchId)) {
      os.announcedSubagentBatches.add(event.batchId);
      writeUiLine(formatSubagentBatchLaunch(event.agentType, event.batchSize ?? 0));
    }
    writeUiLine(formatSubagentStart(event));
    if (subagentRenderer.active) {
      subagentRenderer.setPanels([...os.subagentDisplay.values()]);
    }
  });

  bus.on("subagent:update", (event) => {
    const existing = os.subagentDisplay.get(event.agentId);
    if (!existing) return;
    const summary = summarizeSubagentUpdate(event);
    const nextIteration = event.iteration ?? existing.currentIteration;
    const nextState = pushSubagentActivity({
      ...existing,
      currentIteration: nextIteration,
    }, summary);
    os.subagentDisplay.set(event.agentId, nextState);

    if (verbosity === "quiet" && !isCategoryEnabled("tools", vc)) return;
    if (subagentRenderer.active) {
      subagentRenderer.setPanels([...os.subagentDisplay.values()]);
      return;
    }
    if (event.milestone === "iteration:start") {
      writeUiLine(`  ${dim(`[${event.agentId}:${event.iteration ?? 0}]`)} ${summary}`);
    }
  });

  bus.on("subagent:end", (event) => {
    spinner.stop();
    os.sessionSubagents.set(
      event.agentId,
      loggedSubagentRunFromEvent(event, os.sessionSubagents.get(event.agentId)),
    );
    const display = os.subagentDisplay.get(event.agentId);
    if (display) {
      os.subagentDisplay.set(event.agentId, pushSubagentActivity({
        ...display,
        status: "completed",
        durationMs: event.durationMs,
        currentIteration: event.iterations,
        quality: event.quality
          ? {
              score: event.quality.score,
              completeness: event.quality.completeness,
            }
          : undefined,
      }, `Completed after ${event.iterations} iterations`));
    }

    if (verbosity === "quiet" && !isCategoryEnabled("tools", vc)) return;
    if (subagentRenderer.active) {
      subagentRenderer.setPanels([...os.subagentDisplay.values()]);
    }
    if (verbosity !== "quiet" && !hasRunningSubagents()) {
      spinner.start();
    }
  });

  bus.on("subagent:error", (event) => {
    spinner.stop();
    os.sessionSubagents.set(
      event.agentId,
      loggedSubagentRunFromEvent(event, os.sessionSubagents.get(event.agentId)),
    );
    const display = os.subagentDisplay.get(event.agentId);
    if (display) {
      os.subagentDisplay.set(event.agentId, pushSubagentActivity({
        ...display,
        status: "error",
        durationMs: event.durationMs,
      }, `Failed: ${event.error}`));
    }

    if (verbosity === "quiet" && !isCategoryEnabled("tools", vc)) return;
    if (subagentRenderer.active) {
      subagentRenderer.setPanels([...os.subagentDisplay.values()]);
    } else {
      writeUiLine(formatSubagentError(event));
    }
    if (verbosity !== "quiet") {
      spinner.start();
    }
  });

  bus.on("message:tool", (event) => {
    if (event.agentId) return;
    if (event.toolName !== "delegate" || event.summaryOnly !== true) return;
    if (verbosity === "quiet" && !isCategoryEnabled("tools", vc)) return;
    writeUiLine(`  ${green("✓")} ${dim(event.content)}`);
  });

  // ─── Message events (spinner management) ───────────────────

  bus.on("message:user", () => {
    // Start spinner when user query is sent (waiting for LLM)
    if (verbosity !== "quiet") {
      spinner.start();
    }
  });

  bus.on("message:assistant", (event) => {
    if (event.agentId) {
      return;
    }
    if (event.partial) {
      spinner.stop();
      os.textBuffer += event.content;
    } else {
      flushToolGroup();
    }
  });

  // ─── Context compaction ──────────────────────────────────

  bus.on("context:compacting", (event) => {
    spinner.stop();
    if (verbosity !== "quiet" || isCategoryEnabled("context", vc)) {
      writeUiLine(
        dim(`[context] Compacting… (~${event.estimatedTokens} tokens, limit ${event.maxTokens})`),
      );
      spinner.start("Compacting context…");
    }
  });

  bus.on("context:compacted", (event) => {
    spinner.stop();
    if (verbosity !== "quiet" || isCategoryEnabled("context", vc)) {
      writeUiLine(
        formatCompactionResult({
          tokensBefore: event.tokensBefore,
          estimatedTokens: event.estimatedTokens,
          removedCount: event.removedCount,
          prunedCount: event.prunedCount,
        }),
      );
    }
  });

  // ─── Per-turn and session cost/token tracking ──────────────

  bus.on("cost:update", (event) => {
    os.turnInputTokens += event.inputTokens;
    os.turnCostDelta += event.totalCost;
    os.sessionTotalInputTokens += event.inputTokens;
    os.sessionTotalOutputTokens += event.outputTokens;
    os.sessionTotalCost += event.totalCost;
  });

  // Session-level iteration and tool tracking
  bus.on("iteration:start", () => {
    os.sessionTotalIterations++;
  });
  bus.on("tool:before", (event) => {
    if (!event.name.startsWith("audit:")) {
      os.sessionTotalToolCalls++;
      os.sessionToolUsage.set(event.name, (os.sessionToolUsage.get(event.name) ?? 0) + 1);
    }
  });

  // ─── Rolling tool results window (for error enrichment) ────

  const recentToolResults: Array<{ name: string; success: boolean; durationMs: number }> = [];
  bus.on("tool:after", (event) => {
    recentToolResults.push({ name: event.name, success: event.result.success, durationMs: event.durationMs });
    if (recentToolResults.length > 5) recentToolResults.shift();
  });

  // ─── Errors ────────────────────────────────────────────────

  bus.on("error", (event) => {
    spinner.stop();
    if (recentToolResults.length > 0) {
      const suggestion = inferErrorSuggestion(event.message, recentToolResults);
      writeUiLine(formatEnrichedError({
        message: event.message,
        recentTools: [...recentToolResults],
        suggestion,
      }));
    } else {
      writeUiLine(formatError(event.message));
    }
  });

  // ─── Approval prompts ─────────────────────────────────────

  bus.on("approval:request", (event) => {
    spinner.stop();
    const rl = createInterface({
      input: process.stdin,
      output: process.stdout,
    });
    rl.question(
      yellow(`[approval] ${event.toolName}: ${event.details}\nApprove? (y/n): `),
      (answer) => {
        rl.close();
        bus.emit("approval:response", {
          id: event.id,
          approved: answer.toLowerCase().startsWith("y"),
        });
      },
    );
  });

  return { lspToolCounts };
}

// ─── Helpers ─────────────────────────────────────────────────

/**
 * Set up the LLM-based summarization callback for context compaction.
 * The callback sends older messages to the LLM with a compaction prompt
 * and returns the summary text (following the Codex handoff pattern).
 */
function setupSummarizeCallback(
  contextManager: ContextManager,
  provider: LLMProvider,
  sessionState?: SessionState,
): void {
  contextManager.setSummarizeCallback(async (messages) => {
    // Build context-aware compaction instructions
    let stateContext = "";
    if (sessionState) {
      const stateSnapshot = sessionState.toSystemMessage("compact");
      if (stateSnapshot) {
        stateContext = `\n\nThe following session state has already been saved and will persist across compaction:\n${stateSnapshot}\n\nDo NOT repeat raw diff content or file contents that are already captured in findings or tool summaries above.`;
      }
    }

    const summaryPrompt = [
      {
        role: MessageRole.SYSTEM,
        content:
          `You are performing a CONTEXT CHECKPOINT COMPACTION. Create a handoff summary for another LLM that will resume the task.

IMPORTANT RULES:
1. Preserve EXACT file paths and line numbers of all findings and issues
2. Do NOT include raw diff content or raw file contents — the model already has saved findings
3. Focus on WHAT was analyzed, WHAT was found, and WHAT remains to do
4. Include key decisions made, constraints, and user preferences

Structure your summary as:
- **Progress**: What has been analyzed/completed (with exact file paths)
- **Findings**: Key issues or observations (with file:line references)
- **Remaining**: What still needs to be done (clear next steps)

Be concise and structured.${stateContext}`,
      },
      ...messages,
      {
        role: MessageRole.USER,
        content:
          "Summarize the conversation above into a concise handoff summary. Preserve all file paths and line numbers.",
      },
    ];

    let summary = "";
    const stream = provider.chat(summaryPrompt, []);
    for await (const chunk of stream) {
      if (chunk.type === "text") {
        summary += chunk.content;
      }
    }
    return summary || "No summary available.";
  });
}

/**
 * Flush buffered output and handle TaskCompletionStatus.
 * Falls back to result.lastText when the final streamed response is empty.
 */
function flushOutput(result: TaskLoopResult, verbosity: Verbosity, os: OutputState = outputState): void {
  const streamed = os.textBuffer.trim();

  if (streamed) {
    if (os.hadToolCalls) process.stderr.write("\n");
    process.stdout.write(os.textBuffer + "\n");
  } else if (result.lastText?.trim()) {
    // No final response, but the LLM produced text earlier in the session
    if (verbosity !== "quiet") {
      process.stderr.write(
        yellow("[warning] No final response. Showing last output from agent:") + "\n",
      );
    }
    process.stdout.write(result.lastText + "\n");
  }

  // Status-specific messages
  if (verbosity !== "quiet") {
    switch (result.status) {
      case "empty_response":
        if (!os.textBuffer.trim() && !result.lastText?.trim()) {
          process.stderr.write(
            yellow("[warning] Agent completed but produced no output.") + "\n",
          );
        }
        break;
      case "budget_exceeded":
        process.stderr.write(
          yellow("[warning] Iteration limit reached — partial results shown.") + "\n",
        );
        break;
      case "aborted":
        process.stderr.write(dim("[info] Agent was interrupted.") + "\n");
        break;
    }
  }
}

// ─── Midpoint Callback Factory ───────────────────────────────

function createMidpointCallback(opts: {
  provider: import("@devagent/runtime").LLMProvider;
  mode: TaskMode;
  repoRoot: string;
  skills: SkillRegistry;
  toolRegistry: import("@devagent/runtime").ToolRegistry;
  config: DevAgentConfig;
  getTurnNumber: () => number;
}): MidpointCallback {
  return async (messages, _iteration) => {
    const midBriefing = await synthesizeBriefing(messages, opts.getTurnNumber(), {
      strategy: "auto",
      provider: opts.provider,
    });
    const freshPrompt = assembleSystemPrompt({
      mode: opts.mode,
      repoRoot: opts.repoRoot,
      skills: opts.skills,
      availableTools: opts.toolRegistry.getAll(),
      approvalMode: opts.config.approval.mode,
      provider: opts.config.provider,
      model: opts.config.model,
      agentModelOverrides: opts.config.agentModelOverrides,
      agentReasoningOverrides: opts.config.agentReasoningOverrides,
      briefing: midBriefing,
    });
    const lastUserContent = findLastUserContent(messages);
    return {
      continueMessages: [
        { role: MessageRole.SYSTEM, content: freshPrompt },
        { role: MessageRole.USER, content: lastUserContent },
      ],
    };
  };
}

// ─── Review Query Detection ─────────────────────────────────

const REVIEW_PATTERN = /\b(code\s+)?review\b/i;
const DIFF_PATTERN = /\b(uncommitted|diff|changes|staged|unstaged)\b/i;

/**
 * Check whether a user query is a review task that would benefit from
 * pre-loading the full git diff into context.
 */
export function isReviewQuery(query: string): "staged" | "unstaged" | false {
  if (!REVIEW_PATTERN.test(query)) return false;
  if (!DIFF_PATTERN.test(query)) return false;
  if (/\bstaged\b/i.test(query)) return "staged";
  return "unstaged";
}

/**
 * Pre-load the git diff for review queries.
 * Returns a pre-formatted user message or null if not a review query.
 */
async function maybePreloadReviewDiff(
  query: string,
  repoRoot: string,
): Promise<string | null> {
  const reviewType = isReviewQuery(query);
  if (!reviewType) return null;

  try {
    const cmd = reviewType === "staged" ? "git diff --cached" : "git diff";
    const diffOutput = execSync(cmd, {
      cwd: repoRoot,
      encoding: "utf-8",
      maxBuffer: 1024 * 1024 * 5, // 5MB
      timeout: 10_000,
    });

    if (!diffOutput.trim()) return null;

    const truncated = truncateToolOutput(diffOutput);
    return `[Pre-loaded ${reviewType} diff for review]\n\n${truncated}`;
  } catch {
    return null; // Fail silently — the model will fetch its own diff
  }
}

// ─── Single Query ────────────────────────────────────────────

async function runSingleQuery(options: RunSingleQueryOptions): Promise<void> {
  const {
    query, provider, toolRegistry, bus, gate, config, repoRoot, mode,
    skills, contextManager, doubleCheck, initialMessages, verbosity, sessionState, briefing,
  } = options;

  const systemPrompt = assembleSystemPrompt({
    mode,
    repoRoot,
    skills,
    availableTools: toolRegistry.getAll(),
    approvalMode: config.approval.mode,
    provider: config.provider,
    model: config.model,
    agentModelOverrides: config.agentModelOverrides,
    agentReasoningOverrides: config.agentReasoningOverrides,
    briefing,
  });

  // Set up LLM-based summarization for context compaction
  setupSummarizeCallback(contextManager, provider, sessionState);

  const midpointCallback = createMidpointCallback({
    provider,
    mode,
    repoRoot,
    skills,
    toolRegistry,
    config,
    getTurnNumber: () => 0,
  });

  // Pre-load diff for review tasks to eliminate discovery overhead
  let effectiveInitialMessages = initialMessages;
  if (!initialMessages) {
    const preloadedDiff = await maybePreloadReviewDiff(query, repoRoot);
    if (preloadedDiff) {
      effectiveInitialMessages = [
        { role: MessageRole.SYSTEM, content: systemPrompt },
        { role: MessageRole.USER, content: preloadedDiff, pinned: true },
      ];
    }
  }

  outputState.resetTurn();
  const startTime = Date.now();

  const loop = new TaskLoop({
    provider,
    tools: toolRegistry,
    bus,
    approvalGate: gate,
    config,
    systemPrompt,
    repoRoot,
    mode,
    contextManager,
    doubleCheck,
    initialMessages: effectiveInitialMessages,
    sessionState,
    midpointCallback,
  });
  const result = await loop.run(query);

  // Stop spinner in case LLM finished without emitting text
  spinner.stop();

  // Flush buffered final response to stdout
  flushOutput(result, verbosity);

  if (verbosity !== "quiet") {
    const elapsed = Date.now() - startTime;
    process.stderr.write(formatTurnSummary({
      iterationCount: result.iterations,
      toolCallCount: outputState.turnToolCallCount,
      inputTokens: outputState.turnInputTokens,
      costDelta: outputState.turnCostDelta,
      elapsedMs: elapsed,
    }) + "\n");
  }
}
