import {
  SessionState,
  SessionStore,
  extractErrorMessage,
} from "@devagent/runtime";
import { readFileSync as nodeReadFileSync } from "node:fs";

import {
  loadQueryFromFile,
  parseArgs,
  printHelp,
  printHelpUsageError,
  renderReviewHelpText,
  writeStdout,
} from "./cli-args.js";
import { renderSessionsList } from "./cli-sessions.js";
import { checkForUpdates, getVersion } from "./cli-version.js";
import { dim, formatError, formatSessionSummary, green, isCategoryEnabled, red, setTerminalTitle, yellow } from "./format.js";
import { setupConfig, setupProvider, validateOllamaAvailability } from "./main-config-setup.js";
import { setupLSP } from "./main-lsp-setup.js";
import {
  abortTuiQuery,
  buildInteractiveSystemPrompt,
  renderStdoutForSingleShot,
  resetTuiLoop,
  runSingleQuery,
  runTuiQuery,
  updateTuiSystemPrompt,
} from "./main-query.js";
import { getInteractiveSafetyMode, withInteractiveSafetyMode } from "./main-safety.js";
import { resolveResumeTarget, setupSessionPersistence } from "./main-session-persistence.js";
import { outputState } from "./main-state.js";
import { setupTools } from "./main-tools-setup.js";
import type {
  CliArgs,
  DelegatedWorkSummary,
  InteractiveResumeSeed,
  LSPSetupResult,
  ProviderSetupResult,
  RunOptions,
  SessionPersistenceResult,
  ToolsSetupResult,
} from "./main-types.js";
import { buildSessionPreview } from "./session-preview.js";
import type { InteractiveQueryResult } from "./tui/shared.js";
import type { DevAgentConfig, LLMProvider, SafetyMode, Session } from "@devagent/runtime";

export { loadQueryFromFile, parseArgs, renderHelpText } from "./cli-args.js";
export { renderSessionsList } from "./cli-sessions.js";
export { checkForUpdates, getVersion } from "./cli-version.js";
export { createCrashSessionReporter } from "./cli-crash-session.js";
export {
  buildInteractiveSystemPrompt,
  createInitialTuiLoopOptions,
  isReviewQuery,
  resolveAutoPromptCommandTarget,
} from "./main-query.js";
export { setupSessionPersistence } from "./main-session-persistence.js";
export type { CliArgs, Verbosity } from "./main-types.js";

interface RuntimeServices {
  readonly tools: ToolsSetupResult;
  readonly lsp: LSPSetupResult;
  readonly persistence: SessionPersistenceResult;
}

interface AgentSessionContext {
  readonly cliArgs: CliArgs;
  readonly config: DevAgentConfig;
  readonly lsp: LSPSetupResult;
  readonly persistence: SessionPersistenceResult;
  readonly projectRoot: string;
  readonly provider: LLMProvider;
  readonly tools: ToolsSetupResult;
}

function hasContinueInputConflict(cliArgs: CliArgs): boolean {
  return Boolean(cliArgs.continue_ && (cliArgs.query || cliArgs.file));
}

function shouldSkipResumePreflight(cliArgs: CliArgs): boolean {
  return Boolean((!cliArgs.resume && !cliArgs.continue_) || cliArgs.query || cliArgs.file);
}

function preflightResumeRequest(cliArgs: CliArgs): void {
  if (hasContinueInputConflict(cliArgs)) {
    process.stderr.write(formatError("--continue does not accept a query or file input. Use --resume <id> to target a specific session.") + "\n");
    process.exit(2);
  }
  if (shouldSkipResumePreflight(cliArgs)) return;
  const sessionStore = new SessionStore();
  if (cliArgs.resume) {
    resolveResumeTarget(sessionStore, cliArgs.resume);
    return;
  }
  if ((sessionStore.listSessions(1)[0] ?? null) === null) {
    process.stderr.write(yellow("[session] No session found: most recent") + "\n");
    process.exit(1);
  }
}

async function maybeRunExecuteCommand(): Promise<boolean> {
  if (process.argv[2] !== "execute") return false;
  const { executeTask, loadTaskExecutionRequest, parseExecuteArgs } = await import("@devagent/executor");
  if (process.argv.includes("--help") || process.argv.includes("-h")) {
    process.stdout.write("Usage: devagent execute --request <file> --artifact-dir <dir>\n");
    return true;
  }
  const executeArgs = parseExecuteArgs(process.argv);
  if (!executeArgs) process.exit(1);
  const request = await loadTaskExecutionRequest(executeArgs.requestPath);
  try {
    const { setupAndRunWorkflowQuery } = await import("./workflow-engine.js");
    const result = await executeTask({
      request,
      artifactDir: executeArgs.artifactDir,
      repoRoot: process.cwd(),
      runQuery: setupAndRunWorkflowQuery,
      emit: (event) => { process.stdout.write(JSON.stringify(event) + "\n"); },
    });
    if (result.status !== "success") process.exit(1);
  } catch {
    process.exit(1);
  }
  return true;
}

function validateReviewArgs(cliArgs: CliArgs): boolean {
  if (!cliArgs.review) return false;
  if (cliArgs.review.help) {
    process.stdout.write(renderReviewHelpText() + "\n");
    return true;
  }
  if (!cliArgs.review.patchFile) {
    process.stderr.write(formatError(renderReviewHelpText()) + "\n");
    process.exit(1);
  }
  if (!cliArgs.review.ruleFile) {
    process.stderr.write(formatError("Rule file required: devagent review <file> --rule <rule_file>") + "\n");
    process.exit(1);
  }
  return false;
}

function validateCliArgs(cliArgs: CliArgs): void {
  if (cliArgs.modeParseError) {
    process.stderr.write(formatError(cliArgs.modeParseError) + "\n");
    process.exit(2);
  }
  if (cliArgs.usageError) {
    process.stderr.write(formatError(cliArgs.usageError) + "\n");
    process.exit(2);
  }
}

async function maybeRunSubcommand(cliArgs: CliArgs): Promise<boolean> {
  if (!cliArgs.subcommand) return false;
  const cmd = cliArgs.subcommand.name;
  const cmdArgs = cliArgs.subcommand.args;
  const [
    { runDoctor },
    { runConfig },
    { runSetup, runConfigure, runInit },
    { runUpdate },
    { runInstallLsp },
    { runCompletions },
  ] = await Promise.all([
    import("./commands/doctor.js"),
    import("./commands/config.js"),
    import("./commands/setup.js"),
    import("./commands/update.js"),
    import("./commands/install-lsp.js"),
    import("./commands/completions.js"),
  ]);
  if (cmd === "help") {
    if (cmdArgs.length > 0) printHelpUsageError();
    printHelp();
    return true;
  }
  type NonHelpSubcommand = Exclude<NonNullable<CliArgs["subcommand"]>["name"], "help">;
  const handlers: Record<NonHelpSubcommand, () => void | Promise<void>> = {
    doctor: () => runDoctor(getVersion(), cmdArgs),
    config: () => { runConfig(cmdArgs); },
    setup: () => runSetup(cmdArgs),
    init: () => { runInit(cmdArgs); },
    configure: () => runConfigure(cmdArgs),
    update: () => runUpdate(cmdArgs),
    "install-lsp": () => { runInstallLsp(cmdArgs); },
    completions: () => { runCompletions(cmdArgs); },
  };
  await handlers[cmd]();
  return true;
}

async function maybeRunEarlyCommand(cliArgs: CliArgs): Promise<boolean> {
  if (cliArgs.sessionsCommand) {
    const store = new SessionStore();
    process.stderr.write(renderSessionsList(store.listSessions(20)));
    return true;
  }
  if (cliArgs.authCommand) {
    const { runAuthCommand } = await import("./auth.js");
    await runAuthCommand(cliArgs.authCommand.subcommand, cliArgs.authCommand.args);
    return true;
  }
  return false;
}

async function maybeRunReviewCommand(cliArgs: CliArgs, provider: LLMProvider, projectRoot: string): Promise<boolean> {
  if (!cliArgs.review) return false;
  const { runReviewPipeline } = await import("@devagent/runtime");
  try {
    const result = await runReviewPipeline(
      { provider, workspaceRoot: projectRoot },
      { patchFile: cliArgs.review.patchFile, ruleFile: cliArgs.review.ruleFile! },
    );
    renderReviewResult(result, cliArgs.review.jsonOutput);
  } catch (err) {
    process.stderr.write(formatError(`Review failed: ${extractErrorMessage(err)}`) + "\n");
    process.exit(1);
  }
  return true;
}

function renderReviewResult(result: Awaited<ReturnType<typeof import("@devagent/runtime").runReviewPipeline>>, jsonOutput: boolean): void {
  if (jsonOutput) {
    process.stdout.write(JSON.stringify(result, null, 2) + "\n");
    return;
  }
  const { violations, summary } = result;
  if (violations.length === 0) writeStdout(green("No violations found."));
  else {
    writeStdout(`${dim("")}${violations.length === 0 ? "" : ""}`.slice(0, 0) + `\x1b[1mFound ${violations.length} violation(s) in ${summary.filesReviewed} file(s):\x1b[0m\n`);
    for (const violation of violations) renderReviewViolation(violation);
  }
  writeStdout(dim(`Rule: ${summary.ruleName} | Files: ${summary.filesReviewed} | Violations: ${summary.totalViolations}`));
}

function renderReviewViolation(violation: Awaited<ReturnType<typeof import("@devagent/runtime").runReviewPipeline>>["violations"][number]): void {
  const sevColor = violation.severity === "error" ? red : violation.severity === "warning" ? yellow : dim;
  writeStdout(`  ${sevColor(violation.severity.toUpperCase().padEnd(7))} ${dim(violation.file)}:${violation.line}`);
  writeStdout(`           ${violation.message}`);
  if (violation.codeSnippet) writeStdout(`           ${dim(violation.codeSnippet)}`);
  writeStdout();
}

async function setupRuntimeServices(options: {
  readonly cliArgs: CliArgs;
  readonly config: DevAgentConfig;
  readonly projectRoot: string;
  readonly provider: LLMProvider;
  readonly providerRegistry: ProviderSetupResult["providerRegistry"];
}): Promise<RuntimeServices> {
  const tools = setupTools(options.config, options.cliArgs, options.projectRoot, options.provider, options.providerRegistry);
  const lsp = await setupLSP({
    config: options.config,
    cliArgs: options.cliArgs,
    projectRoot: options.projectRoot,
    toolRegistry: tools.toolRegistry,
    doubleCheck: tools.doubleCheck,
    trackInternalLSPDiagnostics: tools.trackInternalLSPDiagnostics,
  });
  const persistence = await setupSessionPersistence(
    options.config, options.cliArgs, options.projectRoot, options.provider, tools.bus, tools.sessionState,
  );
  tools.sessionState = persistence.sessionState;
  return { tools, lsp, persistence };
}

function buildRunOptions(ctx: AgentSessionContext, seed?: InteractiveResumeSeed): RunOptions {
  return {
    provider: ctx.provider,
    toolRegistry: ctx.tools.toolRegistry,
    bus: ctx.tools.bus,
    gate: ctx.tools.gate,
    config: ctx.config,
    repoRoot: ctx.projectRoot,
    mode: "act",
    skills: ctx.tools.skills,
    contextManager: ctx.tools.contextManager,
    doubleCheck: ctx.tools.doubleCheck,
    initialMessages: seed?.initialMessages ?? ctx.persistence.initialMessages,
    verbosity: ctx.cliArgs.verbosity,
    verbosityConfig: ctx.tools.verbosityConfig,
    sessionState: ctx.tools.sessionState,
    briefing: seed?.briefing ?? ctx.persistence.resumeBriefing,
    lspSync: ctx.lsp.lspSync,
  };
}

function resolveCliQuery(cliArgs: CliArgs): string | null {
  return cliArgs.file ? loadQueryFromFile(cliArgs.file, nodeReadFileSync, cliArgs.query) : cliArgs.query;
}

async function runAgentSession(ctx: AgentSessionContext): Promise<void> {
  try {
    await runRequestedMode(ctx);
  } finally {
    await finalizeAgentSession(ctx);
  }
}

async function runRequestedMode(ctx: AgentSessionContext): Promise<void> {
  const seed: InteractiveResumeSeed = { initialMessages: ctx.persistence.initialMessages, briefing: ctx.persistence.resumeBriefing };
  const query = resolveCliQuery(ctx.cliArgs);
  if (!query) await runInteractiveMode(ctx, seed);
  else if (process.stderr.isTTY && ctx.cliArgs.verbosity !== "quiet") await runSingleShotTuiMode(ctx, query);
  else {
    ctx.persistence.activateSession(query);
    await runSingleQuery({ ...buildRunOptions(ctx), query });
  }
}

async function runInteractiveMode(ctx: AgentSessionContext, seed: InteractiveResumeSeed): Promise<void> {
  if (!process.stdin.isTTY) {
    process.stderr.write(formatError("Interactive TUI requires a terminal. Provide a query: devagent \"<query>\"") + "\n");
    process.exit(1);
  }
  const { startTui } = await import("./tui/index.js");
  const approvalState = { current: getInteractiveSafetyMode(ctx.config) };
  await startTui({
    bus: ctx.tools.bus,
    model: ctx.config.model,
    approvalMode: approvalState.current,
    cwd: ctx.projectRoot,
    version: getVersion(),
    onListSessions: () => listSessionPreviews(ctx.persistence),
    onQuery: createInteractiveQueryHandler(ctx, seed, approvalState),
    onCancelQuery: abortTuiQuery,
    onCycleApprovalMode: createApprovalModeCycler(ctx, seed, approvalState),
    onClear: createInteractiveClearHandler(ctx, seed),
  });
}

function listSessionPreviews(persistence: SessionPersistenceResult): ReturnType<typeof buildSessionPreview>[] {
  try {
    return persistence.sessionStore.listSessions(15).map((session) => buildSessionPreview(session));
  } catch {
    return [];
  }
}

function createInteractiveQueryHandler(
  ctx: AgentSessionContext,
  seed: InteractiveResumeSeed,
  approvalState: { current: SafetyMode },
): (q: string) => Promise<InteractiveQueryResult> {
  return async (q) => {
    outputState.resetTurn();
    ctx.persistence.activateSession(q);
    const config = approvalState.current === getInteractiveSafetyMode(ctx.config)
      ? ctx.config
      : withInteractiveSafetyMode(ctx.config, approvalState.current);
    return toInteractiveQueryResult(await runTuiQuery({ ...buildRunOptions(ctx, seed), config, query: q }));
  };
}

function createApprovalModeCycler(
  ctx: AgentSessionContext,
  seed: InteractiveResumeSeed,
  approvalState: { current: SafetyMode },
): (mode: SafetyMode) => void {
  return (mode) => {
    approvalState.current = mode;
    ctx.tools.gate.setMode(mode);
    ctx.tools.delegateAmbientContext.approvalMode = mode;
    updateTuiSystemPrompt(buildInteractivePrompt(ctx, seed, mode));
  };
}

function buildInteractivePrompt(ctx: AgentSessionContext, seed: InteractiveResumeSeed, safetyMode: SafetyMode): string {
  return buildInteractiveSystemPrompt({
    mode: "act",
    repoRoot: ctx.projectRoot,
    skills: ctx.tools.skills,
    toolRegistry: ctx.tools.toolRegistry,
    config: ctx.config,
    safetyMode,
    briefing: seed.briefing,
  });
}

function createInteractiveClearHandler(ctx: AgentSessionContext, seed: InteractiveResumeSeed): () => void {
  return () => {
    const activeSession = ctx.persistence.getActiveSession();
    if (activeSession) {
      ctx.persistence.sessionStore.updateSessionMetadata(activeSession.id, {
        delegatedWork: outputState.buildDelegatedWorkSummary(),
      });
    }
    ctx.persistence.deactivateSession();
    outputState.resetSession();
    ctx.tools.sessionState = new SessionState(ctx.config.sessionState);
    ctx.persistence.sessionState = ctx.tools.sessionState;
    seed.initialMessages = undefined;
    seed.briefing = undefined;
    resetTuiLoop();
  };
}

async function runSingleShotTuiMode(ctx: AgentSessionContext, query: string): Promise<void> {
  const { startSingleShotTui } = await import("./tui/index.js");
  await startSingleShotTui({
    bus: ctx.tools.bus,
    query,
    model: ctx.config.model,
    onQuery: async (q): Promise<InteractiveQueryResult> => {
      outputState.resetTurn();
      ctx.persistence.activateSession(q);
      try {
        return toInteractiveQueryResult(await runTuiQuery({ ...buildRunOptions(ctx), query: q }));
      } finally {
        resetTuiLoop();
      }
    },
    onFinalOutput: (text) => { process.stdout.write(renderStdoutForSingleShot(text) + "\n"); },
  });
}

function toInteractiveQueryResult(result: import("@devagent/runtime").TaskLoopResult): InteractiveQueryResult {
  return {
    iterations: result.iterations,
    toolCalls: outputState.turnToolCallCount,
    lastText: outputState.textBuffer.trim() || result.lastText || null,
    status: result.status,
  };
}

async function finalizeAgentSession(ctx: AgentSessionContext): Promise<void> {
  writeLspUsageSummary(ctx.tools, ctx.cliArgs);
  writeSessionSummary(ctx);
  ctx.persistence.close();
  if (ctx.lsp.lspRouter) {
    try {
      await ctx.lsp.lspRouter.stopAll();
    } catch {
      // Servers might already be dead.
    }
  }
}

function writeLspUsageSummary(tools: ToolsSetupResult, cliArgs: CliArgs): void {
  if (tools.lspToolCounts.size === 0 || cliArgs.verbosity === "quiet") return;
  const parts = [...tools.lspToolCounts.entries()].map(([name, count]) => `${name}=${count}`).join(", ");
  process.stderr.write(dim(`[lsp-usage] ${parts}`) + "\n");
}

function writeSessionSummary(ctx: AgentSessionContext): void {
  const activeSession = ctx.persistence.getActiveSession();
  if (!activeSession) return;
  const delegatedWork = outputState.buildDelegatedWorkSummary();
  ctx.persistence.sessionStore.updateSessionMetadata(activeSession.id, { delegatedWork });
  if (ctx.cliArgs.verbosity !== "quiet" && isCategoryEnabled("session", ctx.tools.verbosityConfig)) {
    process.stderr.write(formatSessionSummary(buildSessionSummaryInput(ctx, activeSession, delegatedWork)) + "\n");
  }
  if (ctx.cliArgs.verbosity !== "quiet") ctx.persistence.printActiveSessionId();
}

function buildSessionSummaryInput(ctx: AgentSessionContext, activeSession: Session, delegatedWork: DelegatedWorkSummary): Parameters<typeof formatSessionSummary>[0] {
  const planSteps = ctx.tools.sessionState.getPlan();
  return {
    sessionId: activeSession.id,
    totalIterations: outputState.sessionTotalIterations,
    totalToolCalls: outputState.sessionTotalToolCalls,
    toolUsage: outputState.sessionToolUsage,
    filesChanged: ctx.tools.sessionState.getModifiedFiles(),
    planSteps: planSteps?.map((step) => ({ description: step.description, status: step.status })),
    totalCost: outputState.sessionTotalCost,
    totalInputTokens: outputState.sessionTotalInputTokens,
    totalOutputTokens: outputState.sessionTotalOutputTokens,
    elapsedMs: Date.now() - (ctx.persistence.getActiveSessionStartTime() ?? Date.now()),
    completionReason: "completed",
    delegatedWork,
  };
}

export async function main(): Promise<void> {
  checkForUpdates();
  if (await maybeRunExecuteCommand()) return;
  const cliArgs = parseArgs(process.argv);
  if (validateReviewArgs(cliArgs)) return;
  validateCliArgs(cliArgs);
  if (await maybeRunSubcommand(cliArgs)) return;
  if (await maybeRunEarlyCommand(cliArgs)) return;
  const { config, projectRoot } = await setupConfig(cliArgs);
  preflightResumeRequest(cliArgs);
  const { provider, providerRegistry } = setupProvider(config, cliArgs);
  if (await maybeRunReviewCommand(cliArgs, provider, projectRoot)) return;
  await validateOllamaAvailability(config);
  const { tools, lsp, persistence } = await setupRuntimeServices({ cliArgs, config, projectRoot, provider, providerRegistry });
  if (persistence.resumeTargetMissing && !cliArgs.query && !cliArgs.file) process.exit(1);
  await runAgentSession({ cliArgs, config, lsp, persistence, projectRoot, provider, tools });
  setTerminalTitle("devagent");
}
