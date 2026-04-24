import {
  MessageRole,
  TaskLoop,
  findLastUserContent,
  synthesizeBriefing,
  truncateToolOutput,
} from "@devagent/runtime";
import { execSync } from "node:child_process";

import { formatTranscriptPart, formatTurnEnd, formatTurnSummary, terminalBell, yellow, dim } from "./format.js";
import { getInteractiveSafetyMode } from "./main-safety.js";
import { nextTranscriptId, outputState, spinner, transcriptComposer } from "./main-state.js";
import type {
  InitialTuiLoopOptions,
  InteractiveSystemPromptOptions,
  RunOptions,
  RunSingleQueryOptions,
  TaskLoopRunResult,
  Verbosity,
} from "./main-types.js";
import { renderMarkdown } from "./markdown-render.js";
import {
  formatPreloadedDiffMessage,
  preparePromptCommandQuery,
  type ResolvedPromptCommandTarget,
} from "./prompt-commands.js";
import { assembleSystemPrompt } from "./prompts/index.js";
import {
  makeFinalOutputPart,
  makeInfoPart,
  makeTurnSummaryPart,
} from "./transcript-presenter.js";
import type { ContextManager, DevAgentConfig, LLMProvider, Message, MidpointCallback, SessionState, SkillRegistry, TaskLoopResult, TaskMode, ToolRegistry } from "@devagent/runtime";

export function buildInteractiveSystemPrompt(options: InteractiveSystemPromptOptions): string {
  return assembleSystemPrompt({
    mode: options.mode,
    repoRoot: options.repoRoot,
    skills: options.skills,
    availableTools: options.toolRegistry.getLoaded(),
    deferredTools: options.toolRegistry.getDeferred(),
    approvalMode: options.safetyMode,
    provider: options.config.provider,
    model: options.config.model,
    agentModelOverrides: options.config.agentModelOverrides,
    agentReasoningOverrides: options.config.agentReasoningOverrides,
    briefing: options.briefing,
  });
}

export function createInitialTuiLoopOptions(options: RunSingleQueryOptions): InitialTuiLoopOptions {
  const systemPrompt = buildInteractiveSystemPrompt({
    repoRoot: options.repoRoot,
    skills: options.skills,
    toolRegistry: options.toolRegistry,
    config: options.config,
    safetyMode: getInteractiveSafetyMode(options.config),
    mode: options.mode,
    briefing: options.briefing,
  });
  return {
    systemPrompt,
    initialMessages: options.initialMessages,
    injectSessionStateOnFirstTurn:
      Boolean(options.briefing)
      || (options.initialMessages?.length ?? 0) > 0
      || options.sessionState.hasContent(),
  };
}

function setupSummarizeCallback(contextManager: ContextManager, provider: LLMProvider, sessionState?: SessionState): void {
  contextManager.setSummarizeCallback(async (messages) => {
    const stateContext = buildCompactionStateContext(sessionState);
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
      { role: MessageRole.USER, content: "Summarize the conversation above into a concise handoff summary. Preserve all file paths and line numbers." },
    ];
    let summary = "";
    const stream = provider.chat(summaryPrompt, []);
    for await (const chunk of stream) {
      if (chunk.type === "text") summary += chunk.content;
    }
    return summary || "No summary available.";
  });
}

function buildCompactionStateContext(sessionState?: SessionState): string {
  if (!sessionState) return "";
  const stateSnapshot = sessionState.toSystemMessage("compact");
  return stateSnapshot
    ? `\n\nThe following session state has already been saved and will persist across compaction:\n${stateSnapshot}\n\nDo NOT repeat raw diff content or file contents that are already captured in findings or tool summaries above.`
    : "";
}

function renderStdoutText(text: string): string {
  return process.stdout.isTTY ? renderMarkdown(text) : text;
}

function writeRenderedStdout(text: string): void {
  process.stdout.write(renderStdoutText(text) + "\n");
}

function flushPrimaryOutput(result: TaskLoopResult, verbosity: Verbosity): void {
  if (outputState.textBuffer.trim()) {
    if (outputState.hadToolCalls) process.stderr.write("\n");
    writeRenderedStdout(outputState.textBuffer);
    return;
  }
  if (!result.lastText?.trim()) return;
  if (verbosity !== "quiet") process.stderr.write(yellow("[warning] No final response. Showing last output from agent:") + "\n");
  writeRenderedStdout(result.lastText);
}

function flushStatusOutput(result: TaskLoopResult, verbosity: Verbosity): void {
  if (verbosity === "quiet") return;
  const messages: Partial<Record<TaskLoopResult["status"], string>> = {
    budget_exceeded: yellow("[warning] Iteration limit reached — partial results shown."),
    aborted: dim("[info] Agent was interrupted."),
  };
  if (result.status === "empty_response" && !outputState.textBuffer.trim() && !result.lastText?.trim()) {
    process.stderr.write(yellow("[warning] Agent completed but produced no output.") + "\n");
    return;
  }
  const message = messages[result.status];
  if (message) process.stderr.write(message + "\n");
}

function flushOutput(result: TaskLoopResult, verbosity: Verbosity): void {
  flushPrimaryOutput(result, verbosity);
  flushStatusOutput(result, verbosity);
}

function createMidpointCallback(opts: {
  provider: LLMProvider;
  mode: TaskMode;
  repoRoot: string;
  skills: SkillRegistry;
  toolRegistry: ToolRegistry;
  config: DevAgentConfig;
  getTurnNumber: () => number;
}): MidpointCallback {
  return async (messages) => {
    const midBriefing = await synthesizeBriefing(messages, opts.getTurnNumber(), {
      strategy: "auto",
      provider: opts.provider,
    });
    return {
      continueMessages: [
        { role: MessageRole.SYSTEM, content: buildRunSystemPrompt({ ...opts, briefing: midBriefing }) },
        { role: MessageRole.USER, content: findLastUserContent(messages) },
      ],
    };
  };
}

const REVIEW_PATTERN = /\b(code\s+)?review\b/i;
const DIFF_PATTERN = /\b(uncommitted|diff|changes|staged|unstaged)\b/i;

export function isReviewQuery(query: string): "staged" | "unstaged" | false {
  if (!REVIEW_PATTERN.test(query) || !DIFF_PATTERN.test(query)) return false;
  return /\bstaged\b/i.test(query) ? "staged" : "unstaged";
}

function runGitTextCommand(command: string, repoRoot: string): string | null {
  try {
    return execSync(command, { cwd: repoRoot, encoding: "utf-8", maxBuffer: 1024 * 1024 * 5, timeout: 10_000 });
  } catch {
    return null;
  }
}

function shellEscapeArg(value: string): string {
  return `'${value.replace(/'/g, `'\\''`)}'`;
}

function buildDiffScopeSuffix(pathFilters: ReadonlyArray<string>): string {
  return pathFilters.length === 0 ? "" : ` -- ${pathFilters.map(shellEscapeArg).join(" ")}`;
}

function loadLocalDiffTarget(
  repoRoot: string,
  target: ResolvedPromptCommandTarget,
  pathFilters: ReadonlyArray<string> = [],
): string | null {
  const suffix = buildDiffScopeSuffix(pathFilters);
  const command = target.kind === "unstaged"
    ? `git diff${suffix}`
    : target.kind === "staged"
      ? `git diff --cached${suffix}`
      : target.kind === "last-commit"
        ? `git show --format=medium --patch --no-ext-diff HEAD${suffix}`
        : `git show --format=medium --patch --no-ext-diff ${shellEscapeArg(target.ref)}${suffix}`;
  const output = runGitTextCommand(command, repoRoot);
  return output?.trim() ? truncateToolOutput(output) : null;
}

export function resolveAutoPromptCommandTarget(
  repoRoot: string,
  pathFilters: ReadonlyArray<string> = [],
  runGit: (command: string, repoRoot: string) => string | null = runGitTextCommand,
): ResolvedPromptCommandTarget {
  const suffix = buildDiffScopeSuffix(pathFilters);
  if (runGit(`git diff --name-only${suffix}`, repoRoot)?.trim()) return { kind: "unstaged" };
  if (runGit(`git diff --cached --name-only${suffix}`, repoRoot)?.trim()) return { kind: "staged" };
  return pathFilters.length > 0 ? { kind: "unstaged" } : { kind: "last-commit" };
}

async function maybePreloadReviewDiff(query: string, repoRoot: string): Promise<string | null> {
  const reviewType = isReviewQuery(query);
  if (!reviewType) return null;
  const diffOutput = loadLocalDiffTarget(repoRoot, { kind: reviewType });
  return diffOutput ? formatPreloadedDiffMessage({ kind: reviewType }, diffOutput) : null;
}

async function prepareQueryForExecution(query: string, repoRoot: string): Promise<{
  readonly query: string;
  readonly prependedMessages: ReadonlyArray<Message>;
  readonly finalTextValidator?: (candidate: string) => { readonly valid: boolean; readonly retryMessage?: string };
}> {
  const prepared = await preparePromptCommandQuery(query, {
    resolveAutoTarget: async (pathFilters) => resolveAutoPromptCommandTarget(repoRoot, pathFilters),
    loadDiff: async (target, pathFilters) => loadLocalDiffTarget(repoRoot, target, pathFilters),
  });
  if (prepared) {
    return {
      query: prepared.rewrittenQuery,
      prependedMessages: prepared.preloadedDiffs.map((entry) => ({ role: MessageRole.USER, content: entry.content, pinned: true })),
      finalTextValidator: prepared.finalTextValidator,
    };
  }
  const preloadedDiff = await maybePreloadReviewDiff(query, repoRoot);
  return {
    query,
    prependedMessages: preloadedDiff ? [{ role: MessageRole.USER, content: preloadedDiff, pinned: true }] : [],
    finalTextValidator: undefined,
  };
}

function buildRunSystemPrompt(options: Pick<RunOptions, "mode" | "repoRoot" | "skills" | "toolRegistry" | "config" | "briefing">): string {
  return assembleSystemPrompt({
    mode: options.mode,
    repoRoot: options.repoRoot,
    skills: options.skills,
    availableTools: options.toolRegistry.getLoaded(),
    deferredTools: options.toolRegistry.getDeferred(),
    approvalMode: getInteractiveSafetyMode(options.config),
    provider: options.config.provider,
    model: options.config.model,
    agentModelOverrides: options.config.agentModelOverrides,
    agentReasoningOverrides: options.config.agentReasoningOverrides,
    briefing: options.briefing,
  });
}

async function executeSingleQueryLoop(options: RunSingleQueryOptions, systemPrompt: string): Promise<TaskLoopResult> {
  const preparedQuery = await prepareQueryForExecution(options.query, options.repoRoot);
  const loop = new TaskLoop({
    provider: options.provider,
    tools: options.toolRegistry,
    bus: options.bus,
    approvalGate: options.gate,
    config: options.config,
    systemPrompt,
    repoRoot: options.repoRoot,
    mode: options.mode,
    contextManager: options.contextManager,
    doubleCheck: options.doubleCheck,
    initialMessages: options.initialMessages,
    sessionState: options.sessionState,
    midpointCallback: createMidpointCallback({
      provider: options.provider,
      mode: options.mode,
      repoRoot: options.repoRoot,
      skills: options.skills,
      toolRegistry: options.toolRegistry,
      config: options.config,
      getTurnNumber: () => 0,
    }),
  });
  return loop.run(preparedQuery.query, {
    prependedMessages: preparedQuery.prependedMessages,
    finalTextValidator: preparedQuery.finalTextValidator,
  });
}

function appendBudgetNotice(result: TaskLoopResult, verbosity: Verbosity): void {
  if (result.status !== "budget_exceeded") return;
  const budgetPart = makeInfoPart("status", ["Iteration limit exhausted. Type /continue to proceed."]);
  transcriptComposer.appendPart(nextTranscriptId("budget"), budgetPart);
  if (verbosity !== "quiet") {
    const rendered = formatTranscriptPart(budgetPart);
    if (rendered) process.stderr.write(rendered + "\n");
  }
}

function completeSingleQueryTurn(result: TaskLoopResult, startTime: number, verbosity: Verbosity): void {
  const turnSummaryPart = makeTurnSummaryPart({
    iterations: result.iterations,
    toolCalls: outputState.turnToolCallCount,
    cost: outputState.turnCostDelta,
    elapsedMs: Date.now() - startTime,
  });
  transcriptComposer.completeTurn(nextTranscriptId("summary"), turnSummaryPart, {
    status: result.status === "budget_exceeded" ? "budget_exceeded" : "completed",
    finishedAt: Date.now(),
  });
  writeCompletedTurnSummary(result, startTime, verbosity);
}

function writeCompletedTurnSummary(result: TaskLoopResult, startTime: number, verbosity: Verbosity): void {
  if (verbosity === "quiet") return;
  const completedTurnNode = transcriptComposer.getNodes().at(-1);
  const completedTurn = completedTurnNode?.kind === "turn" ? completedTurnNode.turn : null;
  process.stderr.write((completedTurn
    ? formatTurnEnd(completedTurn)
    : formatTurnSummary({
        iterationCount: result.iterations,
        toolCallCount: outputState.turnToolCallCount,
        inputTokens: outputState.turnInputTokens,
        costDelta: outputState.turnCostDelta,
        elapsedMs: Date.now() - startTime,
      })) + "\n");
  if (Date.now() - startTime > 30_000) terminalBell();
}

export async function runSingleQuery(options: RunSingleQueryOptions): Promise<TaskLoopRunResult> {
  setupSummarizeCallback(options.contextManager, options.provider, options.sessionState);
  outputState.resetTurn();
  const startTime = Date.now();
  const result = await executeSingleQueryLoop(options, buildRunSystemPrompt(options));
  spinner.stop();
  flushOutput(result, options.verbosity);
  if (result.lastText) transcriptComposer.appendPart(nextTranscriptId("final"), makeFinalOutputPart(result.lastText));
  appendBudgetNotice(result, options.verbosity);
  completeSingleQueryTurn(result, startTime, options.verbosity);
  return result;
}

let tuiLoop: InstanceType<typeof TaskLoop> | null = null;

export function resetTuiLoop(): void {
  tuiLoop = null;
}

export function abortTuiQuery(): void {
  tuiLoop?.abort();
}

export function updateTuiSystemPrompt(prompt: string): void {
  if (tuiLoop) tuiLoop.updateSystemPrompt(prompt);
}

export async function runTuiQuery(options: RunSingleQueryOptions): Promise<TaskLoopRunResult> {
  if (!tuiLoop) {
    const initialLoopOptions = createInitialTuiLoopOptions(options);
    setupSummarizeCallback(options.contextManager, options.provider, options.sessionState);
    tuiLoop = new TaskLoop({
      provider: options.provider,
      tools: options.toolRegistry,
      bus: options.bus,
      approvalGate: options.gate,
      config: options.config,
      systemPrompt: initialLoopOptions.systemPrompt,
      repoRoot: options.repoRoot,
      mode: options.mode,
      contextManager: options.contextManager,
      doubleCheck: options.doubleCheck,
      initialMessages: initialLoopOptions.initialMessages,
      sessionState: options.sessionState,
      injectSessionStateOnFirstTurn: initialLoopOptions.injectSessionStateOnFirstTurn,
    });
  } else {
    tuiLoop.resetIterations();
  }
  const preparedQuery = await prepareQueryForExecution(options.query, options.repoRoot);
  return tuiLoop.run(preparedQuery.query, {
    prependedMessages: preparedQuery.prependedMessages,
    finalTextValidator: preparedQuery.finalTextValidator,
  });
}

export function renderStdoutForSingleShot(text: string): string {
  return renderStdoutText(text);
}
