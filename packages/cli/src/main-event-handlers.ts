import {
  SafetyMode,
  loggedSubagentRunFromEvent,
} from "@devagent/runtime";

import {
  SubagentPanelRenderer,
  buildVerbosityConfig,
  dim,
  formatContextGauge,
  formatEnrichedError,
  formatError,
  formatPlan,
  formatSubagentBatchLaunch,
  formatSubagentError,
  formatSubagentStart,
  formatToolGroupEnd,
  formatToolGroupStart,
  formatToolStart,
  formatTranscriptPart,
  formatTurnStart,
  inferErrorSuggestion,
  isCategoryEnabled,
  setTerminalTitle,
  summarizeSubagentUpdate,
  summarizeToolParams,
  yellow,
} from "./format.js";
import { registerApprovalEvents } from "./main-approval-events.js";
import { nextTranscriptId, spinner, transcriptComposer } from "./main-state.js";
import type { OutputState, SubagentDisplayState } from "./output-state.js";
import { StatusLine } from "./status-line.js";
import {
  presentContextCompactedEvent,
  presentContextCompactingEvent,
  presentSummaryToolMessage,
  presentToolAfterEvent,
  presentToolBeforeEvent,
} from "./transcript-presenter.js";
import type {
  DevAgentConfig,
  EventBus,
  EventMap,
  VerbosityConfig,
} from "@devagent/runtime";

type Verbosity = "quiet" | "normal" | "verbose";

interface EventHandlerContext {
  readonly bus: EventBus;
  readonly config: DevAgentConfig;
  readonly maxIter: number;
  readonly os: OutputState;
  readonly statusLine: StatusLine | null;
  readonly subagentRenderer: SubagentPanelRenderer;
  readonly toolParamsCache: Map<string, Record<string, unknown>>;
  readonly ungroupedToolNames: Set<string>;
  readonly vc: VerbosityConfig;
  readonly verbosity: Verbosity;
}

interface RecentToolResult {
  readonly durationMs: number; readonly name: string; readonly success: boolean;
}

interface EventHandlerSetupResult {
  readonly lspToolCounts: Map<string, number>; readonly statusLine: StatusLine | null;
}

export function setupEventHandlers(
  bus: EventBus,
  config: DevAgentConfig,
  verbosity: Verbosity,
  verbosityConfig: VerbosityConfig | undefined,
  os: OutputState,
): EventHandlerSetupResult {
  const vc = verbosityConfig ?? buildVerbosityConfig(verbosity, undefined);
  const statusLine = createStatusLine(config, verbosity);
  const ctx: EventHandlerContext = {
    bus,
    config,
    maxIter: config.budget.maxIterations,
    os,
    statusLine,
    subagentRenderer: new SubagentPanelRenderer(verbosity !== "quiet" || isCategoryEnabled("tools", vc)),
    toolParamsCache: new Map(),
    ungroupedToolNames: new Set(["write_file", "replace_in_file"]),
    vc,
    verbosity,
  };

  registerIterationEvents(ctx);
  registerToolEvents(ctx);
  const lspToolCounts = registerLspToolTracking(bus);
  registerPlanEvents(ctx);
  registerSubagentEvents(ctx);
  registerMessageEvents(ctx);
  registerContextEvents(ctx);
  registerCostEvents(ctx);
  registerSessionTrackingEvents(ctx);
  registerErrorEvents(ctx);
  registerApprovalEvents(ctx.bus, statusLine, (line) => writeUiLine(ctx, line));
  bus.on("session:end", () => statusLine?.clear());

  return { lspToolCounts, statusLine };
}

function createStatusLine(config: DevAgentConfig, verbosity: Verbosity): StatusLine | null {
  return verbosity === "quiet"
    ? null
    : new StatusLine(config.model, getInteractiveSafetyMode(config));
}

function getInteractiveSafetyMode(config: DevAgentConfig): SafetyMode {
  return config.approval.mode === SafetyMode.AUTOPILOT
    ? SafetyMode.AUTOPILOT
    : SafetyMode.DEFAULT;
}

function hasRunningSubagents(ctx: EventHandlerContext): boolean {
  for (const panel of ctx.os.subagentDisplay.values()) {
    if (panel.status === "running") return true;
  }
  return false;
}

function withSubagentPanelsHidden(ctx: EventHandlerContext, action: () => void): void {
  if (!ctx.subagentRenderer.active) {
    action();
    return;
  }
  ctx.subagentRenderer.suspend();
  action();
  if (hasRunningSubagents(ctx)) {
    ctx.subagentRenderer.resume();
    ctx.subagentRenderer.setPanels([...ctx.os.subagentDisplay.values()]);
    return;
  }
  ctx.subagentRenderer.setPanels([]);
}

function writeUi(ctx: EventHandlerContext, text: string): void {
  withSubagentPanelsHidden(ctx, () => {
    process.stderr.write(text);
  });
}

function writeUiLine(ctx: EventHandlerContext, line: string): void {
  writeUi(ctx, line + "\n");
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

function flushToolGroup(ctx: EventHandlerContext): void {
  const group = ctx.os.pendingToolGroup;
  if (!group || group.count <= 1) {
    ctx.os.pendingToolGroup = null;
    return;
  }
  ctx.os.pendingToolGroup = null;

  withSubagentPanelsHidden(ctx, () => {
    spinner.log(formatToolGroupStart({
      name: group.name,
      count: group.count,
      paramSummaries: group.params,
      iteration: group.iteration,
      maxIter: group.maxIter,
    }));
    spinner.log(formatToolGroupEnd(group.name, group.count, group.lastSuccess, group.totalDurationMs, group.lastError));
  });
}

function shouldGroupTool(ctx: EventHandlerContext, name: string): boolean {
  return !ctx.ungroupedToolNames.has(name);
}

function registerIterationEvents(ctx: EventHandlerContext): void {
  ctx.bus.on("iteration:start", (event) => {
    if (event.agentId) return;
    ctx.os.currentIteration = event.iteration;
    ctx.os.currentTokens = event.estimatedTokens;
    ctx.os.maxContextTokens = event.maxContextTokens;
    updateStatusLineForIteration(ctx, event);
    setTerminalTitle(`devagent: ${formatIterationLabel(event)}`);
  });
}

function updateStatusLineForIteration(ctx: EventHandlerContext, event: EventMap["iteration:start"]): void {
  if (!ctx.statusLine) return;
  ctx.statusLine.update({
    iteration: event.iteration,
    maxIterations: event.maxIterations,
    inputTokens: event.estimatedTokens,
    maxContextTokens: event.maxContextTokens,
  });
  spinner.updateSuffix(ctx.statusLine.formatSpinnerSuffix());
}

function formatIterationLabel(event: EventMap["iteration:start"]): string {
  return event.maxIterations > 0 ? `iter ${event.iteration}/${event.maxIterations}` : `iter ${event.iteration}`;
}

function registerToolEvents(ctx: EventHandlerContext): void {
  ctx.bus.on("tool:before", (event) => handleToolBefore(ctx, event));
  ctx.bus.on("tool:after", (event) => handleToolAfter(ctx, event));
}

function handleToolBefore(ctx: EventHandlerContext, event: EventMap["tool:before"]): void {
  if (event.name.startsWith("audit:")) return;
  spinner.stop();
  flushBufferedThinking(ctx);
  recordToolStart(ctx, event);
  if (recordSubagentToolCall(ctx, event)) return;
  if (shouldSkipRootToolStart(ctx, event.name)) return;

  const gauge = formatToolGauge(ctx);
  if (isCategoryEnabled("tools", ctx.vc)) {
    renderVerboseToolStart(ctx, event, gauge);
    return;
  }

  renderGroupedToolStart(ctx, event, gauge);
}

function flushBufferedThinking(ctx: EventHandlerContext): void {
  if (!ctx.os.textBuffer.trim() || ctx.verbosity === "quiet") {
    ctx.os.textBuffer = "";
    return;
  }
  for (const line of ctx.os.textBuffer.trim().split(/\n/).filter((entry) => entry.trim()).slice(0, 3)) {
    spinner.log(dim(`  ℹ ${truncateLine(line, 120)}`));
  }
  logThinkingDuration(ctx);
  ctx.os.textBuffer = "";
}

function truncateLine(line: string, maxLength: number): string {
  return line.length > maxLength ? line.slice(0, maxLength - 3) + "..." : line;
}

function logThinkingDuration(ctx: EventHandlerContext): void {
  if (ctx.os.thinkingStartMs === null) return;
  const durationMs = Date.now() - ctx.os.thinkingStartMs;
  if (durationMs > 500) {
    spinner.log(dim(`  ℹ Thought for ${(durationMs / 1000).toFixed(1)}s`));
  }
  ctx.os.thinkingStartMs = null;
}

function recordToolStart(ctx: EventHandlerContext, event: EventMap["tool:before"]): void {
  ctx.os.hadToolCalls = true;
  ctx.os.turnToolCallCount++;
  ctx.toolParamsCache.set(`${event.agentId ?? "root"}:${event.callId}`, event.params);
}

function recordSubagentToolCall(ctx: EventHandlerContext, event: EventMap["tool:before"]): boolean {
  if (!event.agentId) return false;
  const existing = ctx.os.sessionSubagents.get(event.agentId);
  if (existing) {
    ctx.os.sessionSubagents.set(event.agentId, {
      ...existing,
      toolCalls: existing.toolCalls + 1,
    });
  }
  return true;
}

function shouldSkipRootToolStart(ctx: EventHandlerContext, name: string): boolean {
  if (name === "delegate") return true;
  return ctx.verbosity === "quiet" && !isCategoryEnabled("tools", ctx.vc);
}

function formatToolGauge(ctx: EventHandlerContext): string | undefined {
  return isCategoryEnabled("context", ctx.vc) ? formatContextGauge(ctx.os.currentTokens, ctx.os.maxContextTokens) : undefined;
}

function renderVerboseToolStart(
  ctx: EventHandlerContext,
  event: EventMap["tool:before"],
  gauge: string | undefined,
): void {
  flushToolGroup(ctx);
  transcriptComposer.appendPart(nextTranscriptId("tool"), presentToolBeforeEvent(event, ctx.os.currentIteration, ctx.maxIter));
  writeUiLine(ctx, formatToolStart({
    name: event.name,
    params: event.params,
    iteration: ctx.os.currentIteration,
    maxIter: ctx.maxIter,
    gauge,
  }));
  writeUiLine(ctx, dim(`  params: ${JSON.stringify(event.params, null, 2)}`));
}

function renderGroupedToolStart(
  ctx: EventHandlerContext,
  event: EventMap["tool:before"],
  gauge: string | undefined,
): void {
  const summary = summarizeToolParams(event.name, event.params);
  if (appendToPendingToolGroup(ctx, event, summary)) return;
  flushToolGroup(ctx);
  startPendingToolGroup(ctx, event, summary);
  const presented = presentToolBeforeEvent(event, ctx.os.currentIteration, ctx.maxIter);
  transcriptComposer.appendPart(nextTranscriptId("tool"), presented);
  writeUiLine(ctx, formatTranscriptPart(presented) ?? formatToolStart({
    name: event.name,
    params: event.params,
    iteration: ctx.os.currentIteration,
    maxIter: ctx.maxIter,
    gauge,
  }));
}

function appendToPendingToolGroup(
  ctx: EventHandlerContext,
  event: EventMap["tool:before"],
  summary: string | null,
): boolean {
  const group = ctx.os.pendingToolGroup;
  if (!shouldGroupTool(ctx, event.name) || !group || group.name !== event.name) return false;
  group.count++;
  if (summary) group.params.push(summary);
  return true;
}

function startPendingToolGroup(
  ctx: EventHandlerContext,
  event: EventMap["tool:before"],
  summary: string | null,
): void {
  if (!shouldGroupTool(ctx, event.name)) return;
  ctx.os.pendingToolGroup = {
    name: event.name,
    count: 1,
    params: summary ? [summary] : [],
    totalDurationMs: 0,
    lastSuccess: true,
    lastError: undefined,
    iteration: ctx.os.currentIteration,
    maxIter: ctx.maxIter,
  };
}

function handleToolAfter(ctx: EventHandlerContext, event: EventMap["tool:after"]): void {
  if (event.name.startsWith("audit:")) return;
  spinner.stop();
  ctx.toolParamsCache.delete(`${event.agentId ?? "root"}:${event.callId}`);
  if (event.agentId) return;
  if (maybeHandleDelegateCompletion(ctx, event)) return;
  if (ctx.verbosity === "quiet" && !isCategoryEnabled("tools", ctx.vc) && event.result.success) return;

  if (!appendToolResultToGroup(ctx, event)) {
    renderToolResultParts(ctx, event);
  }
  renderVerboseToolOutput(ctx, event);
  restartSpinnerIfIdle(ctx);
}

function maybeHandleDelegateCompletion(ctx: EventHandlerContext, event: EventMap["tool:after"]): boolean {
  if (event.name !== "delegate" || !event.result.metadata?.["agentMeta"]) return false;
  if (ctx.verbosity !== "quiet" && !hasRunningSubagents(ctx)) {
    spinner.start();
  }
  return true;
}

function appendToolResultToGroup(ctx: EventHandlerContext, event: EventMap["tool:after"]): boolean {
  const group = ctx.os.pendingToolGroup;
  if (!group || group.name !== event.name || !shouldGroupTool(ctx, event.name) || isCategoryEnabled("tools", ctx.vc)) {
    return false;
  }
  group.totalDurationMs += event.durationMs;
  if (!event.result.success) {
    group.lastSuccess = false;
    group.lastError = event.result.error ?? undefined;
  }
  if (group.count === 1) {
    renderToolResultParts(ctx, event);
  }
  return true;
}

function renderToolResultParts(ctx: EventHandlerContext, event: EventMap["tool:after"]): void {
  const presentedParts = presentToolAfterEvent(event, ctx.os.currentIteration, ctx.maxIter);
  for (const part of presentedParts) {
    transcriptComposer.appendPart(nextTranscriptId("tool"), part);
  }
  for (const part of presentedParts) {
    const rendered = formatTranscriptPart(part);
    if (rendered) writeUiLine(ctx, rendered);
  }
}

function renderVerboseToolOutput(ctx: EventHandlerContext, event: EventMap["tool:after"]): void {
  if (!isCategoryEnabled("tools", ctx.vc) || !event.result.output) return;
  const output = event.result.output.length > 500 ? event.result.output.substring(0, 500) + "…" : event.result.output;
  writeUiLine(ctx, dim(`  output: ${output}`));
}

function restartSpinnerIfIdle(ctx: EventHandlerContext): void {
  if (ctx.verbosity !== "quiet" && !hasRunningSubagents(ctx)) {
    spinner.start();
  }
}

function registerLspToolTracking(bus: EventBus): Map<string, number> {
  const lspToolNames = new Set(["diagnostics", "definitions", "references", "symbols", "definition_by_name", "references_by_name"]);
  const lspToolCounts = new Map<string, number>();
  bus.on("tool:after", (event) => {
    if (lspToolNames.has(event.name)) {
      lspToolCounts.set(event.name, (lspToolCounts.get(event.name) ?? 0) + 1);
    }
  });
  return lspToolCounts;
}

function registerPlanEvents(ctx: EventHandlerContext): void {
  ctx.bus.on("plan:updated", (event) => {
    if (ctx.verbosity === "quiet" && !isCategoryEnabled("plan", ctx.vc)) return;
    spinner.stop();
    transcriptComposer.appendPart(nextTranscriptId("plan"), { kind: "plan", data: event.steps as Array<{ description: string; status: string }> });
    writeUi(ctx, "\n" + dim("── Plan ──") + "\n");
    writeUi(ctx, formatPlan(event.steps) + "\n\n");
  });
}

function registerSubagentEvents(ctx: EventHandlerContext): void {
  ctx.bus.on("subagent:start", (event) => handleSubagentStart(ctx, event));
  ctx.bus.on("subagent:update", (event) => handleSubagentUpdate(ctx, event));
  ctx.bus.on("subagent:end", (event) => handleSubagentEnd(ctx, event));
  ctx.bus.on("subagent:error", (event) => handleSubagentError(ctx, event));
}

function handleSubagentStart(ctx: EventHandlerContext, event: EventMap["subagent:start"]): void {
  spinner.stop();
  flushToolGroup(ctx);
  ctx.os.sessionSubagents.set(event.agentId, loggedSubagentRunFromEvent(event, ctx.os.sessionSubagents.get(event.agentId)));
  ctx.os.subagentDisplay.set(event.agentId, {
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
  renderSubagentStart(ctx, event);
}

function renderSubagentStart(ctx: EventHandlerContext, event: EventMap["subagent:start"]): void {
  if (ctx.verbosity === "quiet" && !isCategoryEnabled("tools", ctx.vc)) return;
  if (event.batchId && (event.batchSize ?? 0) > 1 && !ctx.os.announcedSubagentBatches.has(event.batchId)) {
    ctx.os.announcedSubagentBatches.add(event.batchId);
    writeUiLine(ctx, formatSubagentBatchLaunch(event.agentType, event.batchSize ?? 0));
  }
  writeUiLine(ctx, formatSubagentStart(event));
  if (ctx.subagentRenderer.active) {
    ctx.subagentRenderer.setPanels([...ctx.os.subagentDisplay.values()]);
  }
}

function handleSubagentUpdate(ctx: EventHandlerContext, event: EventMap["subagent:update"]): void {
  const existing = ctx.os.subagentDisplay.get(event.agentId);
  if (!existing) return;
  const summary = summarizeSubagentUpdate(event);
  ctx.os.subagentDisplay.set(event.agentId, pushSubagentActivity({
    ...existing,
    currentIteration: event.iteration ?? existing.currentIteration,
  }, summary));
  renderSubagentUpdate(ctx, event, summary);
}

function renderSubagentUpdate(ctx: EventHandlerContext, event: EventMap["subagent:update"], summary: string): void {
  if (ctx.verbosity === "quiet" && !isCategoryEnabled("tools", ctx.vc)) return;
  if (ctx.subagentRenderer.active) {
    ctx.subagentRenderer.setPanels([...ctx.os.subagentDisplay.values()]);
    return;
  }
  if (event.milestone === "iteration:start") {
    writeUiLine(ctx, `  ${dim(`[${event.agentId}:${event.iteration ?? 0}]`)} ${summary}`);
  }
}

function handleSubagentEnd(ctx: EventHandlerContext, event: EventMap["subagent:end"]): void {
  spinner.stop();
  ctx.os.sessionSubagents.set(event.agentId, loggedSubagentRunFromEvent(event, ctx.os.sessionSubagents.get(event.agentId)));
  const display = ctx.os.subagentDisplay.get(event.agentId);
  if (display) {
    ctx.os.subagentDisplay.set(event.agentId, pushSubagentActivity({
      ...display,
      status: "completed",
      durationMs: event.durationMs,
      currentIteration: event.iterations,
      quality: event.quality ? { score: event.quality.score, completeness: event.quality.completeness } : undefined,
    }, `Completed after ${event.iterations} iterations`));
  }
  renderSubagentTerminalState(ctx);
}

function handleSubagentError(ctx: EventHandlerContext, event: EventMap["subagent:error"]): void {
  spinner.stop();
  ctx.os.sessionSubagents.set(event.agentId, loggedSubagentRunFromEvent(event, ctx.os.sessionSubagents.get(event.agentId)));
  const display = ctx.os.subagentDisplay.get(event.agentId);
  if (display) {
    ctx.os.subagentDisplay.set(event.agentId, pushSubagentActivity({
      ...display,
      status: "error",
      durationMs: event.durationMs,
    }, `Failed: ${event.error}`));
  }
  renderSubagentErrorState(ctx, event);
}

function renderSubagentTerminalState(ctx: EventHandlerContext): void {
  if (ctx.verbosity === "quiet" && !isCategoryEnabled("tools", ctx.vc)) return;
  if (ctx.subagentRenderer.active) {
    ctx.subagentRenderer.setPanels([...ctx.os.subagentDisplay.values()]);
  }
  restartSpinnerIfIdle(ctx);
}

function renderSubagentErrorState(ctx: EventHandlerContext, event: EventMap["subagent:error"]): void {
  if (ctx.verbosity === "quiet" && !isCategoryEnabled("tools", ctx.vc)) return;
  if (ctx.subagentRenderer.active) {
    ctx.subagentRenderer.setPanels([...ctx.os.subagentDisplay.values()]);
  } else {
    writeUiLine(ctx, formatSubagentError(event));
  }
  if (ctx.verbosity !== "quiet") {
    spinner.start();
  }
}

function registerMessageEvents(ctx: EventHandlerContext): void {
  ctx.bus.on("message:tool", (event) => handleToolMessage(ctx, event));
  ctx.bus.on("message:user", (event) => handleUserMessage(ctx, event));
  ctx.bus.on("message:assistant", (event) => handleAssistantMessage(ctx, event));
}

function handleToolMessage(ctx: EventHandlerContext, event: EventMap["message:tool"]): void {
  if (event.agentId) return;
  if (event.toolName !== "delegate" || event.summaryOnly !== true) return;
  if (ctx.verbosity === "quiet" && !isCategoryEnabled("tools", ctx.vc)) return;
  const part = presentSummaryToolMessage(event);
  transcriptComposer.appendPart(nextTranscriptId("delegate"), part);
  const rendered = formatTranscriptPart(part);
  if (rendered) writeUiLine(ctx, rendered);
}

function handleUserMessage(ctx: EventHandlerContext, event: EventMap["message:user"]): void {
  if (ctx.verbosity === "quiet") return;
  spinner.start();
  transcriptComposer.startTurn(nextTranscriptId("turn"), event.content, Date.now());
  writeUiLine(ctx, formatTurnStart(event.content));
}

function handleAssistantMessage(ctx: EventHandlerContext, event: EventMap["message:assistant"]): void {
  if (event.agentId) return;
  if (!event.partial) {
    flushToolGroup(ctx);
    return;
  }
  spinner.stop();
  ctx.os.textBuffer += event.content;
  if (event.chunk?.type === "thinking" && ctx.os.thinkingStartMs === null) {
    ctx.os.thinkingStartMs = Date.now();
  }
}

function registerContextEvents(ctx: EventHandlerContext): void {
  ctx.bus.on("context:compacting", (event) => {
    spinner.stop();
    if (ctx.verbosity === "quiet" && !isCategoryEnabled("context", ctx.vc)) return;
    const part = presentContextCompactingEvent(event);
    transcriptComposer.appendPart(nextTranscriptId("context"), part);
    const rendered = formatTranscriptPart(part);
    if (rendered) writeUiLine(ctx, rendered);
    spinner.start("Compacting context…");
  });
  ctx.bus.on("context:compacted", (event) => {
    spinner.stop();
    if (ctx.verbosity === "quiet" && !isCategoryEnabled("context", ctx.vc)) return;
    const part = presentContextCompactedEvent(event);
    transcriptComposer.appendPart(nextTranscriptId("context"), part);
    const rendered = formatTranscriptPart(part);
    if (rendered) writeUiLine(ctx, rendered);
  });
}

function registerCostEvents(ctx: EventHandlerContext): void {
  let costWarningFired = false;
  const costThreshold = ctx.config.budget.costWarningThreshold;
  ctx.bus.on("cost:update", (event) => {
    ctx.os.turnInputTokens += event.inputTokens;
    ctx.os.turnCostDelta += event.totalCost;
    ctx.os.sessionTotalInputTokens += event.inputTokens;
    ctx.os.sessionTotalOutputTokens += event.outputTokens;
    ctx.os.sessionTotalCost += event.totalCost;
    updateStatusLineForCost(ctx);
    if (costWarningFired || costThreshold <= 0 || ctx.os.sessionTotalCost < costThreshold) return;
    costWarningFired = true;
    writeUiLine(ctx, yellow(`[cost] Session cost $${ctx.os.sessionTotalCost.toFixed(4)} exceeds threshold $${costThreshold.toFixed(2)}. Use --max-iterations to limit.`));
  });
}

function updateStatusLineForCost(ctx: EventHandlerContext): void {
  if (!ctx.statusLine) return;
  ctx.statusLine.update({
    cost: ctx.os.sessionTotalCost,
    inputTokens: ctx.os.currentTokens || ctx.os.sessionTotalInputTokens,
  });
  spinner.updateSuffix(ctx.statusLine.formatSpinnerSuffix());
}

function registerSessionTrackingEvents(ctx: EventHandlerContext): void {
  ctx.bus.on("iteration:start", () => {
    ctx.os.sessionTotalIterations++;
  });
  ctx.bus.on("tool:before", (event) => {
    if (event.name.startsWith("audit:")) return;
    ctx.os.sessionTotalToolCalls++;
    ctx.os.sessionToolUsage.set(event.name, (ctx.os.sessionToolUsage.get(event.name) ?? 0) + 1);
  });
}

function registerErrorEvents(ctx: EventHandlerContext): void {
  const recentToolResults: RecentToolResult[] = [];
  ctx.bus.on("tool:after", (event) => {
    recentToolResults.push({ name: event.name, success: event.result.success, durationMs: event.durationMs });
    if (recentToolResults.length > 5) recentToolResults.shift();
  });
  ctx.bus.on("error", (event) => {
    spinner.stop();
    transcriptComposer.appendPart(nextTranscriptId("error"), { kind: "error", data: { message: event.message, code: event.code } });
    writeUiLine(ctx, formatRenderedError(event.message, recentToolResults));
  });
}

function formatRenderedError(message: string, recentToolResults: ReadonlyArray<RecentToolResult>): string {
  if (recentToolResults.length === 0) {
    return formatError(message);
  }
  return formatEnrichedError({
    message,
    recentTools: [...recentToolResults],
    suggestion: inferErrorSuggestion(message, recentToolResults),
  });
}
