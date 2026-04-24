import type { DoubleCheck, DoubleCheckResult } from "./double-check.js";
import { extractEnvFact } from "./session-state.js";
import type { SessionState } from "./session-state.js";
import { syncLSPAfterToolResult } from "./task-loop-lsp-sync.js";
import { normalizeRepoPath } from "./task-loop-paths.js";
import { formatToolSummary } from "./tool-summary-formatter.js";
import type {
  ApprovalGate,
  DevAgentConfig,
  EventBus,
  LSPDocumentSync,
  LLMProvider,
  Message,
  ToolContext,
  ToolResult,
  ToolSpec,
  ToolValidationResultMetadata,
} from "../core/index.js";
import { extractErrorMessage, extractToolFileChangePreviewSummary } from "../core/index.js";
import type { ToolRegistry } from "../tools/index.js";

interface PendingToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

interface NormalizedToolCall {
  readonly toolCall: PendingToolCall;
  readonly bypassResult: ToolResult | null;
}

interface ToolExecutionBatchContext {
  readonly batchId?: string;
  readonly batchSize?: number;
}

interface StreamResult {
  readonly textContent: string;
  readonly toolCalls: PendingToolCall[];
  readonly thinking: string;
}

interface ToolExecutionHost {
  readonly provider: LLMProvider;
  readonly tools: ToolRegistry;
  readonly bus: EventBus;
  readonly approvalGate: ApprovalGate;
  readonly config: DevAgentConfig;
  readonly repoRoot: string;
  readonly doubleCheck: DoubleCheck | null;
  readonly lspSync?: LSPDocumentSync | null;
  readonly sessionState: SessionState | null;
  readonly cachedPricing: {
    readonly inputPricePerMillion: number;
    readonly outputPricePerMillion: number;
  } | null;
  messages: Message[];
  totalCost: {
    readonly inputTokens: number;
    readonly outputTokens: number;
    readonly cacheReadTokens: number;
    readonly cacheWriteTokens: number;
    readonly totalCost: number;
  };
  lastReportedInputTokens: number;
  unresolvedDoubleCheckFailure: boolean;
  successfulReadonlyCallKeys: Set<string>;
  iterations: number;
  getMessagesForProvider(): ReadonlyArray<Message>;
  getAgentEventFields(): Pick<ToolContext, "agentId" | "parentAgentId" | "depth" | "agentType">;
  emitSubagentUpdate(event: Record<string, unknown>): void;
  normalizeToolCall(toolCall: PendingToolCall, category: ToolSpec["category"]): NormalizedToolCall;
  applyErrorGuidance(result: ToolResult, tool: ToolSpec): void;
  getSummaryTarget(toolName: string, args: Record<string, unknown>): string | null;
  captureReviewScopeFiles(toolCall: PendingToolCall, originalOutput: string): void;
  stagnationDetector: {
    recordToolResult(toolName: string, args: Record<string, unknown>, success: boolean): void;
  };
}

interface ExecutionState {
  readonly callId: string;
  readonly tool: ToolSpec;
  readonly effectiveCall: PendingToolCall;
  readonly batchContext: ToolExecutionBatchContext;
}

interface ExecutedToolResult {
  result: ToolResult;
  readonly originalOutput: string;
}

export async function streamTaskLoopLLMResponse(
  loop: ToolExecutionHost,
  tools: ReadonlyArray<ToolSpec>,
): Promise<StreamResult> {
  const state = { textContent: "", thinkingContent: "", toolCalls: [] as PendingToolCall[] };
  const stream = loop.provider.chat(loop.getMessagesForProvider(), tools);

  for await (const chunk of stream) {
    switch (chunk.type) {
      case "text":
        state.textContent += chunk.content;
        loop.bus.emit("message:assistant", {
          content: chunk.content,
          partial: true,
          chunk,
          ...loop.getAgentEventFields(),
        });
        break;
      case "thinking":
        state.thinkingContent += chunk.content;
        break;
      case "error":
        throw new Error(chunk.content);
      case "tool_call":
        state.toolCalls.push({
          name: chunk.toolName ?? "",
          arguments: parseToolArgs(chunk.content),
          callId: chunk.toolCallId ?? `call_${state.toolCalls.length}`,
        });
        break;
      case "done":
        applyUsage(loop, chunk.usage);
        break;
    }
  }

  return {
    textContent: state.textContent,
    toolCalls: state.toolCalls,
    thinking: state.thinkingContent,
  };
}

export async function executeTaskLoopToolCall(
  loop: ToolExecutionHost,
  toolCall: PendingToolCall,
  availableToolNames: ReadonlySet<string>,
  availableTools: ReadonlyArray<ToolSpec>,
  batchContext: ToolExecutionBatchContext = {},
): Promise<ToolResult> {
  const unavailable = getUnavailableToolResult(toolCall, availableToolNames, availableTools);
  if (unavailable) return unavailable;

  const state = createExecutionState(loop, toolCall, batchContext);
  if (state instanceof Promise) return state;

  emitToolBefore(loop, state);
  const approvalDenied = await getApprovalDeniedResult(loop, state);
  if (approvalDenied) return approvalDenied;

  const preEditBaseline = await capturePreEditBaseline(loop, state);
  const executed = await runToolHandler(loop, state, preEditBaseline);
  recordSessionState(loop, state, executed);
  recordSuccessfulReadonlyCall(loop, state, executed);
  return executed.result;
}

function parseToolArgs(content: string): Record<string, unknown> {
  try {
    const parsed = JSON.parse(content);
    if (parsed === null || typeof parsed !== "object" || Array.isArray(parsed)) {
      return { raw: content };
    }
    return parsed as Record<string, unknown>;
  } catch {
    return { _parseError: `Malformed tool arguments (invalid JSON): ${content.substring(0, 200)}` };
  }
}

function applyUsage(loop: ToolExecutionHost, usage: { promptTokens: number; completionTokens: number } | undefined): void {
  if (!usage) return;
  loop.lastReportedInputTokens = usage.promptTokens;
  const iterationCost = getIterationCost(loop, usage);
  loop.totalCost = {
    inputTokens: loop.totalCost.inputTokens + usage.promptTokens,
    outputTokens: loop.totalCost.outputTokens + usage.completionTokens,
    cacheReadTokens: loop.totalCost.cacheReadTokens,
    cacheWriteTokens: loop.totalCost.cacheWriteTokens,
    totalCost: loop.totalCost.totalCost + iterationCost,
  };
  loop.bus.emit("cost:update", {
    inputTokens: usage.promptTokens,
    outputTokens: usage.completionTokens,
    totalCost: iterationCost,
    model: loop.config.model,
    ...loop.getAgentEventFields(),
  });
}

function getIterationCost(
  loop: ToolExecutionHost,
  usage: { readonly promptTokens: number; readonly completionTokens: number },
): number {
  const pricing = loop.cachedPricing;
  if (!pricing) return 0;
  return (usage.promptTokens * pricing.inputPricePerMillion
    + usage.completionTokens * pricing.outputPricePerMillion) / 1_000_000;
}

function getUnavailableToolResult(
  toolCall: PendingToolCall,
  availableToolNames: ReadonlySet<string>,
  availableTools: ReadonlyArray<ToolSpec>,
): ToolResult | null {
  if (availableToolNames.has(toolCall.name)) return null;
  const namespaceHint = namespacedToolHint(toolCall.name, availableTools);
  return {
    success: false,
    output: "",
    error: namespaceHint
      ? `Unknown tool: ${toolCall.name}. ${namespaceHint}`
      : `Unknown tool: ${toolCall.name}`,
    artifacts: [],
  };
}

function createExecutionState(
  loop: ToolExecutionHost,
  toolCall: PendingToolCall,
  batchContext: ToolExecutionBatchContext,
): ExecutionState | Promise<ToolResult> {
  const tool = loop.tools.get(toolCall.name);
  const normalizedCall = loop.normalizeToolCall(toolCall, tool.category);
  if (normalizedCall.bypassResult) {
    emitToolAfter(loop, {
      callId: toolCall.callId,
      toolName: toolCall.name,
      result: normalizedCall.bypassResult,
      durationMs: 0,
    });
    return Promise.resolve(normalizedCall.bypassResult);
  }
  return {
    callId: toolCall.callId,
    tool,
    effectiveCall: normalizedCall.toolCall,
    batchContext,
  };
}

function emitToolBefore(loop: ToolExecutionHost, state: ExecutionState): void {
  loop.bus.emit("tool:before", {
    name: state.effectiveCall.name,
    params: state.effectiveCall.arguments,
    callId: state.callId,
    ...loop.getAgentEventFields(),
  });
  loop.emitSubagentUpdate({
    milestone: "tool:before",
    toolName: state.effectiveCall.name,
    toolCallId: state.callId,
    summary: `Running ${state.effectiveCall.name}`,
  });
}

async function getApprovalDeniedResult(
  loop: ToolExecutionHost,
  state: ExecutionState,
): Promise<ToolResult | null> {
  const approvalResult = await loop.approvalGate.check({
    toolName: state.effectiveCall.name,
    toolCategory: state.tool.category,
    filePath: (state.effectiveCall.arguments["path"] as string) ?? null,
    description: `${state.effectiveCall.name}: ${JSON.stringify(state.effectiveCall.arguments).substring(0, 200)}`,
    repoRoot: loop.repoRoot,
    arguments: state.effectiveCall.arguments,
  });
  if (approvalResult.approved) return null;
  const result: ToolResult = {
    success: false,
    output: "",
    error: `Tool execution denied: ${approvalResult.reason}`,
    artifacts: [],
  };
  emitToolAfter(loop, {
    callId: state.callId,
    toolName: state.effectiveCall.name,
    result,
    durationMs: 0,
  });
  return result;
}

async function capturePreEditBaseline(
  loop: ToolExecutionHost,
  state: ExecutionState,
): Promise<import("./double-check.js").DiagnosticBaseline | undefined> {
  if (state.tool.category === "mutating") loop.successfulReadonlyCallKeys.clear();
  if (state.tool.category !== "mutating" || !loop.doubleCheck?.isEnabled()) return undefined;
  const targetPath = state.effectiveCall.arguments["path"] as string | undefined;
  return targetPath ? loop.doubleCheck.captureBaseline([targetPath]) : undefined;
}

async function runToolHandler(
  loop: ToolExecutionHost,
  state: ExecutionState,
  preEditBaseline: import("./double-check.js").DiagnosticBaseline | undefined,
): Promise<ExecutedToolResult> {
  const startTime = Date.now();
  let result = await callToolHandler(loop, state);
  const durationMs = Date.now() - startTime;
  loop.applyErrorGuidance(result, state.tool);
  loop.stagnationDetector.recordToolResult(
    state.effectiveCall.name,
    state.effectiveCall.arguments,
    result.success,
  );

  const originalOutput = result.output;
  await syncLSPAfterToolResult(
    state.effectiveCall.name,
    state.effectiveCall.arguments,
    result,
    loop.lspSync,
  );
  result = await maybeRunDoubleCheck(loop, state, result, preEditBaseline);
  emitToolAfter(loop, {
    callId: state.callId,
    toolName: state.effectiveCall.name,
    result,
    durationMs,
    batchContext: state.batchContext,
  });
  return { result, originalOutput };
}

async function callToolHandler(
  loop: ToolExecutionHost,
  state: ExecutionState,
): Promise<ToolResult> {
  try {
    return await state.tool.handler(state.effectiveCall.arguments, {
      repoRoot: loop.repoRoot,
      config: loop.config,
      sessionId: "",
      callId: state.callId,
      batchId: state.batchContext.batchId,
      batchSize: state.batchContext.batchSize,
      ...(loop.lspSync ? { lspSync: loop.lspSync } : {}),
      ...loop.getAgentEventFields(),
    });
  } catch (err) {
    return {
      success: false,
      output: "",
      error: extractErrorMessage(err),
      artifacts: [],
    };
  }
}

async function maybeRunDoubleCheck(
  loop: ToolExecutionHost,
  state: ExecutionState,
  result: ToolResult,
  preEditBaseline: import("./double-check.js").DiagnosticBaseline | undefined,
): Promise<ToolResult> {
  if (!shouldRunDoubleCheck(loop, state, result)) return result;
  const modifiedFiles = result.artifacts.filter((artifact): artifact is string => typeof artifact === "string");
  if (modifiedFiles.length === 0) return result;

  const checkResult = await loop.doubleCheck!.check(modifiedFiles, preEditBaseline);
  const withMetadata = attachValidationMetadata(result, checkResult);
  if (checkResult.passed) {
    loop.unresolvedDoubleCheckFailure = false;
    return withMetadata;
  }
  loop.unresolvedDoubleCheckFailure = true;
  return appendValidationErrors(loop.doubleCheck!, withMetadata, checkResult);
}

function shouldRunDoubleCheck(loop: ToolExecutionHost, state: ExecutionState, result: ToolResult): boolean {
  return result.success && state.tool.category === "mutating" && Boolean(loop.doubleCheck?.isEnabled());
}

function attachValidationMetadata(result: ToolResult, checkResult: DoubleCheckResult): ToolResult {
  return {
    ...result,
    metadata: {
      ...(result.metadata ?? {}),
      validationResult: buildValidationResultMetadata(checkResult),
    },
  };
}

function appendValidationErrors(
  doubleCheck: DoubleCheck,
  result: ToolResult,
  checkResult: DoubleCheckResult,
): ToolResult {
  const feedback = doubleCheck.formatResults(checkResult);
  return {
    ...result,
    output: `${result.output}\n\nVALIDATION ERRORS:\n${feedback}\nFix these errors before continuing.`,
  };
}

function emitToolAfter(
  loop: ToolExecutionHost,
  options: {
    readonly callId: string;
    readonly toolName: string;
    readonly result: ToolResult;
    readonly durationMs: number;
    readonly batchContext?: ToolExecutionBatchContext;
  },
): void {
  const fileEditSummary = extractToolFileChangePreviewSummary(options.result.metadata);
  loop.bus.emit("tool:after", {
    name: options.toolName,
    result: options.result,
    fileEdits: fileEditSummary.fileEdits,
    fileEditHiddenCount: fileEditSummary.hiddenFileCount > 0 ? fileEditSummary.hiddenFileCount : undefined,
    callId: options.callId,
    durationMs: options.durationMs,
    batchId: options.batchContext?.batchId,
    batchSize: options.batchContext?.batchSize,
    ...loop.getAgentEventFields(),
  });
  loop.emitSubagentUpdate({
    milestone: "tool:after",
    toolName: options.toolName,
    toolCallId: options.callId,
    toolSuccess: options.result.success,
    durationMs: options.durationMs,
    summary: options.result.success ? `Completed ${options.toolName}` : `Failed ${options.toolName}`,
  });
}

function recordSessionState(
  loop: ToolExecutionHost,
  state: ExecutionState,
  executed: ExecutedToolResult,
): void {
  if (!loop.sessionState) return;
  loop.sessionState.batch(() => {
    recordSuccessfulToolState(loop, state, executed);
    recordFailedToolEnvironment(loop, state, executed.result);
  });
}

function recordSuccessfulToolState(
  loop: ToolExecutionHost,
  state: ExecutionState,
  executed: ExecutedToolResult,
): void {
  const hasMutatingArtifacts = executed.result.artifacts.length > 0 && state.tool.category === "mutating";
  if (!executed.result.success && !hasMutatingArtifacts) return;

  const target = getPrimarySummaryTarget(loop, state, executed.result);
  recordModifiedFiles(loop.sessionState!, state.tool.category, executed.result.artifacts);
  loop.captureReviewScopeFiles(state.effectiveCall, executed.originalOutput);
  addToolSummary(loop, state.effectiveCall, target, executed.originalOutput);
  recordReadonlyCoverage(loop, state.tool.category, state.effectiveCall);
}

function getPrimarySummaryTarget(
  loop: ToolExecutionHost,
  state: ExecutionState,
  result: ToolResult,
): string {
  return loop.getSummaryTarget(state.effectiveCall.name, state.effectiveCall.arguments)
    ?? (state.effectiveCall.arguments["path"] as string | undefined)
    ?? (result.artifacts.find((artifact): artifact is string => typeof artifact === "string"))
    ?? state.effectiveCall.name;
}

function recordModifiedFiles(
  sessionState: SessionState,
  category: ToolSpec["category"],
  artifacts: ReadonlyArray<unknown>,
): void {
  if (category !== "mutating") return;
  for (const artifact of artifacts) {
    if (typeof artifact === "string") sessionState.recordModifiedFile(artifact);
  }
}

function addToolSummary(
  loop: ToolExecutionHost,
  toolCall: PendingToolCall,
  target: string,
  output: string,
): void {
  loop.sessionState!.addToolSummary({
    tool: toolCall.name,
    target,
    summary: formatToolSummary(toolCall, output),
    iteration: loop.iterations,
  });
}

function recordReadonlyCoverage(
  loop: ToolExecutionHost,
  category: ToolSpec["category"],
  toolCall: PendingToolCall,
): void {
  if (category !== "readonly") return;
  const coverageTarget = loop.getSummaryTarget(toolCall.name, toolCall.arguments);
  if (coverageTarget) loop.sessionState!.recordReadonlyCoverage(toolCall.name, coverageTarget);
}

function recordFailedToolEnvironment(
  loop: ToolExecutionHost,
  state: ExecutionState,
  result: ToolResult,
): void {
  if (result.success) return;
  const fact = extractEnvFact(state.effectiveCall.name, result.error ?? "", result.output);
  if (fact) loop.sessionState!.addEnvFact(fact.key, fact.message);
}

function recordSuccessfulReadonlyCall(
  loop: ToolExecutionHost,
  state: ExecutionState,
  executed: ExecutedToolResult,
): void {
  if (!executed.result.success || state.tool.category !== "readonly") return;
  loop.successfulReadonlyCallKeys.add(buildReadonlyCallKey(state.effectiveCall.name, state.effectiveCall.arguments));
}

function buildValidationResultMetadata(result: DoubleCheckResult): ToolValidationResultMetadata {
  return {
    passed: result.passed,
    diagnosticErrors: [...result.diagnosticErrors],
    testPassed: result.testPassed,
    ...(result.testSummary
      ? {
        testSummary: {
          framework: result.testSummary.framework,
          passed: result.testSummary.passed,
          failed: result.testSummary.failed,
          failureMessages: [...result.testSummary.failureMessages],
        },
      }
      : {}),
    ...(typeof result.testOutput === "string" && result.testOutput.length > 0
      ? { testOutputPreview: buildValidationTestOutputPreview(result.testOutput) }
      : {}),
    ...(result.baselineFiltered !== undefined ? { baselineFiltered: result.baselineFiltered } : {}),
  };
}

function buildValidationTestOutputPreview(output: string): string {
  const lines = output.split("\n").filter((line) => line.trim().length > 0);
  const preview = lines.slice(0, 12).join("\n");
  if (lines.length > 12 || preview.length > 1_200) {
    return `${preview.slice(0, 1_200)}\n[preview truncated]`;
  }
  return preview;
}

function namespacedToolHint(
  toolName: string,
  availableTools: ReadonlyArray<ToolSpec>,
): string | null {
  if (!hasDisallowedToolPrefix(toolName)) return null;
  const canonical = toolName.replace(/^(?:functions|function|tools)\./, "");
  if (availableTools.some((tool) => tool.name === canonical)) {
    return `Use canonical tool names only. Try "${canonical}" (without namespace prefixes).`;
  }
  return "Use canonical tool names only (no prefixes like functions./function./tools.).";
}

function hasDisallowedToolPrefix(toolName: string): boolean {
  return /^(?:functions|function|tools)\./.test(toolName);
}

function buildReadonlyCallKey(
  toolName: string,
  args: Record<string, unknown>,
): string {
  const normalizedArgs = normalizeArgsForReadonlyKey(args);
  return `${toolName}:${JSON.stringify(normalizedArgs)}`;
}

function normalizeArgsForReadonlyKey(value: unknown): unknown {
  if (Array.isArray(value)) return value.map((item) => normalizeArgsForReadonlyKey(item));
  if (!value || typeof value !== "object") return value;
  return normalizeObjectArgsForReadonlyKey(value as Record<string, unknown>);
}

function normalizeObjectArgsForReadonlyKey(input: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const key of Object.keys(input).sort()) {
    const value = input[key];
    out[key] = key === "path" && typeof value === "string"
      ? normalizeRepoPath(value)
      : normalizeArgsForReadonlyKey(value);
  }
  return out;
}
