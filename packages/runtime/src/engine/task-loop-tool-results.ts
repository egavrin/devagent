import { SUPERSEDED_MARKER_PREFIX } from "./session-state.js";
import type { SessionState } from "./session-state.js";
import { normalizeRepoPath } from "./task-loop-paths.js";
import type { EventBus, Message, ToolResult } from "../core/index.js";
import { MessageRole, estimateMessageTokens, formatDuration } from "../core/index.js";

const DEDUP_TOOLS = new Set(["read_file", "git_diff", "git_status"]);
const MAX_PINNED_DIFFS = 20;
const MAX_TOOL_OUTPUT_CHARS = 48_000;
const TRUNCATION_HEAD_LINES = 200;
const TRUNCATION_TAIL_LINES = 100;

interface ToolResultEntry {
  readonly index: number;
  readonly chars: number;
  readonly tool: string;
  readonly iteration: number;
}

interface ToolReadResult {
  readonly action: "allow" | "nudge" | "block";
  readonly message?: string;
}

interface ToolResultHost {
  readonly bus: EventBus;
  readonly iterations: number;
  readonly sessionState: SessionState | null;
  readonly stagnationDetector: {
    checkRereadStorm(toolName: string, target: string, iteration: number): void;
    trackFileRead(filePath: string): ToolReadResult;
  };
  messages: Message[];
  estimatedTokens: number;
  toolResultIndices: Map<string, number>;
  pinnedDiffCount: number;
  invokedSkillContent: Map<string, string>;
  toolResultTotalChars: number;
  toolResultEntries: ToolResultEntry[];
  pushMessage(message: Message): void;
  getAgentEventFields(): Record<string, unknown>;
}

interface ToolMessageContent {
  readonly content: string;
  readonly summaryOnly: boolean;
}

export function truncateToolOutput(output: string, maxChars: number = MAX_TOOL_OUTPUT_CHARS): string {
  if (output.length <= maxChars) return output;

  const lines = output.split("\n");
  if (lines.length <= TRUNCATION_HEAD_LINES + TRUNCATION_TAIL_LINES) {
    return output.slice(0, maxChars) + "\n\n[... output truncated ...]";
  }

  const headLines = lines.slice(0, TRUNCATION_HEAD_LINES);
  const tailLines = lines.slice(-TRUNCATION_TAIL_LINES);
  const omitted = lines.length - TRUNCATION_HEAD_LINES - TRUNCATION_TAIL_LINES;
  const joined = [...headLines, `\n[... ${omitted} lines truncated ...]\n`, ...tailLines].join("\n");
  if (joined.length <= maxChars) return joined;
  return joined.slice(0, maxChars) + "\n\n[... output truncated ...]";
}

export function appendTaskLoopToolResult(
  loop: ToolResultHost,
  callId: string,
  result: ToolResult,
  toolName?: string,
  toolArgs?: Record<string, unknown>,
): void {
  const toolContent = truncateToolOutput(getRawToolContent(result));
  if (toolName && toolArgs) dedupePreviousReadonlyToolResult(loop, toolName, toolArgs);
  if (maybeBlockRepeatedRead(loop, callId, toolName, toolArgs)) return;

  const shouldPin = shouldPinToolResult(loop, toolName, result);
  const toolMsgIndex = loop.messages.length;
  loop.pushMessage({
    role: MessageRole.TOOL,
    content: toolContent,
    toolCallId: callId,
    ...(shouldPin ? { pinned: true } : {}),
  });

  captureInvokedSkill(loop, toolName, result, toolArgs);
  trackToolResultBudget(loop, toolMsgIndex, toolName, toolContent);
  emitToolMessage(loop, callId, toolName, buildToolMessageContent(toolName, callId, toolContent, result));
}

export function maybeMergeTaskLoopDelegatedState(
  loop: Pick<ToolResultHost, "sessionState">,
  toolName: string,
  result: ToolResult,
): void {
  if (toolName !== "delegate" || !loop.sessionState || !result.metadata) return;
  const childState = result.metadata["childSessionState"];
  if (!childState || typeof childState !== "object") return;
  loop.sessionState.mergeDelegatedState(childState as import("./session-state.js").SessionStateJSON);
}

export function getTaskLoopSummaryTarget(
  toolName: string,
  args: Record<string, unknown>,
): string | null {
  const urlTarget = getUrlSummaryTarget(toolName, args);
  if (urlTarget) return urlTarget;

  const path = args["path"];
  if (typeof path === "string" && path.trim().length > 0) {
    return getPathSummaryTarget(toolName, args, normalizeRepoPath(path));
  }
  return getPatternOnlyTarget(toolName, args);
}

function getRawToolContent(result: ToolResult): string {
  if (result.success) return result.output;
  if (result.output) return `Error: ${result.error}\n\n${result.output}`;
  return `Error: ${result.error}`;
}

function dedupePreviousReadonlyToolResult(
  loop: ToolResultHost,
  toolName: string,
  toolArgs: Record<string, unknown>,
): void {
  if (!DEDUP_TOOLS.has(toolName)) return;
  const target = getDedupTarget(toolName, toolArgs);
  const dedupKey = `${toolName}:${target}`;
  const prevIdx = loop.toolResultIndices.get(dedupKey);
  if (prevIdx !== undefined && prevIdx < loop.messages.length) {
    replacePreviousReadonlyResult(loop, prevIdx, toolName);
    loop.stagnationDetector.checkRereadStorm(toolName, target, loop.iterations);
  }
  loop.toolResultIndices.set(dedupKey, loop.messages.length);
}

function getDedupTarget(toolName: string, toolArgs: Record<string, unknown>): string {
  const target = (toolArgs["path"] as string | undefined) ?? toolName;
  if (toolName === "git_diff") return getGitDiffDedupTarget(target, toolArgs);
  if (toolName === "read_file") return getReadFileDedupTarget(target, toolArgs);
  return target;
}

function getGitDiffDedupTarget(target: string, toolArgs: Record<string, unknown>): string {
  const ref = toolArgs["ref"] as string | undefined;
  const staged = toolArgs["staged"] as boolean | undefined;
  return `${target}:${ref ?? ""}:${staged ? "staged" : ""}`;
}

function getReadFileDedupTarget(target: string, toolArgs: Record<string, unknown>): string {
  const startLine = toolArgs["start_line"] as number | undefined;
  const endLine = toolArgs["end_line"] as number | undefined;
  if (startLine === undefined && endLine === undefined) return target;
  return `${target}:${startLine ?? ""}:${endLine ?? ""}`;
}

function replacePreviousReadonlyResult(loop: ToolResultHost, previousIndex: number, toolName: string): void {
  const previous = loop.messages[previousIndex];
  if (!previous || previous.role !== MessageRole.TOOL) return;
  const replacement = `${SUPERSEDED_MARKER_PREFIX} by later ${toolName}. See recent activity in session state.]`;
  loop.estimatedTokens -= estimateMessageTokens([previous]);
  const replacementMessage = { ...previous, content: replacement };
  loop.messages[previousIndex] = replacementMessage;
  loop.estimatedTokens += estimateMessageTokens([replacementMessage]);
}

function maybeBlockRepeatedRead(
  loop: ToolResultHost,
  callId: string,
  toolName?: string,
  toolArgs?: Record<string, unknown>,
): boolean {
  if (toolName !== "read_file" || !toolArgs) return false;
  const filePath = (toolArgs["path"] as string | undefined) ?? "";
  if (!filePath) return false;

  const readResult = loop.stagnationDetector.trackFileRead(filePath);
  if (readResult.action === "nudge") {
    loop.pushMessage({ role: MessageRole.SYSTEM, content: readResult.message! });
    return false;
  }
  if (readResult.action !== "block") return false;
  loop.pushMessage({ role: MessageRole.TOOL, content: readResult.message!, toolCallId: callId });
  emitToolMessage(loop, callId, toolName, {
    content: `[blocked: ${filePath} read limit exceeded]`,
    summaryOnly: false,
  });
  return true;
}

function shouldPinToolResult(loop: ToolResultHost, toolName: string | undefined, result: ToolResult): boolean {
  const shouldPin = toolName === "git_diff" && result.success && loop.pinnedDiffCount < MAX_PINNED_DIFFS;
  if (shouldPin) loop.pinnedDiffCount++;
  return shouldPin;
}

function captureInvokedSkill(
  loop: ToolResultHost,
  toolName: string | undefined,
  result: ToolResult,
  toolArgs?: Record<string, unknown>,
): void {
  if (toolName !== "invoke_skill" || !result.success || !result.output) return;
  const skillName = (toolArgs?.["name"] as string | undefined) ?? "unknown";
  loop.invokedSkillContent.set(skillName, result.output.slice(0, 20_000));
}

function trackToolResultBudget(
  loop: ToolResultHost,
  toolMsgIndex: number,
  toolName: string | undefined,
  toolContent: string,
): void {
  if (!toolName || toolContent.length === 0) return;
  loop.toolResultTotalChars += toolContent.length;
  loop.toolResultEntries.push({
    index: toolMsgIndex,
    chars: toolContent.length,
    tool: toolName,
    iteration: loop.iterations,
  });
}

function buildToolMessageContent(
  toolName: string | undefined,
  callId: string,
  toolContent: string,
  result: ToolResult,
): ToolMessageContent {
  if (toolName !== "delegate") return { content: toolContent, summaryOnly: false };
  const summary = result.metadata?.["delegateSummary"];
  if (!summary || typeof summary !== "object") {
    return { content: `Delegate ${callId} completed`, summaryOnly: true };
  }
  return buildDelegateMessageContent(summary as Record<string, unknown>);
}

function buildDelegateMessageContent(record: Record<string, unknown>): ToolMessageContent {
  const parts = getDelegateMessageParts(record);
  const details = getDelegateMessageDetails(record);
  const detailSuffix = details.length > 0 ? ` (${details.join(", ")})` : "";
  return {
    content: `${parts.join(" ")} completed${detailSuffix}`,
    summaryOnly: true,
  };
}

function getDelegateMessageParts(record: Record<string, unknown>): string[] {
  const agentId = typeof record["agentId"] === "string" ? record["agentId"] : "subagent";
  const agentType = typeof record["agentType"] === "string" ? record["agentType"] : "delegate";
  const parts = [`Subagent ${agentId} ${agentType}`];
  if (typeof record["laneLabel"] === "string" && record["laneLabel"].length > 0) {
    parts.push(record["laneLabel"]);
  }
  return parts;
}

function getDelegateMessageDetails(record: Record<string, unknown>): string[] {
  const details: string[] = [];
  if (typeof record["durationMs"] === "number") details.push(formatDuration(record["durationMs"]));
  if (typeof record["iterations"] === "number") details.push(`${record["iterations"]} iterations`);
  addDelegateQualityDetails(details, record["quality"]);
  return details;
}

function addDelegateQualityDetails(details: string[], quality: unknown): void {
  if (!quality || typeof quality !== "object") return;
  const qualityRecord = quality as Record<string, unknown>;
  if (typeof qualityRecord["score"] === "number") {
    details.push(`score ${Number(qualityRecord["score"]).toFixed(2)}`);
  }
  if (typeof qualityRecord["completeness"] === "string") {
    details.push(qualityRecord["completeness"]);
  }
}

function emitToolMessage(
  loop: ToolResultHost,
  callId: string,
  toolName: string | undefined,
  message: ToolMessageContent,
): void {
  loop.bus.emit("message:tool", {
    role: "tool" as const,
    content: message.content,
    toolCallId: callId,
    toolName,
    summaryOnly: message.summaryOnly,
    ...loop.getAgentEventFields(),
  });
}

function getUrlSummaryTarget(toolName: string, args: Record<string, unknown>): string | null {
  const url = args["url"];
  if (toolName === "fetch_url" && typeof url === "string" && url.trim().length > 0) return url;
  return null;
}

function getPathSummaryTarget(
  toolName: string,
  args: Record<string, unknown>,
  normalizedPath: string,
): string {
  if (toolName === "read_file") return getReadFileSummaryTarget(args, normalizedPath);
  if (toolName === "search_files") return getPatternPathTarget("search", args, normalizedPath) ?? normalizedPath;
  if (toolName === "find_files") return getPatternPathTarget("find", args, normalizedPath) ?? normalizedPath;
  return normalizedPath;
}

function getReadFileSummaryTarget(args: Record<string, unknown>, normalizedPath: string): string {
  const startLine = args["start_line"] as number | undefined;
  const endLine = args["end_line"] as number | undefined;
  if (startLine === undefined && endLine === undefined) return normalizedPath;
  return `${normalizedPath}:${startLine ?? ""}:${endLine ?? ""}`;
}

function getPatternOnlyTarget(toolName: string, args: Record<string, unknown>): string | null {
  if (toolName === "git_status") return "git_status";
  if (toolName === "search_files") return getPatternTarget("search", args);
  if (toolName === "find_files") return getPatternTarget("find", args);
  return null;
}

function getPatternPathTarget(
  prefix: "search" | "find",
  args: Record<string, unknown>,
  normalizedPath: string,
): string | null {
  const target = getPatternTarget(prefix, args);
  if (!target) return null;
  return normalizedPath !== "." ? `${target}@${normalizedPath}` : target;
}

function getPatternTarget(prefix: "search" | "find", args: Record<string, unknown>): string | null {
  const pattern = args["pattern"];
  if (typeof pattern !== "string") return null;
  const truncated = pattern.length > 60 ? `${pattern.slice(0, 57)}...` : pattern;
  return `${prefix}:${truncated}`;
}
