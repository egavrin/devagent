import { clearPromptCache } from "./agent-prompt.js";
import type { SessionState, ToolResultSummary } from "./session-state.js";
import { PRUNED_MARKER_PREFIX, SUPERSEDED_MARKER_PREFIX } from "./session-state.js";
import { normalizeRepoPath } from "./task-loop-paths.js";
import type { DevAgentConfig, Message } from "../core/index.js";
import { MessageRole, estimateMessageTokens, estimateTokens } from "../core/index.js";

const DEFAULT_TOOL_RESULT_BUDGET = 200_000;
const MIN_PRUNE_MSG_TOKENS = 5_000;
const PRUNE_THRESHOLD_RATIO = 0.75;
const MAX_INLINE_SUMMARY_CHARS = 600;

const COMPACTABLE_TOOLS = new Set([
  "read_file",
  "run_command",
  "search_files",
  "find_files",
  "walk_directory",
  "git_diff",
  "git_status",
  "symbols",
  "diagnostics",
]);

const PRUNE_PRIORITY = new Map<string, number>([
  ["git_status", 0],
  ["find_files", 0],
  ["search_files", 1],
  ["run_command", 1],
  ["symbols", 1],
  ["read_file", 2],
  ["git_diff", 3],
  ["diagnostics", 3],
]);

interface ToolResultEntry {
  readonly index: number;
  readonly chars: number;
  readonly tool: string;
  readonly iteration: number;
}

interface PruningHost {
  readonly config: DevAgentConfig;
  readonly sessionState: SessionState | null;
  readonly stagnationDetector: { notifyCompaction(iteration: number): void };
  readonly iterations: number;
  messages: Message[];
  estimatedTokens: number;
  lastReportedInputTokens: number;
  toolResultIndices: Map<string, number>;
  approachingLimitWarned: boolean;
  toolResultTotalChars: number;
  toolResultEntries: ToolResultEntry[];
}

interface ToolCallInfo {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
}

interface PruneCandidate {
  readonly index: number;
  readonly priority: number;
}

export function microcompactTaskLoop(loop: PruningHost): void {
  const budget = loop.config.context.toolResultBudget ?? DEFAULT_TOOL_RESULT_BUDGET;
  if (loop.toolResultTotalChars <= budget) return;

  const candidates = getMicrocompactCandidates(loop.toolResultEntries);
  let cleared = 0;
  for (const entry of candidates) {
    if (loop.toolResultTotalChars <= budget) break;
    if (clearToolResultEntry(loop, entry)) cleared++;
  }
  if (cleared > 0) removeClearedToolResultEntries(loop);
}

export function pruneTaskLoopToolOutputs(
  loop: PruningHost,
  currentTokens: number,
  threshold: number,
): { savedTokens: number; prunedCount: number } {
  if (!loop.sessionState) return { savedTokens: 0, prunedCount: 0 };

  const targetSavings = getTargetSavings(currentTokens, threshold);
  if (targetSavings <= 0) return { savedTokens: 0, prunedCount: 0 };

  const protectedIndices = collectProtectedToolIndices(loop);
  const reversedSummaries = [...loop.sessionState.getToolSummaries()].reverse();
  const toolCallIndex = buildToolCallIndex(loop.messages);
  const candidates = collectPruneCandidates(loop.messages, protectedIndices, toolCallIndex);
  return applyPruneCandidates(loop, candidates, targetSavings, reversedSummaries, toolCallIndex);
}

export function resetTaskLoopPostCompactionState(loop: PruningHost, full: boolean): void {
  loop.lastReportedInputTokens = 0;
  loop.toolResultIndices.clear();
  loop.approachingLimitWarned = false;
  loop.toolResultTotalChars = 0;
  loop.toolResultEntries = [];
  clearPromptCache();
  if (full) loop.stagnationDetector.notifyCompaction(loop.iterations);
}

function getMicrocompactCandidates(entries: ReadonlyArray<ToolResultEntry>): ToolResultEntry[] {
  return entries
    .filter((entry) => COMPACTABLE_TOOLS.has(entry.tool))
    .sort((a, b) => a.iteration - b.iteration);
}

function clearToolResultEntry(loop: PruningHost, entry: ToolResultEntry): boolean {
  const message = loop.messages[entry.index];
  if (!isPrunableToolMessage(message)) return false;

  const replacement = `[Old tool result content cleared — ${entry.tool}, ${entry.chars} chars]`;
  const oldTokens = estimateMessageTokens([message]);
  loop.messages[entry.index] = { ...message, content: replacement };
  const newTokens = estimateMessageTokens([loop.messages[entry.index]!]);
  loop.estimatedTokens -= oldTokens - newTokens;
  loop.toolResultTotalChars -= entry.chars;
  return true;
}

function isPrunableToolMessage(message: Message | undefined): message is Message {
  if (!message || message.role !== MessageRole.TOOL) return false;
  return !isAlreadyCompactedContent(message.content);
}

function isAlreadyCompactedContent(content: string | null | undefined): boolean {
  return Boolean(content?.startsWith(PRUNED_MARKER_PREFIX) || content?.startsWith(SUPERSEDED_MARKER_PREFIX));
}

function removeClearedToolResultEntries(loop: PruningHost): void {
  loop.toolResultEntries = loop.toolResultEntries.filter((entry) => {
    const message = loop.messages[entry.index];
    return message?.role === MessageRole.TOOL
      && !message.content?.startsWith("[Old tool result content cleared");
  });
}

function getTargetSavings(currentTokens: number, threshold: number): number {
  const targetTokens = threshold * PRUNE_THRESHOLD_RATIO;
  return currentTokens - targetTokens;
}

function collectProtectedToolIndices(loop: PruningHost): Set<number> {
  const protectTokens = loop.config.context.pruneProtectTokens ?? 60_000;
  const protectedIndices = new Set<number>();
  let protectedTokens = 0;
  for (let i = loop.messages.length - 1; i >= 0; i--) {
    const message = loop.messages[i]!;
    if (message.role !== MessageRole.TOOL) continue;
    if (protectedTokens >= protectTokens) break;
    protectedTokens += estimateTokens(message.content ?? "");
    protectedIndices.add(i);
  }
  return protectedIndices;
}

function buildToolCallIndex(messages: ReadonlyArray<Message>): Map<string, ToolCallInfo> {
  const index = new Map<string, ToolCallInfo>();
  for (const message of messages) {
    if (message.role !== MessageRole.ASSISTANT || !message.toolCalls) continue;
    for (const toolCall of message.toolCalls) {
      index.set(toolCall.callId, { name: toolCall.name, arguments: toolCall.arguments });
    }
  }
  return index;
}

function collectPruneCandidates(
  messages: ReadonlyArray<Message>,
  protectedIndices: ReadonlySet<number>,
  toolCallIndex: ReadonlyMap<string, ToolCallInfo>,
): PruneCandidate[] {
  const candidates: PruneCandidate[] = [];
  for (let i = 0; i < messages.length; i++) {
    const candidate = getPruneCandidate(messages[i]!, i, protectedIndices, toolCallIndex);
    if (candidate) candidates.push(candidate);
  }
  return candidates.sort((a, b) => a.priority - b.priority || a.index - b.index);
}

function getPruneCandidate(
  message: Message,
  index: number,
  protectedIndices: ReadonlySet<number>,
  toolCallIndex: ReadonlyMap<string, ToolCallInfo>,
): PruneCandidate | null {
  if (message.role !== MessageRole.TOOL) return null;
  if (protectedIndices.has(index) || message.pinned) return null;
  if (isSmallToolMessage(message) || isAlreadyCompactedContent(message.content)) return null;
  const toolName = message.toolCallId ? toolCallIndex.get(message.toolCallId)?.name : undefined;
  return { index, priority: PRUNE_PRIORITY.get(toolName ?? "") ?? 1 };
}

function isSmallToolMessage(message: Message): boolean {
  return estimateTokens(message.content ?? "") <= MIN_PRUNE_MSG_TOKENS;
}

function applyPruneCandidates(
  loop: PruningHost,
  candidates: ReadonlyArray<PruneCandidate>,
  targetSavings: number,
  reversedSummaries: ReadonlyArray<ToolResultSummary>,
  toolCallIndex: ReadonlyMap<string, ToolCallInfo>,
): { savedTokens: number; prunedCount: number } {
  let savedTokens = 0;
  let prunedCount = 0;
  for (const { index } of candidates) {
    if (savedTokens >= targetSavings) break;
    const delta = pruneMessageAtIndex(loop, index, reversedSummaries, toolCallIndex);
    savedTokens += delta;
    prunedCount++;
  }
  return { savedTokens, prunedCount };
}

function pruneMessageAtIndex(
  loop: PruningHost,
  index: number,
  reversedSummaries: ReadonlyArray<ToolResultSummary>,
  toolCallIndex: ReadonlyMap<string, ToolCallInfo>,
): number {
  const message = loop.messages[index]!;
  const messageTokens = estimateTokens(message.content ?? "");
  const replacement = buildPrunedToolPlaceholder(message, messageTokens, reversedSummaries, toolCallIndex);
  const replacementTokens = estimateTokens(replacement);
  loop.messages[index] = { ...message, content: replacement };
  const delta = messageTokens - replacementTokens;
  loop.estimatedTokens -= delta;
  return delta;
}

function buildPrunedToolPlaceholder(
  message: Message,
  prunedTokens: number,
  reversedSummaries: ReadonlyArray<ToolResultSummary>,
  toolCallIndex: ReadonlyMap<string, ToolCallInfo>,
): string {
  const fallback = `${PRUNED_MARKER_PREFIX} tool output pruned (${prunedTokens} tokens). Check session state for details.]`;
  const summary = findMatchingToolSummary(message, reversedSummaries, toolCallIndex);
  if (!summary) return fallback;

  const inline = `${summary.tool}(${summary.target}): ${summary.summary}`;
  const snippet = inline.length > MAX_INLINE_SUMMARY_CHARS
    ? `${inline.slice(0, MAX_INLINE_SUMMARY_CHARS - 3)}...`
    : inline;
  return `${PRUNED_MARKER_PREFIX} ${snippet} (pruned from ${prunedTokens} tokens)]`;
}

function findMatchingToolSummary(
  message: Message,
  reversedSummaries: ReadonlyArray<ToolResultSummary>,
  toolCallIndex: ReadonlyMap<string, ToolCallInfo>,
): ToolResultSummary | null {
  if (!message.toolCallId) return null;
  const toolCall = toolCallIndex.get(message.toolCallId);
  if (!toolCall) return null;

  const target = getToolCallTarget(toolCall);
  const normalizedTarget = normalizeRepoPath(target);
  return reversedSummaries.find((summary) => {
    if (summary.tool !== toolCall.name) return false;
    const summaryTarget = normalizeRepoPath(summary.target);
    return summaryTarget === normalizedTarget || summary.target === target;
  }) ?? null;
}

function getToolCallTarget(toolCall: ToolCallInfo): string {
  const rawTarget = (toolCall.arguments["path"] as string | undefined) ?? toolCall.name;
  return typeof rawTarget === "string" ? rawTarget : String(rawTarget);
}
