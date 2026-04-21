import { buildPreCompactionSummary, judgeCompactionQuality } from "./compaction-judge.js";
import { extractPreCompactionKnowledge } from "./knowledge-extractor.js";
import { trySessionMemoryCompact } from "./session-memory-compact.js";
import type { SessionState } from "./session-state.js";
import type { ContextManager, EventBus, LLMProvider, Message } from "../core/index.js";
import { MessageRole, estimateMessageTokens, estimateTokens } from "../core/index.js";

const PINNED_TOKEN_BUDGET = 80_000;
const APPROACHING_LIMIT_RATIO = 0.6;
const REACTIVE_COMPACT_MAX_FAILURES = 3;

interface CompactionHost {
  readonly contextManager: ContextManager | null;
  readonly config: {
    readonly context: {
      readonly triggerRatio: number;
    };
  };
  readonly sessionState: SessionState | null;
  readonly bus: EventBus;
  readonly provider: LLMProvider;
  messages: Message[];
  estimatedTokens: number;
  lastReportedInputTokens: number;
  approachingLimitWarned: boolean;
  iterations: number;
  getEffectiveContextBudget(): number;
  pushMessage(message: Message): void;
  pruneToolOutputs(
    currentTokens: number,
    threshold: number,
  ): { savedTokens: number; prunedCount: number };
  resetPostCompactionState(full: boolean): void;
  injectSessionState(knownTokenEstimate?: number): void;
  reinjectSkillContent(): void;
}

interface CompactionStats {
  readonly estimatedTokens: number;
  readonly maxTokens: number;
  readonly threshold: number;
  readonly overhead: number;
}

interface ReactiveCompactionHost {
  readonly bus: EventBus;
  readonly sessionState: SessionState | null;
  readonly provider: LLMProvider;
  messages: Message[];
  iterations: number;
  reactiveCompactFailures: number;
  compactionCycles: number;
  microcompact(): void;
  maybeCompactContext(options?: { force?: boolean }): Promise<void>;
}

export async function maybeCompactTaskLoopContext(
  loop: CompactionHost,
  options?: { force?: boolean },
): Promise<void> {
  const stats = getCompactionStats(loop);
  if (!stats) return;
  if (skipCompaction(loop, stats, options)) return;

  const pruneResult = tryPruneToolOutputs(loop, stats, options);
  if (pruneResult.completed) return;

  if (trySessionMemoryCompaction(loop, stats)) return;
  await runFullCompaction(loop, stats, pruneResult);
}

export async function reactiveCompactTaskLoop(loop: ReactiveCompactionHost): Promise<void> {
  if (loop.reactiveCompactFailures >= REACTIVE_COMPACT_MAX_FAILURES) {
    emitReactiveCompactionCircuitBreaker(loop.bus);
    throw new Error("Reactive compaction circuit breaker: too many consecutive compaction failures");
  }

  try {
    loop.microcompact();
    await loop.maybeCompactContext({ force: true });
    loop.reactiveCompactFailures = 0;
    loop.compactionCycles++;
    await maybeExtractReactiveCompactionKnowledge(loop);
  } catch (err) {
    loop.reactiveCompactFailures++;
    throw err;
  }
}

function emitReactiveCompactionCircuitBreaker(bus: EventBus): void {
  bus.emit("error", {
    message: `Reactive compaction circuit breaker tripped after ${REACTIVE_COMPACT_MAX_FAILURES} consecutive failures.`,
    code: "REACTIVE_COMPACT_CIRCUIT_BREAKER",
    fatal: true,
  });
}

async function maybeExtractReactiveCompactionKnowledge(loop: ReactiveCompactionHost): Promise<void> {
  if (loop.compactionCycles % 2 !== 0 || !loop.sessionState) return;
  try {
    const firstUser = loop.messages.find((message) => message.role === MessageRole.USER && message.content);
    const summary = buildPreCompactionSummary(loop.sessionState, loop.messages, loop.iterations);
    const result = await extractPreCompactionKnowledge(
      loop.provider,
      summary,
      loop.sessionState,
      loop.messages,
      firstUser?.content ?? null,
    );
    if (result) {
      for (const entry of result.entries) {
        loop.sessionState.addKnowledge(entry.key, entry.content, loop.iterations);
      }
    }
  } catch {
    // Session memory extraction is best-effort during reactive recovery.
  }
}

function getCompactionStats(loop: CompactionHost): CompactionStats | null {
  if (!loop.contextManager) return null;
  const maxTokens = loop.getEffectiveContextBudget();
  if (maxTokens <= 0) return null;
  const estimatedTokens = Math.max(loop.estimatedTokens, loop.lastReportedInputTokens);
  const threshold = maxTokens * loop.config.context.triggerRatio;
  const overhead = Math.max(0, loop.lastReportedInputTokens - loop.estimatedTokens);
  return { estimatedTokens, maxTokens, threshold, overhead };
}

function skipCompaction(
  loop: CompactionHost,
  stats: CompactionStats,
  options?: { force?: boolean },
) {
  if (options?.force || stats.estimatedTokens > stats.threshold) return false;
  maybeWarnApproachingLimit(loop, stats.estimatedTokens, stats.threshold);
  return true;
}

function maybeWarnApproachingLimit(
  loop: CompactionHost,
  estimatedTokens: number,
  threshold: number,
): void {
  const warningThreshold = threshold * APPROACHING_LIMIT_RATIO;
  if (loop.approachingLimitWarned || estimatedTokens <= warningThreshold || !loop.sessionState) return;
  loop.approachingLimitWarned = true;
  loop.pushMessage({
    role: MessageRole.SYSTEM,
    content: "Context is filling up. You MUST persist any analysis conclusions or review findings NOW using save_finding. After context pruning, old tool outputs will be replaced with summaries. Do NOT re-read files already listed in session state — rely on the summaries and findings you've saved.",
  });
}

function enforcePinnedTokenBudget(loop: CompactionHost): void {
  let pinnedTokens = 0;
  const pinnedIndices: number[] = [];
  for (let i = 0; i < loop.messages.length; i++) {
    const message = loop.messages[i]!;
    if (!message.pinned) continue;
    pinnedTokens += estimateTokens(message.content ?? "");
    pinnedIndices.push(i);
  }
  unpinOldestOverBudget(loop, pinnedIndices, pinnedTokens);
}

function unpinOldestOverBudget(
  loop: CompactionHost,
  pinnedIndices: ReadonlyArray<number>,
  pinnedTokens: number,
): void {
  let remainingTokens = pinnedTokens;
  for (const index of pinnedIndices) {
    if (remainingTokens <= PINNED_TOKEN_BUDGET) break;
    const message = loop.messages[index]!;
    remainingTokens -= estimateTokens(message.content ?? "");
    loop.messages[index] = { ...message, pinned: undefined };
  }
}

function tryPruneToolOutputs(
  loop: CompactionHost,
  stats: CompactionStats,
  options?: { force?: boolean },
) {
  enforcePinnedTokenBudget(loop);
  const messageThreshold = Math.max(0, stats.threshold - stats.overhead);
  const result = loop.pruneToolOutputs(loop.estimatedTokens, messageThreshold);
  if (result.savedTokens <= 0) return { ...result, completed: false };

  loop.resetPostCompactionState(false);
  loop.injectSessionState();
  loop.estimatedTokens = estimateMessageTokens(loop.messages);
  const postPruneTokens = loop.estimatedTokens;
  if (!options?.force && postPruneTokens + stats.overhead <= stats.threshold) {
    emitCompacted(loop, stats.estimatedTokens, postPruneTokens, result);
    loop.resetPostCompactionState(true);
    return { ...result, completed: true };
  }
  return { ...result, completed: false };
}

function trySessionMemoryCompaction(loop: CompactionHost, stats: CompactionStats): boolean {
  if (!loop.sessionState?.hasContent()) return false;
  const result = trySessionMemoryCompact(loop.messages, loop.sessionState, stats.maxTokens);
  if (!result.success) return false;

  loop.messages = result.messages;
  loop.injectSessionState();
  loop.reinjectSkillContent();
  loop.resetPostCompactionState(true);
  loop.estimatedTokens = estimateMessageTokens(loop.messages);
  loop.bus.emit("context:compacted", {
    removedCount: 0,
    estimatedTokens: loop.estimatedTokens,
    tokensBefore: stats.estimatedTokens,
  });
  maybeEmitAggressiveCompactionWarning(loop, stats.estimatedTokens, loop.estimatedTokens, " via session memory");
  return true;
}

async function runFullCompaction(
  loop: CompactionHost,
  stats: CompactionStats,
  pruneResult: { readonly savedTokens: number; readonly prunedCount: number },
): Promise<void> {
  const summary = buildPreCompactionSummary(loop.sessionState, loop.messages, loop.iterations);
  await extractKnowledgeBeforeCompaction(loop, summary);
  loop.bus.emit("context:compacting", {
    estimatedTokens: stats.estimatedTokens,
    maxTokens: stats.maxTokens,
  });
  try {
    await truncateWithContextManager(loop, stats, summary, pruneResult);
  } catch (err) {
    loop.bus.emit("error", {
      message: `Context compaction failed: ${(err as Error).message}`,
      code: "COMPACTION_FAILED",
      fatal: true,
    });
    throw err;
  }
}

async function truncateWithContextManager(
  loop: CompactionHost,
  stats: CompactionStats,
  summary: string,
  pruneResult: { readonly savedTokens: number; readonly prunedCount: number },
): Promise<void> {
  const result = await loop.contextManager!.truncateAsync(loop.messages, stats.maxTokens, { force: true });
  if (result.truncated) {
    await applyFullCompactionResult(loop, stats, summary, pruneResult, result);
    return;
  }
  if (result.estimatedTokens > stats.maxTokens) {
    throw new Error(`Compaction did not fit budget: ${result.estimatedTokens} > ${stats.maxTokens}`);
  }
}

async function applyFullCompactionResult(
  loop: CompactionHost,
  stats: CompactionStats,
  summary: string,
  pruneResult: { readonly savedTokens: number; readonly prunedCount: number },
  result: { readonly messages: ReadonlyArray<Message>; readonly removedCount: number },
): Promise<void> {
  loop.messages = [...result.messages];
  loop.injectSessionState();
  loop.reinjectSkillContent();
  loop.resetPostCompactionState(true);
  loop.estimatedTokens = estimateMessageTokens(loop.messages);
  emitCompacted(loop, stats.estimatedTokens, loop.estimatedTokens, pruneResult, result.removedCount);
  maybeEmitAggressiveCompactionWarning(loop, stats.estimatedTokens, loop.estimatedTokens, "");
  await maybeWarnCompactionQuality(loop, summary);
  maybeAppendContinuationGuidance(loop);
  if (loop.estimatedTokens > stats.maxTokens) {
    throw new Error(`Compaction did not fit budget: ${loop.estimatedTokens} > ${stats.maxTokens}`);
  }
}

async function extractKnowledgeBeforeCompaction(loop: CompactionHost, summary: string): Promise<void> {
  if (!loop.sessionState) return;
  const firstUser = loop.messages.find((message) => message.role === MessageRole.USER && message.content);
  const result = await extractPreCompactionKnowledge(
    loop.provider,
    summary,
    loop.sessionState,
    loop.messages,
    firstUser?.content ?? null,
  );
  if (!result) return;
  for (const entry of result.entries) {
    loop.sessionState.addKnowledge(entry.key, entry.content, loop.iterations);
  }
}

async function maybeWarnCompactionQuality(loop: CompactionHost, summary: string): Promise<void> {
  const result = await judgeCompactionQuality(loop.provider, summary, loop.messages, loop.sessionState);
  if (!result || result.quality_loss < 0.6) return;
  loop.pushMessage({
    role: MessageRole.SYSTEM,
    content: `COMPACTION GAP WARNING: ${result.recommendation}\nMissing context: ${result.missing_context.join("; ")}`,
  });
  loop.bus.emit("error", {
    message: `Compaction quality loss: ${result.quality_loss.toFixed(2)}`,
    code: "COMPACTION_QUALITY_LOSS",
    fatal: false,
  });
}

function maybeAppendContinuationGuidance(loop: CompactionHost): void {
  if (!loop.sessionState) return;
  loop.pushMessage({
    role: MessageRole.SYSTEM,
    content: "Context was compacted. Your accumulated knowledge, plan, and file edit history are preserved above. Continue from your current plan step. Your next action should be based on the knowledge and progress sections — do NOT re-scan or re-read files listed in recent activity.",
  });
}

function emitCompacted(
  loop: CompactionHost,
  tokensBefore: number,
  estimatedTokens: number,
  pruneResult: { readonly savedTokens: number; readonly prunedCount: number },
  removedCount: number = 0,
): void {
  loop.bus.emit("context:compacted", {
    removedCount,
    prunedCount: pruneResult.prunedCount > 0 ? pruneResult.prunedCount : undefined,
    tokensSaved: pruneResult.savedTokens > 0 ? pruneResult.savedTokens : undefined,
    estimatedTokens,
    tokensBefore,
  });
}

function maybeEmitAggressiveCompactionWarning(
  loop: CompactionHost,
  before: number,
  after: number,
  suffix: string,
): void {
  const reduction = before > 0 ? (before - after) / before : 0;
  if (reduction <= 0.5) return;
  loop.bus.emit("error", {
    message: `Aggressive compaction: ${Math.round(reduction * 100)}% reduction (${before} → ${after} tokens)${suffix}. Critical context may be lost.`,
    code: "COMPACTION_AGGRESSIVE",
    fatal: false,
  });
}
