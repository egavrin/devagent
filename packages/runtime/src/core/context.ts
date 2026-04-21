/**
 * Context management — adaptive message truncation to stay within token budgets.
 * Strategies:
 *   - sliding_window: drop oldest messages beyond keepRecentMessages
 *   - summarize: LLM-based summarization of older context (requires provider callback)
 *   - hybrid: sliding window + optional summarization
 *
 * From Cline: monitor token usage, truncate before hitting limits,
 * always preserve original task message.
 * Context manager with explicit state and token budgeting types.
 */

import type { Message, ContextConfig } from "./types.js";
import { MessageRole } from "./types.js";

// ─── Token Estimation ────────────────────────────────────────

/**
 * Rough token estimate: ~4 chars per token for English text.
 * Good enough for budget monitoring; exact counts come from the provider.
 */
export function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

/**
 * Estimate total tokens in a message array.
 */
export function estimateMessageTokens(messages: ReadonlyArray<Message>): number {
  let total = 0;
  for (const msg of messages) {
    if (msg.content) {
      total += estimateTokens(msg.content);
    }
    if (msg.toolCalls) {
      for (const tc of msg.toolCalls) {
        total += estimateTokens(JSON.stringify(tc.arguments));
        total += estimateTokens(tc.name);
      }
    }
  }
  return total;
}

// ─── Context Manager ─────────────────────────────────────────

export interface ContextTruncationResult {
  readonly messages: ReadonlyArray<Message>;
  readonly truncated: boolean;
  readonly removedCount: number;
  readonly estimatedTokens: number;
}

/**
 * Summarization callback — invoked when the hybrid strategy needs to
 * compress older messages into a summary. The callback should use an
 * LLM to produce a concise summary of the provided messages.
 */
export type SummarizeCallback = (
  messages: ReadonlyArray<Message>,
) => Promise<string>;

export class ContextFitError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ContextFitError";
  }
}

function collectToolResultIds(messages: ReadonlyArray<Message>): Set<string> {
  const ids = new Set<string>();
  for (const message of messages) {
    if (message.role === MessageRole.TOOL && message.toolCallId) {
      ids.add(message.toolCallId);
    }
  }
  return ids;
}

function collectSurvivingCallIds(
  messages: ReadonlyArray<Message>,
  toolResultIds: ReadonlySet<string>,
): Set<string> {
  const ids = new Set<string>();
  for (const message of messages) {
    if (message.role === MessageRole.ASSISTANT && hasAllToolResults(message, toolResultIds)) {
      for (const toolCall of message.toolCalls ?? []) ids.add(toolCall.callId);
    }
  }
  return ids;
}

function sanitizeToolPairMessage(
  message: Message,
  pairs: ToolPairSets,
): Message | null {
  if (message.role === MessageRole.TOOL && message.toolCallId) {
    return pairs.survivingCallIds.has(message.toolCallId) ? message : null;
  }
  if (message.role !== MessageRole.ASSISTANT || !message.toolCalls) {
    return message;
  }
  if (hasAllToolResults(message, pairs.toolResultIds)) {
    return message;
  }
  return message.content && message.content.trim().length > 0
    ? { role: message.role, content: message.content }
    : null;
}

function hasAllToolResults(
  message: Message,
  toolResultIds: ReadonlySet<string>,
): boolean {
  return Boolean(
    message.toolCalls?.every((toolCall) => toolResultIds.has(toolCall.callId)),
  );
}

function getSystemMessage(messages: ReadonlyArray<Message>): Message | null {
  return messages[0]?.role === MessageRole.SYSTEM ? messages[0] : null;
}

function collectCriticalMessages(messages: ReadonlyArray<Message>): Set<Message> {
  const preserved = new Set<Message>();
  const systemMessage = getSystemMessage(messages);
  const firstUser = messages.find((message) => message.role === MessageRole.USER);
  if (systemMessage) preserved.add(systemMessage);
  if (firstUser) preserved.add(firstUser);
  return preserved;
}

function buildHybridResult(
  messages: ReadonlyArray<Message>,
  preserved: ReadonlySet<Message>,
  recentMessages: ReadonlyArray<Message>,
  summary: string,
): Message[] {
  const result = collectOlderPreservedMessages(messages, preserved, recentMessages);
  const trimmedSummary = summary.trim();
  if (trimmedSummary.length > 0) {
    result.push({
      role: MessageRole.ASSISTANT,
      content: `[Conversation summary]: ${summary}`,
    });
  }
  appendMissingRecentMessages(result, recentMessages);
  return result;
}

function collectOlderPreservedMessages(
  messages: ReadonlyArray<Message>,
  preserved: ReadonlySet<Message>,
  recentMessages: ReadonlyArray<Message>,
): Message[] {
  const recentSet = new Set(recentMessages);
  return messages.filter((message) => preserved.has(message) && !recentSet.has(message));
}

function appendMissingRecentMessages(
  result: Message[],
  recentMessages: ReadonlyArray<Message>,
): void {
  const added = new Set(result);
  for (const message of recentMessages) {
    if (!added.has(message)) result.push(message);
  }
}

function collectPinnedMessages(
  messages: ReadonlyArray<Message>,
  preserved: Set<Message>,
): Set<string> {
  const pinnedCallIds = new Set<string>();
  for (const message of messages) {
    if (!message.pinned) continue;
    preserved.add(message);
    if (message.role === MessageRole.TOOL && message.toolCallId) {
      pinnedCallIds.add(message.toolCallId);
    }
  }
  return pinnedCallIds;
}

function collectPinnedAssistantOwners(
  messages: ReadonlyArray<Message>,
  pinnedCallIds: ReadonlySet<string>,
  preserved: Set<Message>,
): void {
  for (const message of messages) {
    if (isPinnedAssistantOwner(message, pinnedCallIds)) preserved.add(message);
  }
}

function isPinnedAssistantOwner(
  message: Message,
  pinnedCallIds: ReadonlySet<string>,
): boolean {
  return Boolean(
    pinnedCallIds.size > 0 &&
      message.role === MessageRole.ASSISTANT &&
      message.toolCalls?.some((toolCall) => pinnedCallIds.has(toolCall.callId)),
  );
}

interface ToolPairSets {
  readonly toolResultIds: ReadonlySet<string>;
  readonly survivingCallIds: ReadonlySet<string>;
}

export class ContextManager {
  private readonly config: ContextConfig;
  private summarize: SummarizeCallback | null = null;

  constructor(config: ContextConfig) {
    this.config = config;
  }

  /**
   * Set the summarization callback for hybrid strategy.
   */
  setSummarizeCallback(callback: SummarizeCallback): void {
    this.summarize = callback;
  }

  /**
   * Truncate messages to fit within the token budget.
   * Always preserves the system prompt (first message) and the original
   * task message (first user message).
   *
   * @param messages - Current message history
   * @param maxTokens - Maximum allowed tokens
   * @returns Truncated message array with metadata
   */
  truncate(
    messages: ReadonlyArray<Message>,
    maxTokens: number,
    options?: { force?: boolean },
  ): ContextTruncationResult {
    if (maxTokens <= 0) {
      return {
        messages,
        truncated: false,
        removedCount: 0,
        estimatedTokens: estimateMessageTokens(messages),
      };
    }

    const currentTokens = estimateMessageTokens(messages);

    // Check if truncation is needed (skip when force=true — caller already checked)
    if (!options?.force) {
      const threshold = maxTokens * this.config.triggerRatio;
      if (currentTokens <= threshold) {
        return {
          messages,
          truncated: false,
          removedCount: 0,
          estimatedTokens: currentTokens,
        };
      }
    }

    switch (this.config.pruningStrategy) {
      case "sliding_window":
        return this.slidingWindow(messages, maxTokens);
      case "summarize":
        return this.slidingWindow(messages, maxTokens); // Sync fallback
      case "hybrid":
        return this.slidingWindow(messages, maxTokens); // Sync fallback
      default:
        return this.slidingWindow(messages, maxTokens);
    }
  }

  /**
   * Async truncation with summarization support (for hybrid strategy).
   */
  async truncateAsync(
    messages: ReadonlyArray<Message>,
    maxTokens: number,
    options?: { force?: boolean },
  ): Promise<ContextTruncationResult> {
    if (maxTokens <= 0) {
      return {
        messages,
        truncated: false,
        removedCount: 0,
        estimatedTokens: estimateMessageTokens(messages),
      };
    }

    const currentTokens = estimateMessageTokens(messages);

    // Check if truncation is needed (skip when force=true — caller already checked)
    if (!options?.force) {
      const threshold = maxTokens * this.config.triggerRatio;
      if (currentTokens <= threshold) {
        return {
          messages,
          truncated: false,
          removedCount: 0,
          estimatedTokens: currentTokens,
        };
      }
    }

    if (
      this.config.pruningStrategy === "hybrid" ||
      this.config.pruningStrategy === "summarize"
    ) {
      if (this.summarize) {
        return this.hybridTruncation(messages, maxTokens);
      }
    }

    return this.slidingWindow(messages, maxTokens);
  }

  /**
   * Remove orphaned tool-call messages from a message array.
   *
   * After compaction, the message array may contain:
   * - ASSISTANT messages with toolCalls whose TOOL results were dropped
   * - TOOL messages whose corresponding ASSISTANT+toolCalls was dropped
   *
   * Both cases cause OpenAI API errors:
   * - "No tool output found for function call call_..." (orphaned ASSISTANT)
   * - Unexpected tool result without matching call (orphaned TOOL)
   *
   * Strategy: remove orphaned messages entirely. For ASSISTANT messages
   * that also have text content, strip the toolCalls rather than removing.
   */
  private sanitizeToolCallPairs(messages: Message[]): Message[] {
    const toolResultIds = collectToolResultIds(messages);
    const survivingCallIds = collectSurvivingCallIds(messages, toolResultIds);
    const result: Message[] = [];
    for (const m of messages) {
      const sanitized = sanitizeToolPairMessage(m, {
        toolResultIds,
        survivingCallIds,
      });
      if (sanitized) result.push(sanitized);
    }

    return result;
  }

  /**
   * Sliding window: keep system prompt, original user message, and N recent messages.
   */
  private slidingWindow(
    messages: ReadonlyArray<Message>,
    maxTokens: number,
  ): ContextTruncationResult {
    const systemMsg = getSystemMessage(messages);
    const preserved = collectCriticalMessages(messages);
    this.collectPinned(messages, preserved);

    // Keep the N most recent messages
    const startIdx = Math.max(systemMsg ? 1 : 0, messages.length - this.config.keepRecentMessages);
    const recentSet = new Set(messages.slice(startIdx));

    // Merge preserved + recent in original order, avoiding duplicates
    const result: Message[] = [];
    for (const m of messages) {
      if (preserved.has(m) || recentSet.has(m)) {
        result.push(m);
      }
    }

    // Sanitize before and after additional pruning to avoid tool-call orphans.
    const sanitized = this.sanitizeToolCallPairs(result);
    const bounded = this.pruneToBudget(sanitized, maxTokens);
    const finalMessages = this.sanitizeToolCallPairs(bounded);
    const finalTokens = estimateMessageTokens(finalMessages);
    if (finalTokens > maxTokens) {
      throw new ContextFitError(
        `Unable to fit context within token budget: ${finalTokens} > ${maxTokens}`,
      );
    }

    return {
      messages: finalMessages,
      truncated: this.didMessagesChange(messages, finalMessages),
      removedCount: messages.length - finalMessages.length,
      estimatedTokens: finalTokens,
    };
  }

  /**
   * Hybrid: summarize older messages, keep recent ones.
   */
  private async hybridTruncation(
    messages: ReadonlyArray<Message>,
    maxTokens: number,
  ): Promise<ContextTruncationResult> {
    if (!this.summarize || messages.length <= 3) {
      return this.slidingWindow(messages, maxTokens);
    }

    // Split into preserved, middle (to summarize), and recent
    const systemMsg = getSystemMessage(messages);
    const recentStart = Math.max(systemMsg ? 1 : 0, messages.length - this.config.keepRecentMessages);
    const preserved = collectCriticalMessages(messages);
    this.collectPinned(messages, preserved);

    const rawMiddle = messages
      .slice(systemMsg ? 1 : 0, recentStart)
      .filter((m) => !preserved.has(m));
    const recentMessages = messages.slice(recentStart);

    if (rawMiddle.length === 0) {
      return this.slidingWindow(messages, maxTokens);
    }

    // Sanitize middle before summarization: the recentStart boundary can split
    // an ASSISTANT+toolCalls from its TOOL results, leaving orphaned messages.
    // The summarize callback sends these to the LLM, which rejects broken pairs.
    const middleMessages = this.sanitizeToolCallPairs([...rawMiddle]);

    // Summarize the middle portion
    const summary = await this.summarize(middleMessages);

    const result = buildHybridResult(messages, preserved, recentMessages, summary);

    // Sanitize: remove orphaned tool-call/tool-result messages
    // (the boundary between middle and recent can split tool-call pairs)
    const sanitized = this.sanitizeToolCallPairs(result);
    const bounded = this.pruneToBudget(sanitized, maxTokens);
    const finalMessages = this.sanitizeToolCallPairs(bounded);
    const tokens = estimateMessageTokens(finalMessages);
    if (tokens > maxTokens) {
      throw new ContextFitError(
        `Unable to fit context within token budget: ${tokens} > ${maxTokens}`,
      );
    }

    return {
      messages: finalMessages,
      truncated: this.didMessagesChange(messages, finalMessages),
      removedCount: messages.length - finalMessages.length,
      estimatedTokens: tokens,
    };
  }

  /**
   * Remove oldest non-critical messages until token budget is met.
   * Critical messages: first SYSTEM and first USER message.
   */
  private pruneToBudget(messages: Message[], maxTokens: number): Message[] {
    const result = [...messages];
    const preserved = new Set<Message>();

    const systemMsg = result[0]?.role === MessageRole.SYSTEM ? result[0] : null;
    if (systemMsg) preserved.add(systemMsg);

    const firstUser = result.find((m) => m.role === MessageRole.USER) ?? null;
    if (firstUser) preserved.add(firstUser);

    this.collectPinned(result, preserved);

    let tokens = estimateMessageTokens(result);
    while (tokens > maxTokens) {
      const removeIdx = result.findIndex((m) => !preserved.has(m));
      if (removeIdx < 0) break;
      result.splice(removeIdx, 1);
      tokens = estimateMessageTokens(result);
    }

    return result;
  }

  /** Collect pinned messages and their owning ASSISTANT messages into `preserved`. */
  private collectPinned(
    messages: ReadonlyArray<Message>,
    preserved: Set<Message>,
  ): void {
    const pinnedCallIds = collectPinnedMessages(messages, preserved);
    collectPinnedAssistantOwners(messages, pinnedCallIds, preserved);
  }

  private didMessagesChange(
    original: ReadonlyArray<Message>,
    next: ReadonlyArray<Message>,
  ): boolean {
    if (original.length !== next.length) return true;
    for (let i = 0; i < original.length; i++) {
      if (original[i] !== next[i]) return true;
    }
    return false;
  }
}
