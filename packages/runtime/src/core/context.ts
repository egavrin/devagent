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
    // Phase 1: Determine which ASSISTANT+toolCalls messages survive.
    // An ASSISTANT survives only if ALL its tool results are present.
    // Build survivingCallIds from surviving ASSISTANTs only.
    const toolResultIds = new Set<string>();
    for (const m of messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        toolResultIds.add(m.toolCallId);
      }
    }

    const survivingCallIds = new Set<string>();
    for (const m of messages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        const allPresent = m.toolCalls.every((tc) => toolResultIds.has(tc.callId));
        if (allPresent) {
          for (const tc of m.toolCalls) {
            survivingCallIds.add(tc.callId);
          }
        }
      }
    }

    // Phase 2: Filter messages using survivingCallIds.
    // TOOL messages are kept only if their callId is in survivingCallIds
    // (i.e., the owning ASSISTANT has ALL its results present).
    const result: Message[] = [];
    for (const m of messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        if (!survivingCallIds.has(m.toolCallId)) continue;
      }

      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        const allPresent = m.toolCalls.every((tc) => toolResultIds.has(tc.callId));
        if (!allPresent) {
          // If the ASSISTANT also has text content, keep it but strip toolCalls
          if (m.content && m.content.trim().length > 0) {
            result.push({ role: m.role, content: m.content });
            continue;
          }
          // Pure tool-call message with no text → drop entirely
          continue;
        }
      }

      result.push(m);
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
    // Find critical messages to preserve
    const systemMsg = messages[0]?.role === MessageRole.SYSTEM ? messages[0] : null;
    const firstUserIdx = messages.findIndex((m) => m.role === MessageRole.USER);
    const firstUserMsg = firstUserIdx >= 0 ? messages[firstUserIdx]! : null;

    // Keep recent messages
    const keepCount = this.config.keepRecentMessages;
    const preserved = new Set<Message>();

    // Always keep system message
    if (systemMsg) {
      preserved.add(systemMsg);
    }

    // Always keep first user message (original task)
    if (firstUserMsg) {
      preserved.add(firstUserMsg);
    }

    this.collectPinned(messages, preserved);

    // Keep the N most recent messages
    const startIdx = Math.max(systemMsg ? 1 : 0, messages.length - keepCount);
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
    const systemMsg = messages[0]?.role === MessageRole.SYSTEM ? messages[0] : null;
    const firstUserIdx = messages.findIndex((m) => m.role === MessageRole.USER);
    const firstUserMsg = firstUserIdx >= 0 ? messages[firstUserIdx]! : null;
    const keepCount = this.config.keepRecentMessages;
    const recentStart = Math.max(systemMsg ? 1 : 0, messages.length - keepCount);
    const middleStart = systemMsg ? 1 : 0;
    const preserved = new Set<Message>();
    if (systemMsg) preserved.add(systemMsg);
    if (firstUserMsg) preserved.add(firstUserMsg);

    this.collectPinned(messages, preserved);

    const rawMiddle = messages
      .slice(middleStart, recentStart)
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

    // Build result: preserved messages in original order, then summary, then recent
    const result: Message[] = [];
    const recentSet = new Set(recentMessages);
    const added = new Set<Message>();
    for (const m of messages) {
      if (preserved.has(m) && !recentSet.has(m)) {
        result.push(m);
        added.add(m);
      }
    }
    if (summary.trim().length > 0) {
      result.push({
        role: MessageRole.ASSISTANT,
        content: `[Conversation summary]: ${summary}`,
      });
    }
    for (const m of recentMessages) {
      if (!added.has(m)) {
        result.push(m);
      }
    }

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
    const pinnedCallIds = new Set<string>();
    for (const m of messages) {
      if (m.pinned) {
        preserved.add(m);
        if (m.role === MessageRole.TOOL && m.toolCallId) {
          pinnedCallIds.add(m.toolCallId);
        }
      }
    }
    if (pinnedCallIds.size > 0) {
      for (const m of messages) {
        if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
          if (m.toolCalls.some((tc) => pinnedCallIds.has(tc.callId))) {
            preserved.add(m);
          }
        }
      }
    }
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
