/**
 * Context management — adaptive message truncation to stay within token budgets.
 * Strategies:
 *   - sliding_window: drop oldest messages beyond keepRecentMessages
 *   - summarize: LLM-based summarization of older context (requires provider callback)
 *   - hybrid: sliding window + optional summarization
 *
 * From Cline: monitor token usage, truncate before hitting limits,
 * always preserve original task message.
 * ArkTS-compatible: no `any`, explicit types.
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
  ): ContextTruncationResult {
    const currentTokens = estimateMessageTokens(messages);

    // Check if truncation is needed
    const threshold = maxTokens * this.config.triggerRatio;
    if (currentTokens <= threshold) {
      return {
        messages,
        truncated: false,
        removedCount: 0,
        estimatedTokens: currentTokens,
      };
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
  ): Promise<ContextTruncationResult> {
    const currentTokens = estimateMessageTokens(messages);

    const threshold = maxTokens * this.config.triggerRatio;
    if (currentTokens <= threshold) {
      return {
        messages,
        truncated: false,
        removedCount: 0,
        estimatedTokens: currentTokens,
      };
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
    // Build maps of which callIds exist for each role
    const assistantCallIds = new Set<string>();
    for (const m of messages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) {
          assistantCallIds.add(tc.callId);
        }
      }
    }

    const toolResultIds = new Set<string>();
    for (const m of messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        toolResultIds.add(m.toolCallId);
      }
    }

    // Filter: remove orphaned messages
    const result: Message[] = [];
    for (const m of messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        // TOOL without matching ASSISTANT → drop
        if (!assistantCallIds.has(m.toolCallId)) continue;
      }

      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        // Check if ALL tool results are present
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
    if (messages.length <= 2) {
      return {
        messages,
        truncated: false,
        removedCount: 0,
        estimatedTokens: estimateMessageTokens(messages),
      };
    }

    // Find critical messages to preserve
    const systemMsg = messages[0]?.role === MessageRole.SYSTEM ? messages[0] : null;
    const firstUserIdx = messages.findIndex((m) => m.role === MessageRole.USER);

    // Keep recent messages
    const keepCount = this.config.keepRecentMessages;
    const preserved: Message[] = [];

    // Always keep system message
    if (systemMsg) {
      preserved.push(systemMsg);
    }

    // Always keep first user message (original task)
    if (firstUserIdx > 0) {
      preserved.push(messages[firstUserIdx]!);
    }

    // Keep the N most recent messages
    const startIdx = Math.max(
      (systemMsg ? 1 : 0) + (firstUserIdx > 0 ? 1 : 0),
      messages.length - keepCount,
    );
    const recentMessages = messages.slice(startIdx);

    // Merge preserved + recent, avoiding duplicates
    const result: Message[] = [...preserved];
    for (const msg of recentMessages) {
      if (!preserved.includes(msg)) {
        result.push(msg);
      }
    }

    // Further prune if still over budget
    let tokens = estimateMessageTokens(result);
    while (tokens > maxTokens && result.length > 2) {
      // Remove the message right after preserved section
      const removeIdx = preserved.length;
      if (removeIdx >= result.length) break;
      result.splice(removeIdx, 1);
      tokens = estimateMessageTokens(result);
    }

    // Sanitize: remove orphaned tool-call/tool-result messages
    const sanitized = this.sanitizeToolCallPairs(result);
    const finalTokens = estimateMessageTokens(sanitized);

    return {
      messages: sanitized,
      truncated: sanitized.length < messages.length,
      removedCount: messages.length - sanitized.length,
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
    const keepCount = this.config.keepRecentMessages;
    const recentStart = Math.max(systemMsg ? 1 : 0, messages.length - keepCount);
    const middleStart = systemMsg ? 1 : 0;

    const middleMessages = messages.slice(middleStart, recentStart);
    const recentMessages = messages.slice(recentStart);

    if (middleMessages.length === 0) {
      return this.slidingWindow(messages, maxTokens);
    }

    // Summarize the middle portion
    const summary = await this.summarize(middleMessages);

    const result: Message[] = [];
    if (systemMsg) {
      result.push(systemMsg);
    }
    result.push({
      role: MessageRole.USER,
      content: `[Conversation summary]: ${summary}`,
    });
    result.push(...recentMessages);

    // Sanitize: remove orphaned tool-call/tool-result messages
    // (the boundary between middle and recent can split tool-call pairs)
    const sanitized = this.sanitizeToolCallPairs(result);
    const tokens = estimateMessageTokens(sanitized);

    return {
      messages: sanitized,
      truncated: true,
      removedCount: messages.length - sanitized.length,
      estimatedTokens: tokens,
    };
  }
}
