import { describe, it, expect } from "vitest";
import {
  ContextManager,
  estimateTokens,
  estimateMessageTokens,
} from "./context.js";
import type { Message, ContextConfig } from "./types.js";
import { MessageRole } from "./types.js";

function makeConfig(overrides?: Partial<ContextConfig>): ContextConfig {
  return {
    pruningStrategy: "sliding_window",
    triggerRatio: 0.8,
    keepRecentMessages: 4,
    ...overrides,
  };
}

function msg(role: MessageRole, content: string): Message {
  return { role, content };
}

describe("estimateTokens", () => {
  it("estimates ~1 token per 4 chars", () => {
    expect(estimateTokens("hello world")).toBe(3); // 11 chars / 4 = 2.75 → 3
  });

  it("returns 0 for empty string", () => {
    expect(estimateTokens("")).toBe(0);
  });
});

describe("estimateMessageTokens", () => {
  it("sums token estimates for all messages", () => {
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "You are an assistant."),    // 21 chars → 6 tokens
      msg(MessageRole.USER, "Hello"),                       // 5 chars → 2 tokens
    ];
    const tokens = estimateMessageTokens(messages);
    expect(tokens).toBeGreaterThan(0);
    expect(tokens).toBe(8); // ceil(21/4) + ceil(5/4) = 6 + 2 = 8
  });
});

describe("ContextManager", () => {
  it("does not truncate when under threshold", () => {
    const mgr = new ContextManager(makeConfig({ triggerRatio: 0.8 }));
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),
      msg(MessageRole.USER, "hello"),
    ];
    const result = mgr.truncate(messages, 10000);
    expect(result.truncated).toBe(false);
    expect(result.messages).toEqual(messages);
    expect(result.removedCount).toBe(0);
  });

  it("applies sliding window when over threshold", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1, // trigger almost immediately
        keepRecentMessages: 2,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt that is somewhat long"),
      msg(MessageRole.USER, "First question"),
      msg(MessageRole.ASSISTANT, "First answer"),
      msg(MessageRole.USER, "Second question"),
      msg(MessageRole.ASSISTANT, "Second answer"),
      msg(MessageRole.USER, "Third question"),
      msg(MessageRole.ASSISTANT, "Third answer"),
    ];

    // Set a small budget to force truncation
    const result = mgr.truncate(messages, 30);
    expect(result.truncated).toBe(true);
    expect(result.removedCount).toBeGreaterThan(0);

    // System message should always be preserved
    expect(result.messages[0]!.role).toBe(MessageRole.SYSTEM);

    // First user message (original task) should be preserved
    const hasOriginalUser = result.messages.some(
      (m) => m.role === MessageRole.USER && m.content === "First question",
    );
    expect(hasOriginalUser).toBe(true);

    // Most recent messages should be kept
    const lastMsg = result.messages[result.messages.length - 1]!;
    expect(lastMsg.content).toBe("Third answer");
  });

  it("preserves system message and first user message", () => {
    const mgr = new ContextManager(
      makeConfig({ triggerRatio: 0.1, keepRecentMessages: 1 }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      msg(MessageRole.ASSISTANT, "Response 1"),
      msg(MessageRole.USER, "Follow up"),
      msg(MessageRole.ASSISTANT, "Response 2"),
    ];

    const result = mgr.truncate(messages, 20);
    expect(result.messages[0]!.role).toBe(MessageRole.SYSTEM);
    expect(result.messages[1]!.content).toBe("Original task");
  });

  it("handles empty message array", () => {
    const mgr = new ContextManager(makeConfig());
    const result = mgr.truncate([], 100);
    expect(result.truncated).toBe(false);
    expect(result.messages).toEqual([]);
  });

  it("handles 1-message array without truncation", () => {
    const mgr = new ContextManager(makeConfig({ triggerRatio: 0.1 }));
    const messages = [msg(MessageRole.SYSTEM, "sys")];
    const result = mgr.truncate(messages, 1);
    expect(result.truncated).toBe(false);
  });

  it("async truncation with summarize callback", async () => {
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    mgr.setSummarizeCallback(async (msgs) => {
      return `Summary of ${msgs.length} messages`;
    });

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Task 1"),
      msg(MessageRole.ASSISTANT, "Answer 1"),
      msg(MessageRole.USER, "Task 2"),
      msg(MessageRole.ASSISTANT, "Answer 2"),
      msg(MessageRole.USER, "Task 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    const result = await mgr.truncateAsync(messages, 30);
    expect(result.truncated).toBe(true);

    // Should have system + summary + recent messages
    expect(result.messages[0]!.role).toBe(MessageRole.SYSTEM);

    // Should have a summary message
    const summaryMsg = result.messages.find(
      (m) => m.content?.includes("[Conversation summary]"),
    );
    expect(summaryMsg).toBeDefined();
  });

  it("async truncation falls back to sliding window without callback", async () => {
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    // Use longer messages so token estimate exceeds the small budget
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "You are a helpful development assistant with many capabilities."),
      msg(MessageRole.USER, "First question about something important in the codebase."),
      msg(MessageRole.ASSISTANT, "Here is a detailed first answer about the codebase structure."),
      msg(MessageRole.USER, "Second question about implementation details and patterns."),
      msg(MessageRole.ASSISTANT, "Here is a detailed second answer about the implementation."),
      msg(MessageRole.USER, "Third question about testing strategies and coverage."),
      msg(MessageRole.ASSISTANT, "Here is a detailed third answer about testing and coverage."),
    ];

    const result = await mgr.truncateAsync(messages, 30);
    expect(result.truncated).toBe(true);
    // Falls back to sliding window — no summary message
    const hasSummary = result.messages.some(
      (m) => m.content?.includes("[Conversation summary]"),
    );
    expect(hasSummary).toBe(false);
  });
});
