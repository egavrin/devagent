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

function collectAssistantCallIds(messages: ReadonlyArray<Message>): Set<string> {
  const ids = new Set<string>();
  for (const message of messages) {
    for (const toolCall of message.toolCalls ?? []) ids.add(toolCall.callId);
  }
  return ids;
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

function expectBalancedToolPairs(messages: ReadonlyArray<Message>): void {
  const callIds = collectAssistantCallIds(messages);
  const resultIds = collectToolResultIds(messages);
  for (const resultId of resultIds) expect(callIds.has(resultId)).toBe(true);
  for (const callId of callIds) expect(resultIds.has(callId)).toBe(true);
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

  it("includes persisted thinking content", () => {
    const tokens = estimateMessageTokens([
      {
        role: MessageRole.ASSISTANT,
        content: "ok",
        thinking: "abcd",
      },
    ]);

    expect(tokens).toBe(2);
  });
});
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

    const result = await mgr.truncateAsync(messages, 40);
    expect(result.truncated).toBe(true);

    // Should have system + summary + recent messages
    expect(result.messages[0]!.role).toBe(MessageRole.SYSTEM);

    // Should have a summary message
    const summaryMsg = result.messages.find(
      (m) => m.content?.includes("[Conversation summary]"),
    );
    expect(summaryMsg).toBeDefined();
  });
  it("no orphaned TOOL messages when sliding window splits a tool-call pair", () => {
    // Bug: startIdx can fall between ASSISTANT+toolCalls and its TOOL result.
    // The ASSISTANT gets dropped but the TOOL result stays → orphaned TOOL.
    //
    // With keepRecentMessages=3 and 8 messages:
    //   startIdx = max(2, 8-3) = 5
    //   recentMessages = messages[5:] = [TOOL(A), USER, ASSISTANT]
    //   ASSISTANT+toolCalls(A) at index 4 is DROPPED
    //   TOOL(A) at index 5 is KEPT → orphan!
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1, // trigger immediately
        keepRecentMessages: 3,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),                          // 0 (preserved)
      msg(MessageRole.USER, "task"),                           // 1 (preserved)
      msg(MessageRole.USER, "follow up"),                      // 2 (dropped)
      msg(MessageRole.ASSISTANT, "answer"),                    // 3 (dropped)
      {                                                        // 4 (dropped!)
        role: MessageRole.ASSISTANT,
        content: "Let me read",
        toolCalls: [{ name: "read_file", arguments: { path: "f" }, callId: "call_SPLIT" }],
      },
      {                                                        // 5 (kept — startIdx lands here!)
        role: MessageRole.TOOL,
        content: "file contents",
        toolCallId: "call_SPLIT",
      },
      msg(MessageRole.USER, "next question"),                  // 6 (kept)
      msg(MessageRole.ASSISTANT, "next answer"),               // 7 (kept)
    ];

    // triggerRatio=0.1, so threshold=50*0.1=5 tokens. Total ≈27 > 5 → triggers truncation.
    // After sliding window: result ≈13 tokens ≤ 50 → no further pruning.
    const result = mgr.truncate(messages, 50);
    expect(result.truncated).toBe(true);

    expectBalancedToolPairs(result.messages);
  });

  it("no orphaned TOOL messages after further-prune loop", () => {
    // Bug: the further-prune loop removes from index preserved.length.
    // If ASSISTANT+toolCalls gets removed but its small TOOL result stays,
    // we get an orphaned TOOL message.
    //
    // Scenario:
    //   result = [SYS, USER, USER(old), ASSISTANT+tools(A), TOOL(A small), USER, ASS]
    //   Prune removes USER(old), then ASSISTANT+tools(A).
    //   If removing ASSISTANT brings us under budget → TOOL(A) is orphaned.
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 20, // high keepCount so initial merge keeps everything
      }),
    );

    // Token budget: total ≈ ceil(3/4)+ceil(4/4)+ceil(50/4)+ceil(10/4)+ceil(5/4)+ceil(10/4)+ceil(10/4)
    //             = 1 + 1 + 13 + 3 + 2 + 3 + 3 = 26
    // Budget of 15 forces the prune loop to remove ~11 tokens worth
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),                            // ~1 token
      msg(MessageRole.USER, "task"),                             // ~1 token
      msg(MessageRole.USER, "x".repeat(50)),                     // ~13 tokens — prune target
      {                                                          // ~3 tokens — prune target
        role: MessageRole.ASSISTANT,
        content: "check",
        toolCalls: [{ name: "f", arguments: { p: "x" }, callId: "call_PRUNE" }],
      },
      {                                                          // ~2 tokens
        role: MessageRole.TOOL,
        content: "ok",
        toolCallId: "call_PRUNE",
      },
      msg(MessageRole.USER, "question 2"),                       // ~3 tokens
      msg(MessageRole.ASSISTANT, "answer 22"),                   // ~3 tokens
    ];

    const result = mgr.truncate(messages, 15);
    expect(result.truncated).toBe(true);

    expectBalancedToolPairs(result.messages);
  });

  it("no orphaned tool messages in hybrid truncation when boundary splits a pair", async () => {
    // Bug: hybrid truncation slices at recentStart.
    // If the boundary falls between ASSISTANT+toolCalls and its TOOL:
    //   middle = [..., ASSISTANT+toolCalls(A)]  → summarized away
    //   recent = [TOOL(A), ...]                 → kept as-is → orphaned TOOL
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 3, // keep last 3
      }),
    );

    mgr.setSummarizeCallback(async (msgs) => `Summary of ${msgs.length} messages`);

    // 8 messages, keepRecent=3: recentStart = max(1, 8-3) = 5
    // messages[4] = ASSISTANT+toolCalls → in middle (summarized away)
    // messages[5] = TOOL → in recent (kept) → ORPHAN!
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),                          // 0
      msg(MessageRole.USER, "task"),                           // 1 (middle)
      msg(MessageRole.ASSISTANT, "answer 1"),                  // 2 (middle)
      msg(MessageRole.USER, "question 2"),                     // 3 (middle)
      {                                                        // 4 (middle - summarized away!)
        role: MessageRole.ASSISTANT,
        content: "Let me check",
        toolCalls: [{ name: "search", arguments: { q: "x" }, callId: "call_HYBRID" }],
      },
      {                                                        // 5 (recent - KEPT!)
        role: MessageRole.TOOL,
        content: "results",
        toolCallId: "call_HYBRID",
      },
      msg(MessageRole.USER, "question 3"),                     // 6 (recent)
      msg(MessageRole.ASSISTANT, "answer 3"),                  // 7 (recent)
    ];

    // triggerRatio=0.1, threshold=50*0.1=5. Total ≈20 > 5 → triggers truncation.
    const result = await mgr.truncateAsync(messages, 50);
    expect(result.truncated).toBe(true);

    expectBalancedToolPairs(result.messages);
  });

  it("summarize callback receives sanitized messages (no orphaned tool-call pairs)", async () => {
    // Root cause of: "OpenAI stream error: AI_APICallError: No tool output found
    // for function call call_..."
    //
    // Bug: hybridTruncation slices middleMessages and passes them directly to the
    // summarize callback. If recentStart falls between ASSISTANT+toolCalls and its
    // TOOL result, middleMessages contains an ASSISTANT with toolCalls but no
    // matching TOOL result. The callback sends these to OpenAI, which rejects them.
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 3,
      }),
    );

    let receivedMessages: ReadonlyArray<Message> = [];
    mgr.setSummarizeCallback(async (msgs) => {
      receivedMessages = msgs;
      return `Summary of ${msgs.length} messages`;
    });

    // 8 messages, keepRecent=3: recentStart = max(1, 8-3) = 5
    // messages[4] = ASSISTANT+toolCalls(call_LEAK) → in middle
    // messages[5] = TOOL(call_LEAK) → in recent
    // Middle has ASSISTANT with toolCalls but no matching TOOL → broken if unsanitized
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),
      msg(MessageRole.USER, "task"),
      msg(MessageRole.ASSISTANT, "answer 1"),
      msg(MessageRole.USER, "question 2"),
      {
        role: MessageRole.ASSISTANT,
        content: "Let me check",
        toolCalls: [{ name: "search", arguments: { q: "x" }, callId: "call_LEAK" }],
      },
      {
        role: MessageRole.TOOL,
        content: "results",
        toolCallId: "call_LEAK",
      },
      msg(MessageRole.USER, "question 3"),
      msg(MessageRole.ASSISTANT, "answer 3"),
    ];

    await mgr.truncateAsync(messages, 50);

    // The summarize callback must NOT receive an ASSISTANT with toolCalls
    // that lacks its matching TOOL result
    expectBalancedToolPairs(receivedMessages);
  });

  it("no orphaned TOOL when ASSISTANT has multiple toolCalls and boundary splits the results", () => {
    // ROOT CAUSE of recurring "No tool call found for function call output with call_id" error.
    //
    // Bug: sanitizeToolCallPairs builds assistantCallIds from ALL original ASSISTANT messages,
    // then drops ASSISTANT messages where not ALL results are present. But TOOL messages
    // whose callIds were in the ORIGINAL assistantCallIds survive — even though the ASSISTANT
    // that owned them was dropped.
    //
    // This only manifests with MULTIPLE toolCalls per ASSISTANT. Single-toolCall cases work
    // because when the ASSISTANT is dropped, the TOOL is also dropped (all or nothing).
    // Multi-toolCall: ASSISTANT{A,B} dropped (B missing), but TOOL(A) survives → orphan.
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 3,
      }),
    );

    // 9 messages, keepRecent=3: startIdx = max(2, 9-3) = 6
    // messages[4] = ASSISTANT with toolCalls [A, B] → DROPPED by sliding window
    // messages[5] = TOOL(A) → DROPPED by sliding window
    // messages[6] = TOOL(B) → KEPT (in recent window)  ← startIdx lands here
    // messages[7] = USER → KEPT
    // messages[8] = ASSISTANT → KEPT
    //
    // After sliding window: [SYS, USER(task), TOOL(B), USER, ASSISTANT]
    // sanitize should drop TOOL(B) since its ASSISTANT was removed.
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),                          // 0 (preserved)
      msg(MessageRole.USER, "task"),                           // 1 (preserved)
      msg(MessageRole.USER, "follow up"),                      // 2 (dropped)
      msg(MessageRole.ASSISTANT, "ok"),                        // 3 (dropped)
      {                                                        // 4 (dropped)
        role: MessageRole.ASSISTANT,
        content: "",
        toolCalls: [
          { name: "read_file", arguments: { path: "a.ts" }, callId: "call_A" },
          { name: "read_file", arguments: { path: "b.ts" }, callId: "call_B" },
        ],
      },
      {                                                        // 5 (dropped)
        role: MessageRole.TOOL,
        content: "contents of a",
        toolCallId: "call_A",
      },
      {                                                        // 6 (kept — startIdx)
        role: MessageRole.TOOL,
        content: "contents of b",
        toolCallId: "call_B",
      },
      msg(MessageRole.USER, "next question"),                  // 7 (kept)
      msg(MessageRole.ASSISTANT, "next answer"),               // 8 (kept)
    ];

    const result = mgr.truncate(messages, 50);
    expect(result.truncated).toBe(true);

    expectBalancedToolPairs(result.messages);
  });

  it("multi-toolCall ASSISTANT partially in middle: summarize callback gets no orphans", async () => {
    // The hybrid boundary splits a multi-toolCall ASSISTANT's results between
    // middle and recent. After sanitizing middle, TOOL(A) should NOT survive
    // when its ASSISTANT{A,B} was stripped (because TOOL(B) is in recent).
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 3,
      }),
    );

    let receivedMessages: ReadonlyArray<Message> = [];
    mgr.setSummarizeCallback(async (msgs) => {
      receivedMessages = msgs;
      return `Summary of ${msgs.length} messages`;
    });

    // 9 messages, keepRecent=3: recentStart = max(1, 9-3) = 6
    // rawMiddle = messages[1..5] = [USER, ASS, USER, ASSISTANT{A,B}, TOOL(A)]
    // recent = messages[6..8] = [TOOL(B), USER, ASSISTANT]
    //
    // Middle contains ASSISTANT{A,B} and TOOL(A), but not TOOL(B).
    // sanitize should drop ASSISTANT{A,B} (not allPresent) AND TOOL(A) (orphaned).
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),                          // 0
      msg(MessageRole.USER, "task"),                           // 1 (middle)
      msg(MessageRole.ASSISTANT, "answer 1"),                  // 2 (middle)
      msg(MessageRole.USER, "question 2"),                     // 3 (middle)
      {                                                        // 4 (middle)
        role: MessageRole.ASSISTANT,
        content: "Let me check both files",
        toolCalls: [
          { name: "read_file", arguments: { path: "a.ts" }, callId: "call_MULTI_A" },
          { name: "read_file", arguments: { path: "b.ts" }, callId: "call_MULTI_B" },
        ],
      },
      {                                                        // 5 (middle)
        role: MessageRole.TOOL,
        content: "contents of a",
        toolCallId: "call_MULTI_A",
      },
      {                                                        // 6 (recent)
        role: MessageRole.TOOL,
        content: "contents of b",
        toolCallId: "call_MULTI_B",
      },
      msg(MessageRole.USER, "question 3"),                     // 7 (recent)
      msg(MessageRole.ASSISTANT, "answer 3"),                  // 8 (recent)
    ];

    await mgr.truncateAsync(messages, 50);

    expectBalancedToolPairs(receivedMessages);
  });

  it("further-prune loop with multi-toolCall: removes ASSISTANT then cleans orphaned TOOLs", () => {
    // The further-prune loop (while tokens > maxTokens) removes one message at a time.
    // If it removes an ASSISTANT{A,B} but leaves TOOL(A) and TOOL(B), sanitize
    // must clean up both orphaned TOOL messages.
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 20, // high keepCount to skip initial window pruning
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),                            // ~1 token
      msg(MessageRole.USER, "task"),                             // ~1 token
      msg(MessageRole.USER, "x".repeat(50)),                     // ~13 tokens — prune target
      {                                                          // ~5 tokens — prune target
        role: MessageRole.ASSISTANT,
        content: "",
        toolCalls: [
          { name: "f", arguments: { p: "x" }, callId: "call_M1" },
          { name: "g", arguments: { p: "y" }, callId: "call_M2" },
        ],
      },
      {                                                          // ~2 tokens
        role: MessageRole.TOOL,
        content: "r1",
        toolCallId: "call_M1",
      },
      {                                                          // ~2 tokens
        role: MessageRole.TOOL,
        content: "r2",
        toolCallId: "call_M2",
      },
      msg(MessageRole.USER, "question 2"),                       // ~3 tokens
      msg(MessageRole.ASSISTANT, "answer 22"),                   // ~3 tokens
    ];

    const result = mgr.truncate(messages, 15);
    expect(result.truncated).toBe(true);

    expectBalancedToolPairs(result.messages);
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

    const result = await mgr.truncateAsync(messages, 40);
    expect(result.truncated).toBe(true);
    // Falls back to sliding window — no summary message
    const hasSummary = result.messages.some(
      (m) => m.content?.includes("[Conversation summary]"),
    );
    expect(hasSummary).toBe(false);
  });

  it("hybrid truncation preserves the original user task message", async () => {
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    mgr.setSummarizeCallback(async () => "Short summary");

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task: keep this exact instruction"),
      msg(MessageRole.ASSISTANT, "Answer 1"),
      msg(MessageRole.USER, "Follow-up question"),
      msg(MessageRole.ASSISTANT, "Answer 2"),
      msg(MessageRole.USER, "Another follow-up"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    const result = await mgr.truncateAsync(messages, 40);
    expect(result.truncated).toBe(true);

    const originalTaskPresent = result.messages.some(
      (m) =>
        m.role === MessageRole.USER &&
        m.content === "Original task: keep this exact instruction",
    );
    expect(originalTaskPresent).toBe(true);
  });

  it("hybrid truncation enforces max token budget even if summary is too long", async () => {
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    // Deliberately pathological summary that would exceed small budgets if not pruned.
    mgr.setSummarizeCallback(async () => "Y".repeat(2000));

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      msg(MessageRole.ASSISTANT, "Answer 1"),
      msg(MessageRole.USER, "Question 2"),
      msg(MessageRole.ASSISTANT, "Answer 2"),
      msg(MessageRole.USER, "Question 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    const maxTokens = 30;
    const result = await mgr.truncateAsync(messages, maxTokens);
    expect(result.truncated).toBe(true);
    expect(result.estimatedTokens).toBeLessThanOrEqual(maxTokens);
    expect(estimateMessageTokens(result.messages)).toBeLessThanOrEqual(maxTokens);
  });

  it("truncateAsync with force bypasses internal threshold check", async () => {
    // Bug: maybeCompactContext uses Math.max(charEstimate, actualTokens)
    // to decide compaction is needed, but truncateAsync has its own
    // threshold check using only charEstimate. If charEstimate < threshold
    // but actualTokens > threshold, truncateAsync returns truncated: false.
    //
    // Fix: force option bypasses the internal threshold check.
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.8,
        keepRecentMessages: 2,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      msg(MessageRole.ASSISTANT, "Answer 1"),
      msg(MessageRole.USER, "Question 2"),
      msg(MessageRole.ASSISTANT, "Answer 2"),
      msg(MessageRole.USER, "Question 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    // Set budget so charEstimate is UNDER the 80% threshold
    // totalTokens ≈ 18, so threshold at 0.8 = budget * 0.8
    // If budget = 30, threshold = 24. 18 < 24 → not truncated.
    const budget = 30;

    // Without force: under threshold → not truncated
    const normalResult = await mgr.truncateAsync(messages, budget);
    expect(normalResult.truncated).toBe(false);

    // With force: bypasses threshold → truncated
    const forcedResult = await mgr.truncateAsync(messages, budget, { force: true });
    expect(forcedResult.truncated).toBe(true);
    expect(forcedResult.removedCount).toBeGreaterThan(0);
  });

  it("truncate (sync) with force bypasses internal threshold check", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.8,
        keepRecentMessages: 2,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      msg(MessageRole.ASSISTANT, "Answer 1"),
      msg(MessageRole.USER, "Question 2"),
      msg(MessageRole.ASSISTANT, "Answer 2"),
      msg(MessageRole.USER, "Question 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    const budget = 30;

    const normalResult = mgr.truncate(messages, budget);
    expect(normalResult.truncated).toBe(false);

    const forcedResult = mgr.truncate(messages, budget, { force: true });
    expect(forcedResult.truncated).toBe(true);
  });

  it("pinned TOOL survives sliding window outside keepRecentMessages", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      {
        role: MessageRole.ASSISTANT,
        content: "Let me check the diff",
        toolCalls: [{ name: "git_diff", arguments: {}, callId: "call_pinned" }],
      },
      {
        role: MessageRole.TOOL,
        content: "diff --git a/file.ts b/file.ts\n+important change",
        toolCallId: "call_pinned",
        pinned: true,
      },
      msg(MessageRole.USER, "Question 2"),
      msg(MessageRole.ASSISTANT, "Answer 2"),
      msg(MessageRole.USER, "Question 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    // keepRecentMessages=2 would normally drop the pinned TOOL at index 3
    const result = mgr.truncate(messages, 200);
    expect(result.truncated).toBe(true);

    // Pinned TOOL must survive
    const pinnedTool = result.messages.find(
      (m) => m.role === MessageRole.TOOL && m.pinned === true,
    );
    expect(pinnedTool).toBeDefined();
    expect(pinnedTool!.content).toContain("important change");
  });

  it("pinned TOOL's owning ASSISTANT survives sliding window", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      {
        role: MessageRole.ASSISTANT,
        content: "Let me check",
        toolCalls: [{ name: "git_diff", arguments: {}, callId: "call_owner" }],
      },
      {
        role: MessageRole.TOOL,
        content: "diff content",
        toolCallId: "call_owner",
        pinned: true,
      },
      msg(MessageRole.USER, "Question 2"),
      msg(MessageRole.ASSISTANT, "Answer 2"),
      msg(MessageRole.USER, "Question 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    const result = mgr.truncate(messages, 200);

    // The ASSISTANT that owns the pinned TOOL must also survive
    const owningAssistant = result.messages.find(
      (m) =>
        m.role === MessageRole.ASSISTANT &&
        m.toolCalls?.some((tc) => tc.callId === "call_owner"),
    );
    expect(owningAssistant).toBeDefined();
  });

  it("pinned messages excluded from hybridTruncation middle section", async () => {
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    let middleMessages: ReadonlyArray<Message> = [];
    mgr.setSummarizeCallback(async (msgs) => {
      middleMessages = msgs;
      return "Summary";
    });

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      {
        role: MessageRole.ASSISTANT,
        content: "Checking diff",
        toolCalls: [{ name: "git_diff", arguments: {}, callId: "call_mid" }],
      },
      {
        role: MessageRole.TOOL,
        content: "diff --git pinned content",
        toolCallId: "call_mid",
        pinned: true,
      },
      msg(MessageRole.USER, "Follow up"),
      msg(MessageRole.ASSISTANT, "Response"),
      msg(MessageRole.USER, "Question 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    const result = await mgr.truncateAsync(messages, 200);

    // Pinned messages should NOT be in the middle section sent to summarizer
    const pinnedInMiddle = middleMessages.some(
      (m) => m.pinned === true,
    );
    expect(pinnedInMiddle).toBe(false);

    // Pinned TOOL must survive in the final result
    const pinnedTool = result.messages.find(
      (m) => m.role === MessageRole.TOOL && m.pinned === true,
    );
    expect(pinnedTool).toBeDefined();
  });

  it("pinned messages survive pruneToBudget", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 20,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),
      msg(MessageRole.USER, "task"),
      {
        role: MessageRole.ASSISTANT,
        content: "check",
        toolCalls: [{ name: "git_diff", arguments: {}, callId: "call_budget" }],
      },
      {
        role: MessageRole.TOOL,
        content: "x".repeat(100), // 25 tokens
        toolCallId: "call_budget",
        pinned: true,
      },
      msg(MessageRole.USER, "q2"),
      msg(MessageRole.ASSISTANT, "a2"),
    ];

    // Budget tight enough that pruneToBudget needs to remove messages, but pinned survives
    const result = mgr.truncate(messages, 40);

    const pinnedTool = result.messages.find(
      (m) => m.role === MessageRole.TOOL && m.pinned === true,
    );
    expect(pinnedTool).toBeDefined();
  });

  it("sliding window with keepRecentMessages=40 keeps at least 40 messages when budget allows", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 40,
      }),
    );

    // Build 50 messages (system + user + 48 alternating user/assistant)
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),
      msg(MessageRole.USER, "original task"),
    ];
    for (let i = 0; i < 24; i++) {
      messages.push(msg(MessageRole.USER, `q${i}`));
      messages.push(msg(MessageRole.ASSISTANT, `a${i}`));
    }

    // Large budget so pruneToBudget doesn't kick in
    const result = mgr.truncate(messages, 10000);
    // With 50 msgs, keepRecentMessages=40 → startIdx = max(1, 50-40) = 10
    // preserved = [SYS, USER(original)] + messages[10..49] = 2 + 40 = 42 messages
    expect(result.messages.length).toBeGreaterThanOrEqual(40);
  });

  it("pruneToBudget enforces ceiling even with keepRecentMessages=40", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 40,
      }),
    );

    // Build 50 short messages
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),
      msg(MessageRole.USER, "task"),
    ];
    for (let i = 0; i < 24; i++) {
      messages.push(msg(MessageRole.USER, `q${i}`));
      messages.push(msg(MessageRole.ASSISTANT, `a${i}`));
    }

    // Very small budget forces pruning even with keepRecentMessages=40
    const result = mgr.truncate(messages, 20);
    expect(result.truncated).toBe(true);
    expect(result.estimatedTokens).toBeLessThanOrEqual(20);
  });

  it("pruneToBudget preserves owning ASSISTANT of pinned TOOL", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    // Setup: pinned TOOL outside the recent window, budget forces pruneToBudget
    // to consider pruning the owning ASSISTANT.
    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "sys"),
      msg(MessageRole.USER, "task"),
      // Filler to push pinned pair outside recent window
      msg(MessageRole.USER, "filler1"),
      msg(MessageRole.ASSISTANT, "filler2"),
      {
        role: MessageRole.ASSISTANT,
        content: null,
        toolCalls: [{ name: "git_diff", arguments: {}, callId: "call_pbt" }],
      },
      {
        role: MessageRole.TOOL,
        content: "x".repeat(80), // 20 tokens
        toolCallId: "call_pbt",
        pinned: true,
      },
      // Recent window (keepRecentMessages=2)
      msg(MessageRole.USER, "x".repeat(40)),  // 10 tokens
      msg(MessageRole.ASSISTANT, "x".repeat(40)),  // 10 tokens
    ];

    // After sliding window: [SYS, USER, ASST(toolCalls), TOOL(pinned), recent_USER, recent_ASST]
    // ~45 tokens total. Budget=42 forces pruneToBudget to remove the ASST(toolCalls)
    // unless we properly protect it as the pinned TOOL's owner.
    const result = mgr.truncate(messages, 42);

    // The pinned TOOL must survive — if its owning ASSISTANT was pruned,
    // sanitizeToolCallPairs would drop the orphaned pinned TOOL
    const pinnedTool = result.messages.find(
      (m) => m.role === MessageRole.TOOL && m.pinned === true,
    );
    expect(pinnedTool).toBeDefined();
    expect(pinnedTool!.toolCallId).toBe("call_pbt");

    // The owning ASSISTANT must also survive
    const owningAssistant = result.messages.find(
      (m) =>
        m.role === MessageRole.ASSISTANT &&
        m.toolCalls?.some((tc) => tc.callId === "call_pbt"),
    );
    expect(owningAssistant).toBeDefined();
  });

  it("pinned USER message survives slidingWindow outside keepRecentMessages", () => {
    const mgr = new ContextManager(
      makeConfig({
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      // Pinned USER message (e.g., preloaded review diff)
      {
        role: MessageRole.USER,
        content: "Review diff content that must be preserved",
        pinned: true,
      },
      msg(MessageRole.ASSISTANT, "Analysis of the diff"),
      msg(MessageRole.USER, "Follow up question"),
      msg(MessageRole.ASSISTANT, "Follow up answer"),
      msg(MessageRole.USER, "Question 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    // keepRecentMessages=2 would normally drop the pinned USER at index 2
    const result = mgr.truncate(messages, 200);
    expect(result.truncated).toBe(true);

    // Pinned USER must survive
    const pinnedUser = result.messages.find(
      (m) => m.role === MessageRole.USER && m.pinned === true,
    );
    expect(pinnedUser).toBeDefined();
    expect(pinnedUser!.content).toContain("Review diff content");
  });

  it("pinned USER message excluded from hybridTruncation middle section", async () => {
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "hybrid",
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    let middleMessages: ReadonlyArray<Message> = [];
    mgr.setSummarizeCallback(async (msgs) => {
      middleMessages = msgs;
      return "Summary";
    });

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "System prompt"),
      msg(MessageRole.USER, "Original task"),
      // Pinned USER message in middle section
      {
        role: MessageRole.USER,
        content: "Preloaded review diff that must not be summarized",
        pinned: true,
      },
      msg(MessageRole.ASSISTANT, "Analysis"),
      msg(MessageRole.USER, "Follow up"),
      msg(MessageRole.ASSISTANT, "Response"),
      msg(MessageRole.USER, "Question 3"),
      msg(MessageRole.ASSISTANT, "Answer 3"),
    ];

    const result = await mgr.truncateAsync(messages, 200);

    // Pinned USER should NOT be in the middle section sent to summarizer
    const pinnedInMiddle = middleMessages.some(
      (m) => m.pinned === true,
    );
    expect(pinnedInMiddle).toBe(false);

    // Pinned USER must survive in the final result
    const pinnedUser = result.messages.find(
      (m) => m.role === MessageRole.USER && m.pinned === true,
    );
    expect(pinnedUser).toBeDefined();
    expect(pinnedUser!.content).toContain("Preloaded review diff");
  });

  it("throws when critical preserved context alone exceeds max token budget", async () => {
    const mgr = new ContextManager(
      makeConfig({
        pruningStrategy: "sliding_window",
        triggerRatio: 0.1,
        keepRecentMessages: 2,
      }),
    );

    const messages: Message[] = [
      msg(MessageRole.SYSTEM, "S".repeat(400)),
      msg(MessageRole.USER, "Original task " + "T".repeat(400)),
      msg(MessageRole.ASSISTANT, "Answer"),
      msg(MessageRole.USER, "Follow-up"),
      msg(MessageRole.ASSISTANT, "Answer 2"),
    ];

    await expect(mgr.truncateAsync(messages, 20)).rejects.toThrow(
      "Unable to fit context within token budget",
    );
  });
