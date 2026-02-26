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

    // Invariant: every TOOL message must have its ASSISTANT present
    const assistantCallIds = new Set<string>();
    for (const m of result.messages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) assistantCallIds.add(tc.callId);
      }
    }
    for (const m of result.messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        expect(assistantCallIds.has(m.toolCallId)).toBe(true);
      }
    }

    // Invariant: every ASSISTANT+toolCalls must have all TOOL results present
    const toolResultIds = new Set<string>();
    for (const m of result.messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) toolResultIds.add(m.toolCallId);
    }
    for (const m of result.messages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) {
          expect(toolResultIds.has(tc.callId)).toBe(true);
        }
      }
    }
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

    // Invariant check
    const assistantCallIds = new Set<string>();
    for (const m of result.messages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) assistantCallIds.add(tc.callId);
      }
    }
    for (const m of result.messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        expect(assistantCallIds.has(m.toolCallId)).toBe(true);
      }
    }
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

    // Invariant check
    const assistantCallIds = new Set<string>();
    for (const m of result.messages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) assistantCallIds.add(tc.callId);
      }
    }
    for (const m of result.messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        expect(assistantCallIds.has(m.toolCallId)).toBe(true);
      }
    }
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
    const callIdsInCallback = new Set<string>();
    const resultIdsInCallback = new Set<string>();
    for (const m of receivedMessages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) callIdsInCallback.add(tc.callId);
      }
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        resultIdsInCallback.add(m.toolCallId);
      }
    }

    // Every ASSISTANT toolCall must have its TOOL result present
    for (const callId of callIdsInCallback) {
      expect(resultIdsInCallback.has(callId)).toBe(true);
    }
    // Every TOOL result must have its ASSISTANT toolCall present
    for (const resultId of resultIdsInCallback) {
      expect(callIdsInCallback.has(resultId)).toBe(true);
    }
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

    // Invariant: every TOOL must have a surviving ASSISTANT with matching callId
    const survivingCallIds = new Set<string>();
    for (const m of result.messages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) survivingCallIds.add(tc.callId);
      }
    }
    for (const m of result.messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        expect(survivingCallIds.has(m.toolCallId)).toBe(true);
      }
    }
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

    // The callback must NOT receive orphaned TOOL messages
    const callIdsInCallback = new Set<string>();
    const resultIdsInCallback = new Set<string>();
    for (const m of receivedMessages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) callIdsInCallback.add(tc.callId);
      }
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        resultIdsInCallback.add(m.toolCallId);
      }
    }

    // Every TOOL result must have its ASSISTANT toolCall present
    for (const resultId of resultIdsInCallback) {
      expect(callIdsInCallback.has(resultId)).toBe(true);
    }
    // Every ASSISTANT toolCall must have its TOOL result present
    for (const callId of callIdsInCallback) {
      expect(resultIdsInCallback.has(callId)).toBe(true);
    }
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

    // Invariant: every TOOL must have a surviving ASSISTANT with matching callId
    const survivingCallIds = new Set<string>();
    for (const m of result.messages) {
      if (m.role === MessageRole.ASSISTANT && m.toolCalls) {
        for (const tc of m.toolCalls) survivingCallIds.add(tc.callId);
      }
    }
    for (const m of result.messages) {
      if (m.role === MessageRole.TOOL && m.toolCallId) {
        expect(survivingCallIds.has(m.toolCallId)).toBe(true);
      }
    }
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
