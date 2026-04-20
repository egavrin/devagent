import { describe, it, expect } from "vitest";

import { trySessionMemoryCompact } from "./session-memory-compact.js";
import { SessionState } from "./session-state.js";
import { MessageRole } from "../core/index.js";
import type { Message } from "../core/index.js";

function makeMessages(count: number): Message[] {
  const msgs: Message[] = [
    { role: MessageRole.SYSTEM, content: "You are a helpful agent." },
    { role: MessageRole.USER, content: "Fix the bug in auth.ts" },
  ];
  for (let i = 0; i < count; i++) {
    msgs.push({
      role: i % 2 === 0 ? MessageRole.ASSISTANT : MessageRole.USER,
      content: `Message ${i}: ${"x".repeat(50)}`,
    });
  }
  return msgs;
}

function makePopulatedState(): SessionState {
  const state = new SessionState({ persist: false });
  // Add enough entries to pass MIN_ENTRIES_FOR_COMPACT (3)
  state.addKnowledge("arch", "Monorepo with packages/runtime", 1);
  state.addKnowledge("lang", "TypeScript with Bun", 2);
  state.addFinding("bug", "Auth token not refreshed on 401", 3);
  state.addToolSummary({ tool: "read_file", target: "auth.ts", summary: "Contains login logic", iteration: 1 });
  state.addEnvFact("shell", "rg, fd, jq available");
  state.recordModifiedFile("src/auth.ts");
  return state;
}

describe("trySessionMemoryCompact", () => {
  it("returns success:false when too few entries", () => {
    const state = new SessionState({ persist: false });
    const messages = makeMessages(10);

    const result = trySessionMemoryCompact(messages, state, 100_000);
    expect(result.success).toBe(false);
    expect(result.messages).toHaveLength(0);
  });

  it("returns success:true with sufficient state", () => {
    const state = makePopulatedState();
    const messages = makeMessages(20);

    const result = trySessionMemoryCompact(messages, state, 100_000);
    expect(result.success).toBe(true);
    expect(result.messages.length).toBeGreaterThan(0);
  });

  it("preserves system prompt and first user message", () => {
    const state = makePopulatedState();
    const messages = makeMessages(10);

    const result = trySessionMemoryCompact(messages, state, 100_000);
    expect(result.success).toBe(true);

    const systemMsg = result.messages.find((m) => m.role === MessageRole.SYSTEM);
    expect(systemMsg).toBeDefined();
    expect(systemMsg!.content).toBe("You are a helpful agent.");

    // First user message preserved (should be early in the array)
    const userMsgs = result.messages.filter((m) => m.role === MessageRole.USER);
    expect(userMsgs.some((m) => m.content === "Fix the bug in auth.ts")).toBe(true);
  });

  it("keeps recent messages within budget", () => {
    const state = makePopulatedState();
    // Generate many long messages so the token budget actually constrains
    const msgs: Message[] = [
      { role: MessageRole.SYSTEM, content: "You are a helpful agent." },
      { role: MessageRole.USER, content: "Fix the bug in auth.ts" },
    ];
    for (let i = 0; i < 200; i++) {
      msgs.push({
        role: i % 2 === 0 ? MessageRole.ASSISTANT : MessageRole.USER,
        content: `Long message ${i}: ${"x".repeat(500)}`,
      });
    }

    // Budget that cannot fit all 200 extra messages
    const result = trySessionMemoryCompact(msgs, state, 5_000);
    expect(result.success).toBe(true);

    // Result should be significantly smaller than the original message array
    expect(result.messages.length).toBeLessThan(msgs.length);

    // Should contain the session memory summary
    const summaryMsg = result.messages.find(
      (m) => m.role === MessageRole.ASSISTANT && m.content?.includes("[Session memory summary"),
    );
    expect(summaryMsg).toBeDefined();
  });
});
