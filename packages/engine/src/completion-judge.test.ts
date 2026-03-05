import { describe, it, expect, vi } from "vitest";
import type { LLMProvider, StreamChunk } from "@devagent/core";
import { judgeCompletion } from "./completion-judge.js";
import { SessionState } from "./session-state.js";

// ─── Mock helpers ────────────────────────────────────────────

function mockProvider(responseJson: string): LLMProvider {
  return {
    id: "mock",
    chat: vi.fn(async function* (): AsyncIterable<StreamChunk> {
      yield { type: "text", content: responseJson };
      yield { type: "done", content: "" };
    }),
    abort: vi.fn(),
  };
}

function throwingProvider(): LLMProvider {
  return {
    id: "mock-error",
    chat: vi.fn(async function* (): AsyncIterable<StreamChunk> {
      throw new Error("Provider error");
    }),
    abort: vi.fn(),
  };
}

// ─── Tests ───────────────────────────────────────────────────

describe("completion-judge — judgeCompletion", () => {
  it("classifies a final answer correctly (is_final=true)", async () => {
    const provider = mockProvider(
      '{"is_final": true, "confidence": 0.95, "reason": "Response summarizes findings and directly addresses the user request"}',
    );
    const result = await judgeCompletion(
      provider,
      "Here are the results of my analysis. I found three issues in the auth module: 1) Missing input validation, 2) No rate limiting, 3) Weak password hashing. Each has been fixed.",
      "Review the authentication module and fix any issues",
      null,
      10,
      true,
    );
    expect(result).not.toBeNull();
    expect(result!.is_final).toBe(true);
    expect(result!.confidence).toBe(0.95);
    expect(result!.reason).toBeTruthy();
  });

  it("classifies a progress update correctly (is_final=false)", async () => {
    const provider = mockProvider(
      '{"is_final": false, "confidence": 0.9, "reason": "Response states intent to continue with next steps"}',
    );
    const result = await judgeCompletion(
      provider,
      "I've found the first issue in the auth module. Let me now check the database layer for similar problems.",
      "Review the authentication module and fix any issues",
      null,
      5,
      true,
    );
    expect(result).not.toBeNull();
    expect(result!.is_final).toBe(false);
    expect(result!.confidence).toBe(0.9);
    expect(result!.reason).toBeTruthy();
  });

  it("returns null on provider error", async () => {
    const provider = throwingProvider();
    const result = await judgeCompletion(
      provider,
      "Here are the results of my analysis.",
      "Review the auth module",
      null,
      10,
      true,
    );
    expect(result).toBeNull();
  });

  it("returns null on invalid JSON response", async () => {
    const provider = mockProvider("This is not valid JSON at all");
    const result = await judgeCompletion(
      provider,
      "Here are the results.",
      "Review the auth module",
      null,
      10,
      true,
    );
    expect(result).toBeNull();
  });

  it("includes session state context in judge prompt", async () => {
    const provider = mockProvider(
      '{"is_final": true, "confidence": 0.8, "reason": "Final answer"}',
    );
    const state = new SessionState();
    state.recordModifiedFile("src/auth.ts");
    state.addFinding("Missing validation", "No input validation in login()", 5);

    await judgeCompletion(
      provider,
      "All issues have been fixed.",
      "Fix the auth module",
      state,
      10,
      true,
    );

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const messages = chatCall[0] as Array<{ content: string | null }>;
    const userMsg = messages.find((m) => m.content?.includes("Modified files"));
    expect(userMsg).toBeDefined();
    expect(userMsg!.content).toContain("Findings");
  });

  it("includes hadToolCalls and iteration in judge prompt", async () => {
    const provider = mockProvider(
      '{"is_final": false, "confidence": 0.7, "reason": "progress update"}',
    );
    await judgeCompletion(
      provider,
      "Let me continue investigating.",
      "Fix the auth module",
      null,
      15,
      false,
    );

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const messages = chatCall[0] as Array<{ content: string | null }>;
    const userMsg = messages.find((m) =>
      m.content?.includes("15") && m.content?.includes("tool"),
    );
    expect(userMsg).toBeDefined();
  });

  it("handles markdown-fenced JSON correctly", async () => {
    const provider = mockProvider(
      '```json\n{"is_final": true, "confidence": 0.85, "reason": "Complete answer with results"}\n```',
    );
    const result = await judgeCompletion(
      provider,
      "Here are the complete results.",
      "Analyze the codebase",
      null,
      8,
      true,
    );
    expect(result).not.toBeNull();
    expect(result!.is_final).toBe(true);
    expect(result!.confidence).toBe(0.85);
    expect(result!.reason).toBe("Complete answer with results");
  });
});
