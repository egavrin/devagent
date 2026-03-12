import { describe, it, expect, vi } from "vitest";
import type { LLMProvider, StreamChunk } from "../core/index.js";
import { judgeSubagentOutput } from "./subagent-judge.js";

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

describe("subagent-judge — judgeSubagentOutput", () => {
  it("returns null on provider error", async () => {
    const provider = throwingProvider();
    const result = await judgeSubagentOutput(
      provider, "Analyze the auth system", "general",
      "Found several issues in the auth module.", 10, 30,
    );
    expect(result).toBeNull();
  });

  it("skips when iterations < 5", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.9, "completeness": "complete", "note": "Good output"}',
    );
    const result = await judgeSubagentOutput(
      provider, "Simple task", "general", "Done.", 3, 30,
    );
    expect(result).toBeNull();
    expect(provider.chat).not.toHaveBeenCalled();
  });

  it("returns high score for complete, relevant output", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.9, "completeness": "complete", "note": "Thorough analysis with actionable findings"}',
    );
    const result = await judgeSubagentOutput(
      provider, "Review the authentication module",
      "reviewer",
      "Found 3 issues: 1) Missing input validation in login(), 2) SQL injection in query(), 3) No rate limiting. Recommended fixes for each.",
      15, 30,
    );
    expect(result).not.toBeNull();
    expect(result!.quality_score).toBe(0.9);
    expect(result!.completeness).toBe("complete");
  });

  it("returns low score for empty/off-topic output", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.1, "completeness": "off_topic", "note": "Output does not address the assigned task"}',
    );
    const result = await judgeSubagentOutput(
      provider, "Implement the caching layer",
      "general",
      "I looked at some files but could not determine what to do.",
      25, 30,
    );
    expect(result).not.toBeNull();
    expect(result!.quality_score).toBe(0.1);
    expect(result!.completeness).toBe("off_topic");
  });

  it("detects 'hit max iterations without conclusion'", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.2, "completeness": "partial", "note": "Agent exhausted iteration budget without reaching a conclusion"}',
    );
    const result = await judgeSubagentOutput(
      provider, "Refactor the database layer",
      "general",
      "Still investigating...",
      30, 30, // hit max
    );
    expect(result).not.toBeNull();
    expect(result!.quality_score).toBe(0.2);
    expect(result!.completeness).toBe("partial");
  });

  it("includes efficiency signal in judge context", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.8, "completeness": "complete", "note": "ok"}',
    );
    await judgeSubagentOutput(
      provider, "task", "general", "output", 28, 30,
    );
    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const messages = chatCall[0] as Array<{ content: string | null }>;
    const userMsg = messages.find((m) => m.content?.includes("28/30"));
    expect(userMsg).toBeDefined();
  });
});
