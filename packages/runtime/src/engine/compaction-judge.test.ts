import { describe, it, expect, vi } from "vitest";

import { judgeCompactionQuality, buildPreCompactionSummary } from "./compaction-judge.js";
import { SessionState } from "./session-state.js";
import { MessageRole } from "../core/index.js";
import type { LLMProvider, Message, StreamChunk } from "../core/index.js";

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

function makeMessages(): Message[] {
  return [
    { role: MessageRole.SYSTEM, content: "You are a helpful assistant." },
    { role: MessageRole.USER, content: "Fix the login bug in auth.ts" },
    {
      role: MessageRole.ASSISTANT,
      content: "I found the issue in the authentication handler.",
      toolCalls: [
        { name: "read_file", arguments: { path: "src/auth.ts" }, callId: "tc1" },
      ],
    },
    { role: MessageRole.TOOL, content: "function login() { ... }", toolCallId: "tc1" },
  ];
}

// ─── Tests ───────────────────────────────────────────────────

describe("compaction-judge — judgeCompactionQuality", () => {
  it("returns null on provider error (graceful degradation)", async () => {
    const provider = throwingProvider();
    const result = await judgeCompactionQuality(
      provider, "pre-compaction summary", makeMessages(), null,
    );
    expect(result).toBeNull();
  });

  it("returns quality_loss=0.0 when all context preserved", async () => {
    const provider = mockProvider(
      '{"quality_loss": 0.0, "missing_context": [], "recommendation": "No action needed"}',
    );
    const result = await judgeCompactionQuality(
      provider, "Plan: step 1 complete, working on step 2", makeMessages(), null,
    );
    expect(result).not.toBeNull();
    expect(result!.quality_loss).toBe(0.0);
    expect(result!.missing_context).toHaveLength(0);
  });

  it("returns high quality_loss when plan/files missing from post-compaction", async () => {
    const provider = mockProvider(
      '{"quality_loss": 0.8, "missing_context": ["Current plan step context", "Modified file list"], "recommendation": "Re-read the plan and list of modified files"}',
    );
    const result = await judgeCompactionQuality(
      provider, "Plan: step 3/5 in progress. Modified: src/auth.ts, src/login.ts",
      [{ role: MessageRole.SYSTEM, content: "Compacted context" }],
      null,
    );
    expect(result).not.toBeNull();
    expect(result!.quality_loss).toBeGreaterThanOrEqual(0.6);
    expect(result!.missing_context.length).toBeGreaterThan(0);
  });

  it("provides recommendation for gap compensation", async () => {
    const provider = mockProvider(
      '{"quality_loss": 0.7, "missing_context": ["Error context being debugged"], "recommendation": "Inject the error message from the failing test into context"}',
    );
    const result = await judgeCompactionQuality(
      provider, "Debugging test failure in auth.test.ts",
      makeMessages(), null,
    );
    expect(result).not.toBeNull();
    expect(result!.recommendation).toContain("error message");
  });

  it("includes session state context when provided", async () => {
    const provider = mockProvider(
      '{"quality_loss": 0.1, "missing_context": [], "recommendation": "No action needed"}',
    );
    const state = new SessionState();
    state.recordModifiedFile("src/auth.ts");

    await judgeCompactionQuality(
      provider, "summary", makeMessages(), state,
    );

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const messages = chatCall[0] as Array<{ content: string | null }>;
    const userMsg = messages.find((m) => m.content?.includes("Modified files"));
    expect(userMsg).toBeDefined();
  });
});

describe("compaction-judge — buildPreCompactionSummary", () => {
  it("includes plan step info from session state", () => {
    const state = new SessionState();
    state.setPlan([
      { description: "Setup project", status: "completed" },
      { description: "Implement feature", status: "in_progress" },
    ]);
    const messages: Message[] = [
      { role: MessageRole.USER, content: "Build the auth system" },
    ];

    const summary = buildPreCompactionSummary(state, messages, 10);
    expect(summary).toContain("Active plan step: Implement feature");
    expect(summary).toContain("Iteration: 10");
  });

  it("includes modified files", () => {
    const state = new SessionState();
    state.recordModifiedFile("src/foo.ts");
    state.recordModifiedFile("src/bar.ts");

    const summary = buildPreCompactionSummary(state, [], 5);
    expect(summary).toContain("Modified files: src/foo.ts, src/bar.ts");
  });

  it("includes last user message (truncated)", () => {
    const longMsg = "x".repeat(600);
    const messages: Message[] = [
      { role: MessageRole.USER, content: longMsg },
    ];

    const summary = buildPreCompactionSummary(null, messages, 1);
    expect(summary.length).toBeLessThan(longMsg.length);
    expect(summary).toContain("Last user message:");
  });

  it("includes findings count", () => {
    const state = new SessionState();
    state.addFinding("Bug", "details", 1);
    state.addFinding("Issue", "more details", 2);

    const summary = buildPreCompactionSummary(state, [], 5);
    expect(summary).toContain("Findings: 2");
  });

  it("handles null session state", () => {
    const summary = buildPreCompactionSummary(null, [], 1);
    expect(summary).toContain("Iteration: 1");
  });
});
