import { describe, it, expect, vi } from "vitest";
import { MessageRole } from "@devagent/core";
import type { LLMProvider, Message, StreamChunk } from "@devagent/core";
import {
  collectStreamText,
  parseJudgeResponse,
  formatMessageForJudge,
  formatToolArgs,
  buildSessionStateContext,
  JUDGE_ARG_MAX_CHARS,
  JUDGE_RESULT_MAX_CHARS,
} from "./llm-judge.js";
import { SessionState } from "./session-state.js";

// ─── Mock helpers ────────────────────────────────────────────

function mockProvider(responseText: string): LLMProvider {
  return {
    id: "mock",
    chat: vi.fn(async function* (): AsyncIterable<StreamChunk> {
      yield { type: "text", content: responseText };
      yield { type: "done", content: "" };
    }),
    abort: vi.fn(),
  };
}

// ─── Tests ───────────────────────────────────────────────────

describe("llm-judge — formatMessageForJudge", () => {
  it("preserves tool call names and arguments", () => {
    const msg: Message = {
      role: MessageRole.ASSISTANT,
      content: null,
      toolCalls: [
        { name: "search_files", arguments: { pattern: "login", path: "/src" }, callId: "tc1" },
      ],
    };
    const result = formatMessageForJudge(msg);
    expect(result).toContain("[assistant]");
    expect(result).toContain("tool_call: search_files");
    expect(result).toContain("pattern=login");
    expect(result).toContain("path=/src");
  });

  it("truncates long content to JUDGE_RESULT_MAX_CHARS", () => {
    const longContent = "x".repeat(JUDGE_RESULT_MAX_CHARS + 100);
    const msg: Message = { role: MessageRole.USER, content: longContent };
    const result = formatMessageForJudge(msg);
    // Should contain truncated version with "..."
    expect(result).toContain("...");
    // The full long content should NOT appear
    expect(result).not.toContain(longContent);
  });

  it("formats tool result messages with callId", () => {
    const msg: Message = {
      role: MessageRole.TOOL,
      content: "Found 3 matches",
      toolCallId: "tc1",
    };
    const result = formatMessageForJudge(msg);
    expect(result).toContain("[tool]");
    expect(result).toContain("tool_result [tc1]");
    expect(result).toContain("Found 3 matches");
  });
});

describe("llm-judge — formatToolArgs", () => {
  it("truncates long argument values", () => {
    const longVal = "y".repeat(JUDGE_ARG_MAX_CHARS + 50);
    const result = formatToolArgs({ key: longVal });
    expect(result).toContain("key=");
    expect(result).toContain("...");
    expect(result.length).toBeLessThan(JUDGE_ARG_MAX_CHARS + 20); // key= + truncated + ...
  });

  it("returns empty string for no arguments", () => {
    expect(formatToolArgs({})).toBe("");
  });

  it("formats multiple key=value pairs", () => {
    const result = formatToolArgs({ a: "1", b: "2" });
    expect(result).toBe("a=1, b=2");
  });
});

describe("llm-judge — parseJudgeResponse", () => {
  it("strips markdown fences and parses JSON", () => {
    const raw = '```json\n{"score": 0.5, "note": "ok"}\n```';
    const result = parseJudgeResponse<{ score: number; note: string }>(raw);
    expect(result).toEqual({ score: 0.5, note: "ok" });
  });

  it("parses plain JSON without fences", () => {
    const raw = '{"score": 1.0}';
    const result = parseJudgeResponse<{ score: number }>(raw);
    expect(result).toEqual({ score: 1.0 });
  });

  it("throws on invalid JSON", () => {
    expect(() => parseJudgeResponse("not json at all")).toThrow();
  });

  it("handles JSON with leading/trailing whitespace", () => {
    const raw = '  \n  {"ok": true}\n  ';
    const result = parseJudgeResponse<{ ok: boolean }>(raw);
    expect(result).toEqual({ ok: true });
  });
});

describe("llm-judge — collectStreamText", () => {
  it("collects text chunks from provider", async () => {
    const provider = mockProvider("hello world");
    const messages: Message[] = [{ role: MessageRole.USER, content: "test" }];
    const result = await collectStreamText(provider, messages);
    expect(result).toBe("hello world");
  });

  it("ignores non-text chunks", async () => {
    const provider: LLMProvider = {
      id: "mock",
      chat: vi.fn(async function* (): AsyncIterable<StreamChunk> {
        yield { type: "thinking", content: "hmm" };
        yield { type: "text", content: "actual" };
        yield { type: "done", content: "" };
      }),
      abort: vi.fn(),
    };
    const messages: Message[] = [{ role: MessageRole.USER, content: "test" }];
    const result = await collectStreamText(provider, messages);
    expect(result).toBe("actual");
  });
});

describe("llm-judge — buildSessionStateContext", () => {
  it("includes plan progress, files, and findings", () => {
    const state = new SessionState();
    state.setPlan([
      { description: "Step 1: setup", status: "completed" },
      { description: "Step 2: implement", status: "in_progress" },
      { description: "Step 3: test", status: "pending" },
    ]);
    state.recordModifiedFile("src/foo.ts");
    state.recordModifiedFile("src/bar.ts");
    state.addFinding("Bug found", "null pointer in foo()", 3);

    const result = buildSessionStateContext(state);
    expect(result).toContain("Plan progress: 1/3 steps completed");
    expect(result).toContain("[completed] Step 1: setup");
    expect(result).toContain("[in_progress] Step 2: implement");
    expect(result).toContain("[pending] Step 3: test");
    expect(result).toContain("Modified files: 2");
    expect(result).toContain("Findings: 1");
  });

  it("handles null session state", () => {
    const result = buildSessionStateContext(null);
    expect(result).toBe("No session state available.");
  });

  it("handles empty plan", () => {
    const state = new SessionState();
    const result = buildSessionStateContext(state);
    expect(result).toContain("No plan set.");
  });
});
