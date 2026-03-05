import { describe, it, expect, vi } from "vitest";
import { MessageRole } from "@devagent/core";
import type { LLMProvider, Message, StreamChunk } from "@devagent/core";
import { classifyError } from "./error-judge.js";

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

function makeRecentContext(): string {
  return "[assistant]\n  tool_call: edit_file(path=src/auth.ts)\n\n[tool]\n  tool_result [tc1]\n  SyntaxError: Unexpected token";
}

// ─── Tests ───────────────────────────────────────────────────

describe("error-judge — classifyError", () => {
  it("returns null on provider error", async () => {
    const provider = throwingProvider();
    const result = await classifyError(
      provider, "run_command", { command: "bun test" },
      "SyntaxError: Unexpected token in auth.ts at line 42",
      makeRecentContext(),
    );
    expect(result).toBeNull();
  });

  it("classifies syntax error as code_error", async () => {
    const provider = mockProvider(
      '{"category": "code_error", "severity": "medium", "recovery_hint": "Fix the syntax error in auth.ts at line 42"}',
    );
    const result = await classifyError(
      provider, "run_command", { command: "bun test" },
      "SyntaxError: Unexpected token '}' at auth.ts:42. Expected expression.",
      makeRecentContext(),
    );
    expect(result).not.toBeNull();
    expect(result!.category).toBe("code_error");
    expect(result!.severity).toBe("medium");
    expect(result!.recovery_hint).toContain("syntax error");
  });

  it("classifies access denied as permission", async () => {
    const provider = mockProvider(
      '{"category": "permission", "severity": "high", "recovery_hint": "Ask the user to grant write access to /etc/config"}',
    );
    const result = await classifyError(
      provider, "write_file", { path: "/etc/config" },
      "EACCES: permission denied, open '/etc/config' - you don't have write access to this file",
      makeRecentContext(),
    );
    expect(result).not.toBeNull();
    expect(result!.category).toBe("permission");
  });

  it("classifies invalid path as tool_misuse", async () => {
    const provider = mockProvider(
      '{"category": "tool_misuse", "severity": "low", "recovery_hint": "Check file path - the file does not exist at the specified location"}',
    );
    const result = await classifyError(
      provider, "read_file", { path: "/nonexistent/file.ts" },
      "ENOENT: no such file or directory, open '/nonexistent/file.ts' - verify the file path is correct",
      makeRecentContext(),
    );
    expect(result).not.toBeNull();
    expect(result!.category).toBe("tool_misuse");
  });

  it("returns recovery_hint with actionable guidance", async () => {
    const provider = mockProvider(
      '{"category": "infrastructure", "severity": "high", "recovery_hint": "Network timeout - wait and retry, or check if the service is available"}',
    );
    const result = await classifyError(
      provider, "fetch_url", { url: "https://api.example.com" },
      "ETIMEDOUT: connection timed out after 30000ms waiting for api.example.com to respond",
      makeRecentContext(),
    );
    expect(result).not.toBeNull();
    expect(result!.category).toBe("infrastructure");
    expect(result!.recovery_hint).toBeTruthy();
    expect(result!.recovery_hint.length).toBeGreaterThan(10);
  });

  it("includes tool name and args in judge context", async () => {
    const provider = mockProvider(
      '{"category": "code_error", "severity": "low", "recovery_hint": "fix it"}',
    );
    await classifyError(
      provider, "run_command", { command: "bun test", cwd: "/app" },
      "Test failed: expected true to be false in auth.test.ts. The assertion on line 42 does not match.",
      makeRecentContext(),
    );
    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const messages = chatCall[0] as Array<{ content: string | null }>;
    const userMsg = messages.find((m) => m.content?.includes("run_command"));
    expect(userMsg).toBeDefined();
    expect(userMsg!.content).toContain("bun test");
  });
});
