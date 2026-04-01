import { describe, it, expect } from "vitest";
import { classifyProviderError, stripNullArgs, resolveCapabilities, convertMessages } from "./shared.js";
import {
  ProviderError,
  RateLimitError,
  ProviderConnectionError,
  OverloadedError,
  MessageRole,
} from "@devagent/runtime";

// ─── classifyProviderError ──────────────────────────────────

describe("classifyProviderError", () => {
  it("classifies 429 as RateLimitError", () => {
    const err = { status: 429, message: "Too Many Requests" };
    const result = classifyProviderError(err, "TestProvider");

    expect(result).toBeInstanceOf(RateLimitError);
    expect(result.message).toContain("TestProvider");
    expect(result.message).toContain("rate limited");
  });

  it("extracts retryAfterMs from headers on 429", () => {
    const err = {
      status: 429,
      message: "rate limited",
      headers: { "retry-after": "2.5" },
    };
    const result = classifyProviderError(err, "Test") as RateLimitError;

    expect(result).toBeInstanceOf(RateLimitError);
    expect(result.retryAfterMs).toBe(2500);
  });

  it("extracts retryAfterMs from responseHeaders", () => {
    const err = {
      status: 429,
      message: "rate limited",
      responseHeaders: { "Retry-After": "1" },
    };
    const result = classifyProviderError(err, "Test") as RateLimitError;

    expect(result).toBeInstanceOf(RateLimitError);
    expect(result.retryAfterMs).toBe(1000);
  });

  it("classifies 529 as OverloadedError", () => {
    const err = { status: 529, message: "Overloaded" };
    const result = classifyProviderError(err, "TestProvider");

    expect(result).toBeInstanceOf(OverloadedError);
    expect(result.message).toContain("overloaded");
  });

  it("classifies connection errors as ProviderConnectionError", () => {
    const cases = [
      "ECONNRESET",
      "ECONNREFUSED",
      "socket hang up",
      "fetch failed",
      "network error",
      "connection reset",
    ];

    for (const msg of cases) {
      const err = new Error(msg);
      const result = classifyProviderError(err, "Test");
      expect(result).toBeInstanceOf(ProviderConnectionError);
    }
  });

  it("classifies generic errors as ProviderError", () => {
    const err = { message: "something went wrong" };
    const result = classifyProviderError(err, "TestProvider");

    expect(result).toBeInstanceOf(ProviderError);
    expect(result.message).toContain("TestProvider API error");
  });

  it("returns existing ProviderError as-is", () => {
    const original = new ProviderError("already classified");
    const result = classifyProviderError(original, "Test");

    expect(result).toBe(original);
  });

  it("extracts status from nested cause object", () => {
    const err = {
      message: "wrapper error",
      cause: { statusCode: 429 },
    };
    const result = classifyProviderError(err, "Test");
    expect(result).toBeInstanceOf(RateLimitError);
  });

  it("extracts status from nested data object", () => {
    const err = {
      message: "wrapper error",
      data: { status: 529 },
    };
    const result = classifyProviderError(err, "Test");
    expect(result).toBeInstanceOf(OverloadedError);
  });

  it("extracts status from message string when no status field", () => {
    const err = { message: "Server returned 429 error" };
    const result = classifyProviderError(err, "Test");
    expect(result).toBeInstanceOf(RateLimitError);
  });

  it("handles statusCode field", () => {
    const err = { statusCode: 529, message: "overloaded" };
    const result = classifyProviderError(err, "Test");
    expect(result).toBeInstanceOf(OverloadedError);
  });

  it("handles null/undefined input gracefully", () => {
    const result = classifyProviderError(null, "Test");
    expect(result).toBeInstanceOf(ProviderError);

    const result2 = classifyProviderError(undefined, "Test");
    expect(result2).toBeInstanceOf(ProviderError);
  });
});

// ─── stripNullArgs ──────────────────────────────────────────

describe("stripNullArgs", () => {
  it("removes null-valued keys", () => {
    const result = stripNullArgs({ a: 1, b: null, c: "hello", d: null });
    expect(result).toEqual({ a: 1, c: "hello" });
  });

  it("preserves false, 0, and empty string", () => {
    const result = stripNullArgs({ a: false, b: 0, c: "" });
    expect(result).toEqual({ a: false, b: 0, c: "" });
  });

  it("returns empty object for all-null input", () => {
    const result = stripNullArgs({ a: null, b: null });
    expect(result).toEqual({});
  });
});

// ─── resolveCapabilities ────────────────────────────────────

describe("resolveCapabilities", () => {
  it("returns safe defaults when undefined", () => {
    const caps = resolveCapabilities(undefined);
    expect(caps.useResponsesApi).toBe(false);
    expect(caps.reasoning).toBe(false);
    expect(caps.supportsTemperature).toBe(true);
    expect(caps.defaultMaxTokens).toBe(4096);
  });

  it("respects explicit values", () => {
    const caps = resolveCapabilities({
      useResponsesApi: true,
      reasoning: true,
      supportsTemperature: false,
      defaultMaxTokens: 8192,
    });
    expect(caps.useResponsesApi).toBe(true);
    expect(caps.reasoning).toBe(true);
    expect(caps.supportsTemperature).toBe(false);
    expect(caps.defaultMaxTokens).toBe(8192);
  });
});

// ─── convertMessages ────────────────────────────────────────

describe("convertMessages", () => {
  it("converts basic user/assistant messages", () => {
    const messages = [
      { role: MessageRole.USER, content: "hello" },
      { role: MessageRole.ASSISTANT, content: "hi" },
    ];
    const result = convertMessages(messages);

    expect(result).toHaveLength(2);
    expect(result[0]).toEqual({ role: "user", content: "hello" });
    expect(result[1]).toEqual({ role: "assistant", content: "hi" });
  });

  it("converts system messages", () => {
    const messages = [{ role: MessageRole.SYSTEM, content: "You are helpful" }];
    const result = convertMessages(messages);
    expect(result[0]).toEqual({ role: "system", content: "You are helpful" });
  });

  it("converts tool call messages", () => {
    const messages = [
      {
        role: MessageRole.ASSISTANT,
        content: "Let me check",
        toolCalls: [
          { name: "read_file", arguments: { path: "/foo" }, callId: "tc1" },
        ],
      },
    ];
    const result = convertMessages(messages);

    expect(result).toHaveLength(1);
    const content = result[0]!.content as Array<{ type: string }>;
    expect(content).toHaveLength(2);
    expect(content[0]!.type).toBe("text");
    expect(content[1]!.type).toBe("tool-call");
  });

  it("converts tool result messages", () => {
    const messages = [
      { role: MessageRole.TOOL, content: "file contents", toolCallId: "tc1" },
    ];
    const result = convertMessages(messages);

    expect(result).toHaveLength(1);
    expect(result[0]!.role).toBe("tool");
  });
});
