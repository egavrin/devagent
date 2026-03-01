import { describe, it, expect } from "vitest";
import { createOpenAIProvider, resolveCapabilities, stripNullArgs } from "./openai.js";
import type { ProviderConfig, ModelCapabilities } from "@devagent/core";

describe("createOpenAIProvider", () => {
  it("creates a provider without API key (for local endpoints)", () => {
    const config: ProviderConfig = { model: "llama3" };
    const provider = createOpenAIProvider(config);
    expect(provider.id).toBe("openai");
    expect(typeof provider.chat).toBe("function");
    expect(typeof provider.abort).toBe("function");
  });

  it("creates a provider with valid config", () => {
    const config: ProviderConfig = {
      model: "gpt-4o",
      apiKey: "test-key",
    };
    const provider = createOpenAIProvider(config);
    expect(provider.id).toBe("openai");
    expect(typeof provider.chat).toBe("function");
    expect(typeof provider.abort).toBe("function");
  });

  it("accepts a custom baseUrl", () => {
    const config: ProviderConfig = {
      model: "llama3",
      baseUrl: "http://localhost:11434/v1",
    };
    const provider = createOpenAIProvider(config);
    expect(provider.id).toBe("openai");
  });
});

describe("resolveCapabilities", () => {
  it("returns safe defaults when no capabilities configured", () => {
    const caps = resolveCapabilities(undefined);
    expect(caps.useResponsesApi).toBe(false);
    expect(caps.reasoning).toBe(false);
    expect(caps.supportsTemperature).toBe(true);
    expect(caps.defaultMaxTokens).toBe(4096);
  });

  it("uses explicit config for reasoning model", () => {
    const explicit: ModelCapabilities = {
      useResponsesApi: true,
      reasoning: true,
      supportsTemperature: false,
      defaultMaxTokens: 16384,
    };
    const caps = resolveCapabilities(explicit);
    expect(caps.useResponsesApi).toBe(true);
    expect(caps.reasoning).toBe(true);
    expect(caps.supportsTemperature).toBe(false);
    expect(caps.defaultMaxTokens).toBe(16384);
  });

  it("partial config fills gaps with safe defaults", () => {
    const explicit: ModelCapabilities = { defaultMaxTokens: 8192 };
    const caps = resolveCapabilities(explicit);
    expect(caps.useResponsesApi).toBe(false);
    expect(caps.reasoning).toBe(false);
    expect(caps.supportsTemperature).toBe(true);
    expect(caps.defaultMaxTokens).toBe(8192);
  });

  it("explicit false is preserved", () => {
    const explicit: ModelCapabilities = {
      reasoning: false,
      supportsTemperature: false,
    };
    const caps = resolveCapabilities(explicit);
    expect(caps.reasoning).toBe(false);
    expect(caps.supportsTemperature).toBe(false);
  });

  it("supports high token limits for large-context models", () => {
    const explicit: ModelCapabilities = { defaultMaxTokens: 131072 };
    const caps = resolveCapabilities(explicit);
    expect(caps.defaultMaxTokens).toBe(131072);
  });
});

describe("stripNullArgs", () => {
  it("removes null-valued keys (OpenAI strict schema sends null for unused optionals)", () => {
    const args = {
      path: "src/x.cpp",
      search: "std.core.",
      replace: "std:core.",
      replacements: null,
      all: null,
      expected_replacements: null,
    };
    const cleaned = stripNullArgs(args);
    expect(cleaned).toEqual({
      path: "src/x.cpp",
      search: "std.core.",
      replace: "std:core.",
    });
    expect("replacements" in cleaned).toBe(false);
    expect("all" in cleaned).toBe(false);
  });

  it("preserves non-null values including false, 0, and empty string", () => {
    const args = {
      path: "file.ts",
      all: false,
      count: 0,
      note: "",
      data: [1, 2],
    };
    const cleaned = stripNullArgs(args);
    expect(cleaned).toEqual(args);
  });

  it("returns empty object when all values are null", () => {
    const cleaned = stripNullArgs({ a: null, b: null });
    expect(cleaned).toEqual({});
  });
});
