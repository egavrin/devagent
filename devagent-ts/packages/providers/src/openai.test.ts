import { describe, it, expect } from "vitest";
import { createOpenAIProvider, resolveCapabilities } from "./openai.js";
import type { ProviderConfig, ModelCapabilities } from "@devagent/core";

describe("createOpenAIProvider", () => {
  it("throws when no API key provided", () => {
    const config: ProviderConfig = { model: "gpt-4o" };
    expect(() => createOpenAIProvider(config)).toThrow(
      "requires an API key",
    );
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
