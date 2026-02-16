import { describe, it, expect } from "vitest";
import { createOpenAIProvider } from "./openai.js";
import type { ProviderConfig } from "@devagent/core";

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
