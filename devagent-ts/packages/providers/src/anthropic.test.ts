import { describe, it, expect } from "vitest";
import { createAnthropicProvider } from "./anthropic.js";
import type { ProviderConfig } from "@devagent/core";

describe("createAnthropicProvider", () => {
  it("throws when no API key provided", () => {
    const config: ProviderConfig = { model: "claude-sonnet-4-20250514" };
    expect(() => createAnthropicProvider(config)).toThrow(
      "requires an API key",
    );
  });

  it("creates a provider with valid config", () => {
    const config: ProviderConfig = {
      model: "claude-sonnet-4-20250514",
      apiKey: "test-key",
    };
    const provider = createAnthropicProvider(config);
    expect(provider.id).toBe("anthropic");
    expect(typeof provider.chat).toBe("function");
    expect(typeof provider.abort).toBe("function");
  });
});
