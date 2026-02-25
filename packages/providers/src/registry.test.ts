import { describe, it, expect } from "vitest";
import { ProviderRegistry } from "./registry.js";
import type { LLMProvider, ProviderConfig, StreamChunk } from "@devagent/core";

function makeMockProvider(id: string): LLMProvider {
  return {
    id,
    async *chat(): AsyncIterable<StreamChunk> {
      yield { type: "text", content: "hello" };
      yield { type: "done", content: "" };
    },
    abort(): void {},
  };
}

function makeMockFactory(id: string) {
  return (_config: ProviderConfig) => makeMockProvider(id);
}

describe("ProviderRegistry", () => {
  it("registers and retrieves a provider", () => {
    const registry = new ProviderRegistry();
    registry.register("test", makeMockFactory("test"));

    const provider = registry.get("test", { model: "test-model" });
    expect(provider.id).toBe("test");
  });

  it("caches provider instances", () => {
    const registry = new ProviderRegistry();
    registry.register("test", makeMockFactory("test"));

    const config: ProviderConfig = { model: "test-model" };
    const p1 = registry.get("test", config);
    const p2 = registry.get("test", config);
    expect(p1).toBe(p2); // same instance
  });

  it("throws on unknown provider", () => {
    const registry = new ProviderRegistry();
    registry.register("anthropic", makeMockFactory("anthropic"));

    expect(() => registry.get("unknown", { model: "m" })).toThrow(
      'Unknown provider "unknown"',
    );
  });

  it("lists registered providers", () => {
    const registry = new ProviderRegistry();
    registry.register("anthropic", makeMockFactory("anthropic"));
    registry.register("openai", makeMockFactory("openai"));

    const list = registry.list();
    expect(list).toContain("anthropic");
    expect(list).toContain("openai");
    expect(list.length).toBe(2);
  });

  it("has() checks registration", () => {
    const registry = new ProviderRegistry();
    registry.register("anthropic", makeMockFactory("anthropic"));

    expect(registry.has("anthropic")).toBe(true);
    expect(registry.has("unknown")).toBe(false);
  });

  it("clear() removes cached instances", () => {
    const registry = new ProviderRegistry();
    let callCount = 0;
    registry.register("test", (config) => {
      callCount++;
      return makeMockProvider("test");
    });

    registry.get("test", { model: "m" });
    expect(callCount).toBe(1);

    registry.get("test", { model: "m" }); // cached
    expect(callCount).toBe(1);

    registry.clear();

    registry.get("test", { model: "m" }); // re-created
    expect(callCount).toBe(2);
  });
});
