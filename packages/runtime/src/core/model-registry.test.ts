import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it, beforeEach } from "vitest";

import {
  loadModelRegistry,
  lookupModelCapabilities,
  lookupModelEntry,
  lookupModelPricing,
  getProvidersForModel,
  isModelRegisteredForProvider,
  _resetRegistryForTesting,
} from "./model-registry.js";

beforeEach(() => {
  _resetRegistryForTesting();
});

describe("model registry custom files", () => {
  it("uses default_max_tokens when present, falling back to response_headroom", () => {
    const modelWithDefault = "zz-test-model-default-max-tokens";
    const modelWithFallback = "zz-test-model-fallback-headroom";
    const dir = mkdtempSync(join(tmpdir(), "devagent-model-registry-"));

    try {
      writeFileSync(
        join(dir, "test-models.toml"),
        `provider = "openai"

["${modelWithDefault}"]
context_window = 1000
response_headroom = 2222
default_max_tokens = 7777
supports_temperature = false

["${modelWithFallback}"]
context_window = 1000
response_headroom = 3333
supports_temperature = true
`,
        "utf-8",
      );

      loadModelRegistry(undefined, [dir]);

      const capsDefault = lookupModelCapabilities(modelWithDefault);
      expect(capsDefault).toBeDefined();
      expect(capsDefault?.defaultMaxTokens).toBe(7777);

      const capsFallback = lookupModelCapabilities(modelWithFallback);
      expect(capsFallback).toBeDefined();
      expect(capsFallback?.defaultMaxTokens).toBe(3333);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });

  it("parses pricing fields from TOML", () => {
    const modelName = "zz-test-model-with-pricing";
    const modelNoPricing = "zz-test-model-no-pricing";
    const dir = mkdtempSync(join(tmpdir(), "devagent-model-pricing-"));

    try {
      writeFileSync(
        join(dir, "pricing-test.toml"),
        `provider = "test"

["${modelName}"]
context_window = 1000
response_headroom = 2000
supports_temperature = true
input_price_per_million = 2.50
output_price_per_million = 10
cache_hit_input_price_per_million = 1.25

["${modelNoPricing}"]
context_window = 1000
response_headroom = 2000
supports_temperature = true
`,
        "utf-8",
      );

      loadModelRegistry(undefined, [dir]);

      const pricing = lookupModelPricing(modelName);
      expect(pricing).toBeDefined();
      expect(pricing!.inputPricePerMillion).toBe(2.50);
      expect(pricing!.outputPricePerMillion).toBe(10);
      expect(pricing!.cacheHitInputPricePerMillion).toBe(1.25);

      const noPricing = lookupModelPricing(modelNoPricing);
      expect(noPricing).toBeUndefined();
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

describe("bundled model registry", () => {
  it("loads current DeepSeek V4 model capabilities and pricing", () => {
    const modelsDir = join(import.meta.dirname ?? new URL(".", import.meta.url).pathname, "../../../../models");
    loadModelRegistry(undefined, [modelsDir]);

    expectDeepSeekV4Model("deepseek-v4-flash", {
      inputPricePerMillion: 0.14,
      cacheHitInputPricePerMillion: 0.028,
      outputPricePerMillion: 0.28,
    });
    expectDeepSeekV4Model("deepseek-v4-pro", {
      inputPricePerMillion: 1.74,
      cacheHitInputPricePerMillion: 0.145,
      outputPricePerMillion: 3.48,
    });
  });
});

describe("model registry provider lookup", () => {
  it("tracks the same model name across multiple providers", () => {
    const modelName = "zz-shared-model";
    const dir = mkdtempSync(join(tmpdir(), "devagent-model-shared-"));

    try {
      writeFileSync(
        join(dir, "openai.toml"),
        `provider = "openai"

["${modelName}"]
context_window = 1000
response_headroom = 2000
supports_temperature = false
`,
        "utf-8",
      );
      writeFileSync(
        join(dir, "chatgpt.toml"),
        `provider = "chatgpt"

["${modelName}"]
context_window = 3000
response_headroom = 4000
supports_temperature = false
`,
        "utf-8",
      );

      loadModelRegistry(undefined, [dir]);

      expect(getProvidersForModel(modelName)).toEqual(["chatgpt", "openai"]);
      expect(isModelRegisteredForProvider("openai", modelName)).toBe(true);
      expect(isModelRegisteredForProvider("chatgpt", modelName)).toBe(true);
      expect(lookupModelEntry(modelName, "openai")?.contextWindow).toBe(1000);
      expect(lookupModelEntry(modelName, "chatgpt")?.contextWindow).toBe(3000);
      expect(lookupModelCapabilities(modelName, "chatgpt")?.defaultMaxTokens).toBe(4000);
    } finally {
      rmSync(dir, { recursive: true, force: true });
    }
  });
});

function expectDeepSeekV4Model(
  model: string,
  pricing: {
    readonly inputPricePerMillion: number;
    readonly cacheHitInputPricePerMillion: number;
    readonly outputPricePerMillion: number;
  },
): void {
  const entry = lookupModelEntry(model, "deepseek");
  expect(entry?.contextWindow).toBe(1_000_000);
  expect(entry?.responseHeadroom).toBe(384_000);
  expect(entry?.capabilities.reasoning).toBe(true);
  expect(entry?.capabilities.supportsTemperature).toBe(false);
  expect(entry?.pricing).toEqual(pricing);
}
