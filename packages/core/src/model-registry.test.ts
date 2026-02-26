import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { describe, expect, it } from "vitest";
import { loadModelRegistry, lookupModelCapabilities } from "./model-registry.js";

describe("model registry", () => {
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
});
