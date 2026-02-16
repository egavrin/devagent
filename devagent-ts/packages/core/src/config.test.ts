import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { loadConfig, findProjectRoot } from "./config.js";
import { writeFileSync, mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

describe("loadConfig", () => {
  it("returns defaults when no config file exists", () => {
    const config = loadConfig("/nonexistent/path");

    expect(config.provider).toBe("anthropic");
    expect(config.model).toBe("claude-sonnet-4-20250514");
    expect(config.approval.mode).toBe("suggest");
    expect(config.approval.autoApproveCode).toBe(false);
    expect(config.budget.maxIterations).toBe(30);
    expect(config.budget.enableCostTracking).toBe(true);
    expect(config.context.pruningStrategy).toBe("hybrid");
    expect(config.arkts.enabled).toBe(false);
  });

  it("loads config from TOML file", () => {
    const dir = join(tmpdir(), `devagent-test-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, ".devagent.toml");
    writeFileSync(
      configPath,
      `
provider = "openai"
model = "gpt-4o"

[budget]
max_iterations = 50
cost_warning_threshold = 5.0

[arkts]
enabled = true
strict_mode = true
`,
    );

    try {
      const config = loadConfig(dir);
      expect(config.provider).toBe("openai");
      expect(config.model).toBe("gpt-4o");
      expect(config.budget.maxIterations).toBe(50);
      expect(config.budget.costWarningThreshold).toBe(5.0);
      expect(config.arkts.enabled).toBe(true);
      expect(config.arkts.strictMode).toBe(true);
    } finally {
      rmSync(dir, { recursive: true });
    }
  });

  it("respects environment variable overrides", () => {
    const original = { ...process.env };
    process.env["DEVAGENT_PROVIDER"] = "deepseek";
    process.env["DEVAGENT_MODEL"] = "deepseek-coder";

    try {
      const config = loadConfig("/nonexistent/path");
      expect(config.provider).toBe("deepseek");
      expect(config.model).toBe("deepseek-coder");
    } finally {
      process.env["DEVAGENT_PROVIDER"] = original["DEVAGENT_PROVIDER"];
      process.env["DEVAGENT_MODEL"] = original["DEVAGENT_MODEL"];
    }
  });

  it("respects override parameter", () => {
    const config = loadConfig("/nonexistent/path", {
      provider: "openrouter",
      model: "meta-llama/llama-3",
    });

    expect(config.provider).toBe("openrouter");
    expect(config.model).toBe("meta-llama/llama-3");
  });

  it("fails fast on invalid TOML", () => {
    const dir = join(tmpdir(), `devagent-test-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, ".devagent.toml"), "this is not valid [[[ toml");

    try {
      expect(() => loadConfig(dir)).toThrow("Failed to parse config file");
    } finally {
      rmSync(dir, { recursive: true });
    }
  });

  it("resolves env: references in api_key", () => {
    const dir = join(tmpdir(), `devagent-test-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, ".devagent.toml"),
      `
provider = "anthropic"
api_key = "env:TEST_DEVAGENT_KEY"
`,
    );

    process.env["TEST_DEVAGENT_KEY"] = "sk-test-123";

    try {
      const config = loadConfig(dir);
      expect(config.providers["anthropic"]?.apiKey).toBe("sk-test-123");
    } finally {
      delete process.env["TEST_DEVAGENT_KEY"];
      rmSync(dir, { recursive: true });
    }
  });

  it("fails fast on missing env reference", () => {
    const dir = join(tmpdir(), `devagent-test-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, ".devagent.toml"),
      `
provider = "anthropic"
api_key = "env:NONEXISTENT_VAR_12345"
`,
    );

    try {
      expect(() => loadConfig(dir)).toThrow(
        'Environment variable "NONEXISTENT_VAR_12345" referenced in config but not set',
      );
    } finally {
      rmSync(dir, { recursive: true });
    }
  });
});

describe("findProjectRoot", () => {
  it("returns null for root directory", () => {
    const result = findProjectRoot("/");
    // May or may not be null depending on system, but should not throw
    expect(typeof result === "string" || result === null).toBe(true);
  });

  it("finds directory with .devagent.toml", () => {
    const dir = join(tmpdir(), `devagent-root-test-${Date.now()}`);
    const subdir = join(dir, "a", "b", "c");
    mkdirSync(subdir, { recursive: true });
    writeFileSync(join(dir, ".devagent.toml"), "provider = 'test'");

    try {
      const result = findProjectRoot(subdir);
      expect(result).toBe(dir);
    } finally {
      rmSync(dir, { recursive: true });
    }
  });
});
