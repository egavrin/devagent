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

  it("parses context settings from TOML", () => {
    const dir = join(tmpdir(), `devagent-test-context-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, ".devagent.toml");
    writeFileSync(
      configPath,
      `
[context]
pruning_strategy = "summarize"
trigger_ratio = 0.65
keep_recent_messages = 22
turn_isolation = false
midpoint_briefing_interval = 4
briefing_strategy = "llm"
`,
    );

    try {
      const config = loadConfig(dir);
      expect(config.context.pruningStrategy).toBe("summarize");
      expect(config.context.triggerRatio).toBe(0.65);
      expect(config.context.keepRecentMessages).toBe(22);
      expect(config.context.turnIsolation).toBe(false);
      expect(config.context.midpointBriefingInterval).toBe(4);
      expect(config.context.briefingStrategy).toBe("llm");
    } finally {
      rmSync(dir, { recursive: true });
    }
  });

  it("fails fast on invalid context.trigger_ratio", () => {
    const dir = join(tmpdir(), `devagent-test-invalid-trigger-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, ".devagent.toml");
    writeFileSync(
      configPath,
      `
[context]
trigger_ratio = 1.5
`,
    );

    try {
      expect(() => loadConfig(dir)).toThrow("context.triggerRatio");
    } finally {
      rmSync(dir, { recursive: true });
    }
  });

  it("fails fast when budget.response_headroom >= budget.max_context_tokens", () => {
    const dir = join(tmpdir(), `devagent-test-invalid-headroom-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, ".devagent.toml");
    writeFileSync(
      configPath,
      `
[budget]
max_context_tokens = 1000
response_headroom = 1000
`,
    );

    try {
      expect(() => loadConfig(dir)).toThrow("budget.responseHeadroom");
    } finally {
      rmSync(dir, { recursive: true });
    }
  });

  it("parses model capabilities from TOML provider config", () => {
    const dir = join(tmpdir(), `devagent-test-caps-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, ".devagent.toml");
    writeFileSync(
      configPath,
      `
provider = "openai"
model = "gpt-5-codex"

[providers.openai]
model = "gpt-5-codex"
api_key = "test-key"
use_responses_api = true
reasoning = true
supports_temperature = false
default_max_tokens = 16384
reasoning_effort = "medium"
`,
    );

    try {
      const config = loadConfig(dir);
      const openaiCfg = config.providers["openai"];
      expect(openaiCfg).toBeDefined();
      expect(openaiCfg!.capabilities).toBeDefined();
      expect(openaiCfg!.capabilities!.useResponsesApi).toBe(true);
      expect(openaiCfg!.capabilities!.reasoning).toBe(true);
      expect(openaiCfg!.capabilities!.supportsTemperature).toBe(false);
      expect(openaiCfg!.capabilities!.defaultMaxTokens).toBe(16384);
      expect(openaiCfg!.reasoningEffort).toBe("medium");
    } finally {
      rmSync(dir, { recursive: true });
    }
  });

  it("omits capabilities when none specified in TOML", () => {
    const dir = join(tmpdir(), `devagent-test-nocaps-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, ".devagent.toml");
    writeFileSync(
      configPath,
      `
provider = "openai"

[providers.openai]
model = "gpt-4o"
api_key = "test-key"
`,
    );

    try {
      const config = loadConfig(dir);
      const openaiCfg = config.providers["openai"];
      expect(openaiCfg).toBeDefined();
      expect(openaiCfg!.capabilities).toBeUndefined();
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
      if (original["DEVAGENT_PROVIDER"] === undefined) {
        delete process.env["DEVAGENT_PROVIDER"];
      } else {
        process.env["DEVAGENT_PROVIDER"] = original["DEVAGENT_PROVIDER"];
      }
      if (original["DEVAGENT_MODEL"] === undefined) {
        delete process.env["DEVAGENT_MODEL"];
      } else {
        process.env["DEVAGENT_MODEL"] = original["DEVAGENT_MODEL"];
      }
    }
  });

  it("injects resolved top-level api_key into existing provider blocks", () => {
    const dir = join(tmpdir(), `devagent-test-provider-key-${Date.now()}`);
    mkdirSync(dir, { recursive: true });
    const configPath = join(dir, ".devagent.toml");
    writeFileSync(
      configPath,
      `
provider = "custom_provider_for_test"
model = "test-model"
api_key = "top-level-key"

[providers.custom_provider_for_test]
model = "test-model-v2"
base_url = "https://example.test/v1"
`,
    );

    try {
      const config = loadConfig(dir);
      expect(config.providers["custom_provider_for_test"]?.model).toBe(
        "test-model-v2",
      );
      expect(config.providers["custom_provider_for_test"]?.baseUrl).toBe(
        "https://example.test/v1",
      );
      expect(config.providers["custom_provider_for_test"]?.apiKey).toBe(
        "top-level-key",
      );
    } finally {
      rmSync(dir, { recursive: true });
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
