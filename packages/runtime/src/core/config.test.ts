import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { loadConfig, findProjectRoot, DEFAULT_CONTEXT } from "./config.js";
import { writeFileSync, mkdirSync, rmSync, mkdtempSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

function writeHomeConfig(home: string, contents: string): string {
  const configDir = join(home, ".config", "devagent");
  mkdirSync(configDir, { recursive: true });
  const configPath = join(configDir, "config.toml");
  writeFileSync(configPath, contents);
  return configPath;
}

describe("loadConfig", () => {
  let originalHome: string | undefined;
  let tempHome: string;

  beforeEach(() => {
    tempHome = mkdtempSync(join(tmpdir(), "devagent-config-home-"));
    originalHome = process.env["HOME"];
    process.env["HOME"] = tempHome;
  });

  afterEach(() => {
    if (originalHome === undefined) {
      delete process.env["HOME"];
    } else {
      process.env["HOME"] = originalHome;
    }
    rmSync(tempHome, { recursive: true, force: true });
  });

  it("returns defaults when no config file exists", () => {
    const config = loadConfig("/nonexistent/path");

    expect(config.provider).toBe("anthropic");
    expect(config.model).toBe("claude-sonnet-4-20250514");
    expect(config.approval.mode).toBe("autopilot");
    expect(config.approval.approvalPolicy).toBe("never");
    expect(config.budget.maxIterations).toBe(0);
    expect(config.budget.enableCostTracking).toBe(true);
    expect(config.context.pruningStrategy).toBe("hybrid");
    expect(config.context.triggerRatio).toBe(0.9);
    expect(config.context.keepRecentMessages).toBe(40);
    expect(config.arkts.enabled).toBe(false);
  });

  it("loads config from TOML file", () => {
    writeHomeConfig(
      tempHome,
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

    const config = loadConfig("/nonexistent/path");
    expect(config.provider).toBe("openai");
    expect(config.model).toBe("gpt-4o");
    expect(config.budget.maxIterations).toBe(50);
    expect(config.budget.costWarningThreshold).toBe(5.0);
    expect(config.arkts.enabled).toBe(true);
    expect(config.arkts.strictMode).toBe(true);
  });

  it("fails fast on legacy [approval] config", () => {
    writeHomeConfig(
      tempHome,
      `
[approval]
mode = "suggest"
`,
    );

    expect(() => loadConfig("/nonexistent/path")).toThrow("The [approval] section has been removed");
  });

  it("parses context settings from TOML", () => {
    writeHomeConfig(
      tempHome,
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

    const config = loadConfig("/nonexistent/path");
    expect(config.context.pruningStrategy).toBe("summarize");
    expect(config.context.triggerRatio).toBe(0.65);
    expect(config.context.keepRecentMessages).toBe(22);
    expect(config.context.turnIsolation).toBe(false);
    expect(config.context.midpointBriefingInterval).toBe(4);
    expect(config.context.briefingStrategy).toBe("llm");
  });

  it("fails fast on invalid context.trigger_ratio", () => {
    writeHomeConfig(
      tempHome,
      `
[context]
trigger_ratio = 1.5
`,
    );

    expect(() => loadConfig("/nonexistent/path")).toThrow("context.triggerRatio");
  });

  it("fails fast when budget.response_headroom >= budget.max_context_tokens", () => {
    writeHomeConfig(
      tempHome,
      `
[budget]
max_context_tokens = 1000
response_headroom = 1000
`,
    );

    expect(() => loadConfig("/nonexistent/path")).toThrow("budget.responseHeadroom");
  });

  it("parses model capabilities from TOML provider config", () => {
    writeHomeConfig(
      tempHome,
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

    const config = loadConfig("/nonexistent/path");
    const openaiCfg = config.providers["openai"];
    expect(openaiCfg).toBeDefined();
    expect(openaiCfg!.capabilities).toBeDefined();
    expect(openaiCfg!.capabilities!.useResponsesApi).toBe(true);
    expect(openaiCfg!.capabilities!.reasoning).toBe(true);
    expect(openaiCfg!.capabilities!.supportsTemperature).toBe(false);
    expect(openaiCfg!.capabilities!.defaultMaxTokens).toBe(16384);
    expect(openaiCfg!.reasoningEffort).toBe("medium");
  });

  it("omits capabilities when none specified in TOML", () => {
    writeHomeConfig(
      tempHome,
      `
provider = "openai"

[providers.openai]
model = "gpt-4o"
api_key = "test-key"
`,
    );

    const config = loadConfig("/nonexistent/path");
    const openaiCfg = config.providers["openai"];
    expect(openaiCfg).toBeDefined();
    expect(openaiCfg!.capabilities).toBeUndefined();
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

  it("injects provider-specific env credentials for the active provider", () => {
    const original = { ...process.env };
    process.env["OPENAI_API_KEY"] = "sk-openai";

    try {
      const config = loadConfig("/nonexistent/path", {
        provider: "openai",
        model: "gpt-4.1",
      });
      expect(config.providers["openai"]?.apiKey).toBe("sk-openai");
    } finally {
      if (original["OPENAI_API_KEY"] === undefined) {
        delete process.env["OPENAI_API_KEY"];
      } else {
        process.env["OPENAI_API_KEY"] = original["OPENAI_API_KEY"];
      }
    }
  });

  it("does not apply DEVAGENT_API_KEY to non-gateway providers", () => {
    const original = { ...process.env };
    process.env["DEVAGENT_API_KEY"] = "ilg-gateway";

    try {
      const config = loadConfig("/nonexistent/path", {
        provider: "openai",
        model: "gpt-4.1",
      });
      expect(config.providers["openai"]?.apiKey).toBeUndefined();
    } finally {
      if (original["DEVAGENT_API_KEY"] === undefined) {
        delete process.env["DEVAGENT_API_KEY"];
      } else {
        process.env["DEVAGENT_API_KEY"] = original["DEVAGENT_API_KEY"];
      }
    }
  });

  it("injects resolved top-level api_key into existing provider blocks", () => {
    writeHomeConfig(
      tempHome,
      `
provider = "custom_provider_for_test"
model = "test-model"
api_key = "top-level-key"

[providers.custom_provider_for_test]
model = "test-model-v2"
base_url = "https://example.test/v1"
`,
    );

    const config = loadConfig("/nonexistent/path");
    expect(config.providers["custom_provider_for_test"]?.model).toBe(
      "test-model-v2",
    );
    expect(config.providers["custom_provider_for_test"]?.baseUrl).toBe(
      "https://example.test/v1",
    );
    expect(config.providers["custom_provider_for_test"]?.apiKey).toBe(
      "top-level-key",
    );
  });

  it("respects override parameter", () => {
    const config = loadConfig("/nonexistent/path", {
      provider: "openrouter",
      model: "meta-llama/llama-3",
    });

    expect(config.provider).toBe("openrouter");
    expect(config.model).toBe("meta-llama/llama-3");
  });

  it("does not fail on inactive provider env refs when overrides select a different provider", () => {
    writeHomeConfig(
      tempHome,
      `
provider = "openai"
model = "gpt-4.1"

[providers.openai]
api_key = "env:OPENAI_API_KEY"
model = "gpt-4.1"
`,
    );

    const config = loadConfig("/nonexistent/path", {
      provider: "devagent-api",
      model: "cortex",
    });

    expect(config.provider).toBe("devagent-api");
    expect(config.model).toBe("cortex");
    expect(config.providers["openai"]?.apiKey).toBeUndefined();
  });

  it("applies OpenAI-family default subagent model and reasoning profiles", () => {
    writeHomeConfig(
      tempHome,
      `
provider = "openai"
model = "gpt-5.4"
`,
    );

    const config = loadConfig("/nonexistent/path");
    expect(config.agentModelOverrides?.explore).toBe("gpt-5.4-mini");
    expect(config.agentModelOverrides?.reviewer).toBe("gpt-5.4");
    expect(config.agentModelOverrides?.architect).toBe("gpt-5.4");
    expect(config.agentReasoningOverrides?.explore).toBe("low");
    expect(config.agentReasoningOverrides?.reviewer).toBe("high");
    expect(config.agentReasoningOverrides?.architect).toBe("high");
    expect(config.agentModelOverrides?.general).toBeUndefined();
    expect(config.agentReasoningOverrides?.general).toBeUndefined();
  });

  it("allows explicit subagent overrides to replace defaults while keeping unspecified defaults", () => {
    writeHomeConfig(
      tempHome,
      `
provider = "openai"
model = "gpt-5.4"

[subagents.agent_model_overrides]
explore = "gpt-5.4"

[subagents.agent_reasoning_overrides]
reviewer = "medium"
architect = "invalid"
`,
    );

    const config = loadConfig("/nonexistent/path");
    expect(config.agentModelOverrides?.explore).toBe("gpt-5.4");
    expect(config.agentModelOverrides?.reviewer).toBe("gpt-5.4");
    expect(config.agentReasoningOverrides?.explore).toBe("low");
    expect(config.agentReasoningOverrides?.reviewer).toBe("medium");
    expect(config.agentReasoningOverrides?.architect).toBe("high");
  });

  it("applies devagent-api default subagent reasoning without forcing model overrides", () => {
    writeHomeConfig(
      tempHome,
      `
provider = "devagent-api"
model = "cortex"
`,
    );

    const config = loadConfig("/nonexistent/path");
    expect(config.agentReasoningOverrides?.explore).toBe("low");
    expect(config.agentReasoningOverrides?.reviewer).toBe("high");
    expect(config.agentReasoningOverrides?.architect).toBe("high");
    expect(config.agentModelOverrides).toBeUndefined();
  });

  it("fails fast on invalid TOML", () => {
    writeHomeConfig(tempHome, "this is not valid [[[ toml");

    expect(() => loadConfig("/nonexistent/path")).toThrow("Failed to parse config file");
  });

  it("resolves env: references in api_key", () => {
    writeHomeConfig(
      tempHome,
      `
provider = "anthropic"
api_key = "env:TEST_DEVAGENT_KEY"
`,
    );

    process.env["TEST_DEVAGENT_KEY"] = "sk-test-123";

    try {
      const config = loadConfig("/nonexistent/path");
      expect(config.providers["anthropic"]?.apiKey).toBe("sk-test-123");
    } finally {
      delete process.env["TEST_DEVAGENT_KEY"];
    }
  });

  it("fails fast on missing env reference", () => {
    writeHomeConfig(
      tempHome,
      `
provider = "anthropic"
api_key = "env:NONEXISTENT_VAR_12345"
`,
    );

    expect(() => loadConfig("/nonexistent/path")).toThrow(
      'Environment variable "NONEXISTENT_VAR_12345" referenced in config but not set',
    );
  });
});

describe("DEFAULT_CONTEXT export", () => {
  it("is accessible and has expected default keepRecentMessages", () => {
    expect(DEFAULT_CONTEXT).toBeDefined();
    expect(DEFAULT_CONTEXT.keepRecentMessages).toBe(40);
    expect(DEFAULT_CONTEXT.pruningStrategy).toBe("hybrid");
    expect(DEFAULT_CONTEXT.triggerRatio).toBe(0.9);
  });
});

describe("findProjectRoot", () => {
  it("returns null for root directory", () => {
    const result = findProjectRoot("/");
    // May or may not be null depending on system, but should not throw
    expect(typeof result === "string" || result === null).toBe(true);
  });

  it("prefers a workspace with .agents/skills over a parent git root", () => {
    const parentDir = join(tmpdir(), `devagent-root-agents-${Date.now()}`);
    const workspaceDir = join(parentDir, "arkts-helloworld");
    const nestedDir = join(workspaceDir, "src");
    mkdirSync(join(parentDir, ".git"), { recursive: true });
    mkdirSync(join(workspaceDir, ".agents", "skills", "implement-arkts"), { recursive: true });
    mkdirSync(nestedDir, { recursive: true });

    try {
      expect(findProjectRoot(workspaceDir)).toBe(workspaceDir);
      expect(findProjectRoot(nestedDir)).toBe(workspaceDir);
    } finally {
      rmSync(parentDir, { recursive: true, force: true });
    }
  });

  it("does not treat .devagent alone as a workspace root anchor", () => {
    const parentDir = join(tmpdir(), `devagent-root-devagent-${Date.now()}`);
    const workspaceDir = join(parentDir, "arkts-helloworld");
    const nestedDir = join(workspaceDir, "src");
    mkdirSync(join(parentDir, ".git"), { recursive: true });
    mkdirSync(join(workspaceDir, ".devagent"), { recursive: true });
    writeFileSync(join(workspaceDir, ".devagent", "ai_agent_instructions.md"), "# test\n");
    mkdirSync(nestedDir, { recursive: true });

    try {
      expect(findProjectRoot(workspaceDir)).toBe(parentDir);
      expect(findProjectRoot(nestedDir)).toBe(parentDir);
    } finally {
      rmSync(parentDir, { recursive: true, force: true });
    }
  });

  it("parses session_state settings including trackFindings, maxFindings, maxEnvFacts from TOML", () => {
    const home = mkdtempSync(join(tmpdir(), "devagent-config-session-home-"));
    const originalHome = process.env["HOME"];
    process.env["HOME"] = home;

    writeHomeConfig(
      home,
      `
[session_state]
persist = false
track_plan = true
track_files = false
track_env = true
track_tool_results = false
track_findings = false
max_modified_files = 10
max_env_facts = 5
max_tool_summaries = 15
max_findings = 8
`,
    );

    try {
      const config = loadConfig("/nonexistent/path");
      expect(config.sessionState).toBeDefined();
      expect(config.sessionState!.persist).toBe(false);
      expect(config.sessionState!.trackPlan).toBe(true);
      expect(config.sessionState!.trackFiles).toBe(false);
      expect(config.sessionState!.trackEnv).toBe(true);
      expect(config.sessionState!.trackToolResults).toBe(false);
      expect(config.sessionState!.trackFindings).toBe(false);
      expect(config.sessionState!.maxModifiedFiles).toBe(10);
      expect(config.sessionState!.maxEnvFacts).toBe(5);
      expect(config.sessionState!.maxToolSummaries).toBe(15);
      expect(config.sessionState!.maxFindings).toBe(8);
    } finally {
      if (originalHome === undefined) {
        delete process.env["HOME"];
      } else {
        process.env["HOME"] = originalHome;
      }
      rmSync(home, { recursive: true, force: true });
    }
  });

  it("parses pruneProtectTokens from TOML context config", () => {
    const home = mkdtempSync(join(tmpdir(), "devagent-config-context-home-"));
    const originalHome = process.env["HOME"];
    process.env["HOME"] = home;

    writeHomeConfig(
      home,
      `
[context]
prune_protect_tokens = 90000
`,
    );

    try {
      const config = loadConfig("/nonexistent/path");
      expect(config.context.pruneProtectTokens).toBe(90000);
    } finally {
      if (originalHome === undefined) {
        delete process.env["HOME"];
      } else {
        process.env["HOME"] = originalHome;
      }
      rmSync(home, { recursive: true, force: true });
    }
  });

  it("falls back to the nearest git root when no workspace markers are present", () => {
    const dir = join(tmpdir(), `devagent-root-test-${Date.now()}`);
    const subdir = join(dir, "a", "b", "c");
    mkdirSync(join(dir, ".git"), { recursive: true });
    mkdirSync(subdir, { recursive: true });

    try {
      const result = findProjectRoot(subdir);
      expect(result).toBe(dir);
    } finally {
      rmSync(dir, { recursive: true });
    }
  });
});
