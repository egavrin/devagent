import { describe, it, expect } from "vitest";
import { createCommitPlugin } from "./commit-plugin.js";
import { createReviewPlugin } from "./review-plugin.js";
import {
  createFeatureDevPlugin,
  getFeaturePhases,
} from "./feature-dev-plugin.js";
import { createBuiltinPlugins } from "./index.js";
import type { PluginContext } from "@devagent/core";
import { EventBus, ApprovalMode } from "@devagent/core";
import type { DevAgentConfig } from "@devagent/core";

function makeContext(): PluginContext {
  const config: DevAgentConfig = {
    provider: "test",
    model: "test-model",
    providers: {},
    approval: {
      mode: ApprovalMode.SUGGEST,
      autoApprovePlan: false,
      autoApproveCode: false,
      autoApproveShell: false,
      auditLog: false,
      toolOverrides: {},
      pathRules: [],
    },
    budget: {
      maxIterations: 10,
      maxContextTokens: 4096,
      responseHeadroom: 1024,
      costWarningThreshold: 1.0,
      enableCostTracking: false,
    },
    context: {
      pruningStrategy: "sliding_window",
      triggerRatio: 0.8,
      keepRecentMessages: 10,
    },
    arkts: { enabled: false, strictMode: false, targetVersion: "5.0" },
  };

  return {
    bus: new EventBus(),
    config,
    repoRoot: "/tmp/test-repo",
  };
}

describe("createBuiltinPlugins", () => {
  it("creates 3 built-in plugins", () => {
    const plugins = createBuiltinPlugins();
    expect(plugins).toHaveLength(3);
    expect(plugins.map((p) => p.name).sort()).toEqual([
      "commit",
      "feature-dev",
      "review",
    ]);
  });
});

describe("CommitPlugin", () => {
  it("has name and version", () => {
    const plugin = createCommitPlugin();
    expect(plugin.name).toBe("commit");
    expect(plugin.version).toBe("1.0.0");
  });

  it("exposes /commit command", () => {
    const plugin = createCommitPlugin();
    expect(plugin.commands).toBeDefined();
    expect(plugin.commands!["commit"]).toBeDefined();
    expect(plugin.commands!["commit"]!.description).toContain("commit");
  });

  it("activates without error", () => {
    const plugin = createCommitPlugin();
    expect(() => plugin.activate(makeContext())).not.toThrow();
  });
});

describe("ReviewPlugin", () => {
  it("has name and version", () => {
    const plugin = createReviewPlugin();
    expect(plugin.name).toBe("review");
    expect(plugin.version).toBe("1.0.0");
  });

  it("exposes /review command", () => {
    const plugin = createReviewPlugin();
    expect(plugin.commands).toBeDefined();
    expect(plugin.commands!["review"]).toBeDefined();
  });

  it("returns usage when no args", async () => {
    const plugin = createReviewPlugin();
    const result = await plugin.commands!["review"]!.execute("", makeContext());
    expect(result).toContain("Usage");
  });

  it("returns file-not-found for missing file", async () => {
    const plugin = createReviewPlugin();
    const result = await plugin.commands!["review"]!.execute(
      "nonexistent-file.ts",
      makeContext(),
    );
    expect(result).toContain("not found");
  });
});

describe("FeatureDevPlugin", () => {
  it("has name and version", () => {
    const plugin = createFeatureDevPlugin();
    expect(plugin.name).toBe("feature-dev");
    expect(plugin.version).toBe("1.0.0");
  });

  it("exposes /feature command", () => {
    const plugin = createFeatureDevPlugin();
    expect(plugin.commands).toBeDefined();
    expect(plugin.commands!["feature"]).toBeDefined();
  });

  it("returns usage when no description", async () => {
    const plugin = createFeatureDevPlugin();
    const result = await plugin.commands!["feature"]!.execute("", makeContext());
    expect(result).toContain("Usage");
  });

  it("generates workflow plan for a feature", async () => {
    const plugin = createFeatureDevPlugin();
    const result = await plugin.commands!["feature"]!.execute(
      "add caching layer",
      makeContext(),
    );
    expect(result).toContain("Feature Development Workflow");
    expect(result).toContain("add caching layer");
    expect(result).toContain("understand");
    expect(result).toContain("implement");
    expect(result).toContain("verify");
  });
});

describe("getFeaturePhases", () => {
  it("returns 7 phases", () => {
    const phases = getFeaturePhases("test feature", "/tmp");
    expect(phases).toHaveLength(7);
  });

  it("includes feature description in prompts", () => {
    const phases = getFeaturePhases("add dark mode", "/repo");
    expect(phases[0]!.prompt).toContain("add dark mode");
  });

  it("phases are in correct order", () => {
    const phases = getFeaturePhases("test", "/tmp");
    const names = phases.map((p) => p.name);
    expect(names).toEqual([
      "understand",
      "explore",
      "plan",
      "implement",
      "verify",
      "review",
      "summarize",
    ]);
  });
});
