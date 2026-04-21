import { AgentType, ApprovalMode } from "@devagent/runtime";
import { describe, expect, it } from "vitest";

import { buildProviderConfig } from "./provider-config.js";
import type { DevAgentConfig } from "@devagent/runtime";

function makeConfig(): DevAgentConfig {
  return {
    provider: "chatgpt",
    model: "gpt-5.4",
    providers: {},
    approval: {
      mode: ApprovalMode.FULL_AUTO,
      auditLog: false,
      toolOverrides: {},
      pathRules: [],
    },
    budget: {
      maxIterations: 30,
      maxContextTokens: 100_000,
      responseHeadroom: 2_000,
      costWarningThreshold: 1,
      enableCostTracking: true,
    },
    context: {
      pruningStrategy: "hybrid",
      triggerRatio: 0.9,
      keepRecentMessages: 40,
      turnIsolation: true,
      midpointBriefingInterval: 15,
      briefingStrategy: "auto",
    },
  };
}

describe("buildProviderConfig", () => {
  it("forces the resolved model even when provider config contains a stale model", () => {
    const config: DevAgentConfig = {
      ...makeConfig(),
      model: "gpt-5.4-mini",
      providers: {
        chatgpt: {
          model: "gpt-5.4",
          oauthToken: "test-token",
        },
      },
    };

    const providerConfig = buildProviderConfig(config, "high");

    expect(providerConfig.model).toBe("gpt-5.4-mini");
    expect(providerConfig.oauthToken).toBe("test-token");
    expect(providerConfig.reasoningEffort).toBe("high");
  });

  it("uses per-agent reasoning overrides ahead of session-level reasoning", () => {
    const config: DevAgentConfig = {
      ...makeConfig(),
      agentReasoningOverrides: {
        [AgentType.EXPLORE]: "low",
        [AgentType.REVIEWER]: "high",
      },
    };

    const reviewerConfig = buildProviderConfig(config, "medium", AgentType.REVIEWER);
    const generalConfig = buildProviderConfig(config, "medium", AgentType.GENERAL);

    expect(reviewerConfig.reasoningEffort).toBe("high");
    expect(generalConfig.reasoningEffort).toBe("medium");
  });

  it("defaults the main devagent-api agent to high reasoning", () => {
    const config: DevAgentConfig = {
      ...makeConfig(),
      provider: "devagent-api",
      model: "cortex",
      providers: {
        "devagent-api": {
          model: "cortex",
        },
      },
    };

    const providerConfig = buildProviderConfig(config);

    expect(providerConfig.reasoningEffort).toBe("high");
  });

  it("keeps subagent reasoning overrides ahead of the devagent-api main default", () => {
    const config: DevAgentConfig = {
      ...makeConfig(),
      provider: "devagent-api",
      model: "cortex",
      providers: {
        "devagent-api": {
          model: "cortex",
        },
      },
      agentReasoningOverrides: {
        [AgentType.EXPLORE]: "low",
        [AgentType.REVIEWER]: "high",
        [AgentType.ARCHITECT]: "high",
      },
    };

    const exploreConfig = buildProviderConfig(config, undefined, AgentType.EXPLORE);
    const reviewerConfig = buildProviderConfig(config, undefined, AgentType.REVIEWER);
    const architectConfig = buildProviderConfig(config, undefined, AgentType.ARCHITECT);

    expect(exploreConfig.reasoningEffort).toBe("low");
    expect(reviewerConfig.reasoningEffort).toBe("high");
    expect(architectConfig.reasoningEffort).toBe("high");
  });
});
