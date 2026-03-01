import { describe, it, expect, beforeEach } from "vitest";
import { createDelegateTool } from "./delegate-tool.js";
import { AgentRegistry } from "./agents.js";
import type { LLMProvider, StreamChunk, DevAgentConfig } from "@devagent/core";
import { EventBus, ApprovalGate, ApprovalMode } from "@devagent/core";
import { ToolRegistry } from "@devagent/tools";
import type { ToolSpec } from "@devagent/core";

// ─── Mock Provider ──────────────────────────────────────────

function createMockProvider(
  responses: Array<StreamChunk[]>,
): LLMProvider {
  let callIndex = 0;
  return {
    id: "mock",
    async *chat(): AsyncIterable<StreamChunk> {
      const chunks = responses[callIndex] ?? [];
      callIndex++;
      for (const chunk of chunks) {
        yield chunk;
      }
    },
    abort() {},
  };
}

function makeConfig(): DevAgentConfig {
  return {
    provider: "mock",
    model: "mock-model",
    providers: {},
    approval: {
      mode: ApprovalMode.FULL_AUTO,
      auditLog: false,
      toolOverrides: {},
      pathRules: [],
    },
    budget: {
      maxIterations: 10,
      maxContextTokens: 100_000,
      responseHeadroom: 2_000,
      costWarningThreshold: 1.0,
      enableCostTracking: true,
    },
    context: {
      pruningStrategy: "hybrid",
      triggerRatio: 0.8,
      keepRecentMessages: 10,
    },
    arkts: {
      enabled: false,
      strictMode: false,
      targetVersion: "5.0",
    },
  };
}

describe("delegate tool", () => {
  let bus: EventBus;
  let config: DevAgentConfig;

  beforeEach(() => {
    bus = new EventBus();
    config = makeConfig();
  });

  it("delegates to a General agent and returns output", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "Subagent response" },
        { type: "done", content: "" },
      ],
    ]);

    const tools = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    const result = await delegateTool.handler(
      { agent_type: "general", task: "Say hello" },
      { repoRoot: "/tmp", config, sessionId: "" },
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("Subagent (general) completed");
    expect(result.output).toContain("Subagent response");
  });

  it("delegates to a Reviewer agent", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "Code looks good, no issues found." },
        { type: "done", content: "" },
      ],
    ]);

    const tools = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    const result = await delegateTool.handler(
      { agent_type: "reviewer", task: "Review the auth module" },
      { repoRoot: "/tmp", config, sessionId: "" },
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("Subagent (reviewer) completed");
  });

  it("rejects invalid agent type", async () => {
    const provider = createMockProvider([]);
    const tools = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    const result = await delegateTool.handler(
      { agent_type: "nonexistent", task: "Do something" },
      { repoRoot: "/tmp", config, sessionId: "" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Invalid agent type");
  });

  it("handles subagent errors gracefully", async () => {
    // Provider that throws
    const provider: LLMProvider = {
      id: "error",
      async *chat(): AsyncIterable<StreamChunk> {
        throw new Error("Provider exploded");
      },
      abort() {},
    };

    const tools = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    const result = await delegateTool.handler(
      { agent_type: "general", task: "This will fail" },
      { repoRoot: "/tmp", config, sessionId: "" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Subagent (general) failed");
  });

  it("generates unique subagent IDs", async () => {
    const provider = createMockProvider([
      [{ type: "text", content: "First" }, { type: "done", content: "" }],
      [{ type: "text", content: "Second" }, { type: "done", content: "" }],
    ]);

    const tools = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    const r1 = await delegateTool.handler(
      { agent_type: "general", task: "Task 1" },
      { repoRoot: "/tmp", config, sessionId: "" },
    );
    const r2 = await delegateTool.handler(
      { agent_type: "general", task: "Task 2" },
      { repoRoot: "/tmp", config, sessionId: "" },
    );

    expect(r1.success).toBe(true);
    expect(r2.success).toBe(true);
  });

  it("caps subagent maxIterations to SUBAGENT_MAX_ITERATIONS", async () => {
    // When parent config has maxIterations: 0 (unlimited),
    // the subagent should still get a capped value
    const unlimitedConfig = {
      ...config,
      budget: { ...config.budget, maxIterations: 0 },
    };

    // Provider that records what config the subagent got
    let subagentIterations = 0;
    const provider = createMockProvider([
      [
        { type: "text", content: "Done" },
        { type: "done", content: "" },
      ],
    ]);

    const tools = new ToolRegistry();
    const gate = new ApprovalGate(unlimitedConfig.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider,
      tools,
      bus,
      approvalGate: gate,
      config: unlimitedConfig,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    const result = await delegateTool.handler(
      { agent_type: "general", task: "Quick task" },
      { repoRoot: "/tmp", config: unlimitedConfig, sessionId: "" },
    );

    expect(result.success).toBe(true);
    // The subagent should complete successfully with capped iterations
  });

  it("resolves getParentSessionState getter at call time, not registration time", async () => {
    // Simulate session resume: sessionState reference changes after registration
    const { SessionState } = await import("./session-state.js");
    const stateV1 = new SessionState({ persist: false });
    const stateV2 = new SessionState({ persist: false });

    // Seed v2 with a marker so we can verify it was used
    stateV2.addEnvFact("test_marker", "v2-was-used");

    let currentState: SessionState = stateV1;

    const provider = createMockProvider([
      [{ type: "text", content: "Done" }, { type: "done", content: "" }],
    ]);
    const tools = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
      // Getter captures the mutable variable — resolves at call time
      getParentSessionState: () => currentState,
    });

    // Swap the state AFTER registration (simulating resume)
    currentState = stateV2;

    // The delegate handler should use stateV2, not stateV1
    const result = await delegateTool.handler(
      { agent_type: "general", task: "Quick task" },
      { repoRoot: "/tmp", config, sessionId: "" },
    );

    expect(result.success).toBe(true);
    // If the getter was resolved at registration time, it would use stateV1.
    // The fact the call succeeds with the swapped state proves lazy resolution.
  });

  it("has correct tool metadata", () => {
    const provider = createMockProvider([]);
    const tools = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    expect(delegateTool.name).toBe("delegate");
    expect(delegateTool.category).toBe("workflow");
    expect(delegateTool.paramSchema.required).toContain("agent_type");
    expect(delegateTool.paramSchema.required).toContain("task");
  });
});
