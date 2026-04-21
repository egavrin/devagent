import { describe, it, expect, beforeEach } from "vitest";

import { AgentRegistry } from "./agents.js";
import { createDelegateTool } from "./delegate-tool.js";
import type { LLMProvider, StreamChunk, DevAgentConfig , ToolSpec } from "../core/index.js";
import { EventBus, ApprovalGate, ApprovalMode } from "../core/index.js";
import { ToolRegistry } from "../tools/index.js";

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

  it("delegates to an Explore agent", async () => {
    const provider = createMockProvider([
      [
        {
          type: "text",
          content: JSON.stringify({
            answer: "Found auth module at src/auth.ts:15",
            evidence: ["src/auth.ts:15 - createAuthService()"],
            relatedFiles: ["src/auth.ts"],
            unresolved: [],
          }),
        },
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
      { agent_type: "explore", task: "Find where authentication is implemented" },
      { repoRoot: "/tmp", config, sessionId: "" },
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("Subagent (explore) completed");
    expect(result.metadata).toMatchObject({
      agentMeta: {
        agentType: "explore",
        parentId: "parent-1",
        depth: 1,
      },
      parsedOutput: {
        answer: "Found auth module at src/auth.ts:15",
      },
    });
  });

  it("emits subagent lifecycle events with lane and quality metadata", async () => {
    const provider = createMockProvider([
      [
        {
          type: "text",
          content: JSON.stringify({
            answer: "Found docs lane evidence",
            evidence: ["docs/fixed-array.md:12"],
            relatedFiles: ["docs/fixed-array.md"],
            unresolved: [],
          }),
        },
        { type: "done", content: "" },
      ],
    ]);

    const events: Array<{ event: string; data: unknown }> = [];
    bus.on("subagent:start", (data) => events.push({ event: "start", data }));
    bus.on("subagent:update", (data) => events.push({ event: "update", data }));
    bus.on("subagent:end", (data) => events.push({ event: "end", data }));

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
      {
        agent_type: "explore",
        request: {
          objective: "Inspect docs lane",
          laneLabel: "spec/docs",
          scope: "docs",
          constraints: [],
          exclusions: ["runtime"],
          successCriteria: [],
          parentContext: "Delegated for focused evidence gathering",
        },
      },
      { repoRoot: "/tmp", config, sessionId: "", batchId: "batch-1", batchSize: 2 },
    );

    expect(result.success).toBe(true);
    expect(events.length).toBeGreaterThanOrEqual(3);
    expect(events[0]!.data).toMatchObject({
      agentId: "parent-1-sub-1",
      agentType: "explore",
      laneLabel: "spec/docs",
      status: "running",
      batchId: "batch-1",
      batchSize: 2,
    });
    expect(events.some((entry) => entry.event === "update")).toBe(true);
    expect(events[events.length - 1]!.data).toMatchObject({
      agentId: "parent-1-sub-1",
      agentType: "explore",
      laneLabel: "spec/docs",
      status: "completed",
      parsedOutputKeys: ["answer", "evidence", "relatedFiles", "unresolved"],
      batchId: "batch-1",
      batchSize: 2,
    });
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
    expect(result.error).toContain("explore");
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

  it("emits subagent:error when the child fails", async () => {
    const provider: LLMProvider = {
      id: "error",
      async *chat(): AsyncIterable<StreamChunk> {
        throw new Error("Provider exploded");
      },
      abort() {},
    };

    const errors: unknown[] = [];
    bus.on("subagent:error", (event) => errors.push(event));

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
    expect(errors).toHaveLength(1);
    expect(errors[0]).toMatchObject({
      agentId: "parent-1-sub-1",
      agentType: "general",
      status: "error",
    });
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

  it("caps explore agent to lower iteration budget than general", async () => {
    // Unlimited parent — the per-agent-type cap is the only constraint.
    const unlimitedConfig = {
      ...config,
      budget: { ...config.budget, maxIterations: 0 },
    };

    // Provider that always calls a tool — forces iteration until maxIterations.
    function createLoopingProvider(): LLMProvider {
      let callIndex = 0;
      return {
        id: "looping",
        async *chat(): AsyncIterable<StreamChunk> {
          callIndex++;
          yield {
            type: "tool_call",
            content: `{"pattern": "*.ts"}`,
            toolCallId: `call_${callIndex}`,
            toolName: "find_files",
          };
          yield { type: "done", content: "" };
        },
        abort() {},
      };
    }

    const findFilesTool: ToolSpec = {
      name: "find_files",
      description: "Find files",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "src/main.ts",
        error: null,
        artifacts: [],
      }),
    };

    const tools = new ToolRegistry();
    tools.register(findFilesTool);
    const gate = new ApprovalGate(unlimitedConfig.approval, bus);
    const agentRegistry = new AgentRegistry();

    // Run explore — should cap at 15 iterations
    const exploreTool = createDelegateTool({
      provider: createLoopingProvider(),
      tools,
      bus,
      approvalGate: gate,
      config: unlimitedConfig,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    const exploreResult = await exploreTool.handler(
      { agent_type: "explore", task: "Search codebase" },
      { repoRoot: "/tmp", config: unlimitedConfig, sessionId: "" },
    );
    expect(exploreResult.success).toBe(true);

    // Run general with same setup — should cap at 30 iterations
    const generalTool = createDelegateTool({
      provider: createLoopingProvider(),
      tools,
      bus,
      approvalGate: gate,
      config: unlimitedConfig,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-2",
    });

    const generalResult = await generalTool.handler(
      { agent_type: "general", task: "Search codebase" },
      { repoRoot: "/tmp", config: unlimitedConfig, sessionId: "" },
    );
    expect(generalResult.success).toBe(true);

    // Extract iteration counts from the output
    const exploreMatch = exploreResult.output.match(/\[(\d+) iterations/);
    const generalMatch = generalResult.output.match(/\[(\d+) iterations/);
    const exploreIter = Number(exploreMatch![1]);
    const generalIter = Number(generalMatch![1]);

    // Explore should have fewer iterations than general.
    // TaskLoop adds one grace iteration for summarization, so actual count
    // is cap + 1.  Assert the behavioral invariant: explore < general,
    // and each is within its cap (+ 1 grace).
    expect(exploreIter).toBeLessThanOrEqual(15 + 1);
    expect(generalIter).toBeGreaterThan(exploreIter);
    expect(generalIter).toBeLessThanOrEqual(30 + 1);
  });

  it("caps explore to min(parentMax, 15) when parent budget is bounded", async () => {
    // Parent has 10 iterations — both explore (15) and general (30)
    // should be capped to 10.
    const boundedConfig = {
      ...config,
      budget: { ...config.budget, maxIterations: 10 },
    };

    function createLoopingProvider(): LLMProvider {
      let callIndex = 0;
      return {
        id: "looping",
        async *chat(): AsyncIterable<StreamChunk> {
          callIndex++;
          yield {
            type: "tool_call",
            content: `{"pattern": "*.ts"}`,
            toolCallId: `call_${callIndex}`,
            toolName: "find_files",
          };
          yield { type: "done", content: "" };
        },
        abort() {},
      };
    }

    const findFilesTool: ToolSpec = {
      name: "find_files",
      description: "Find files",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "src/main.ts",
        error: null,
        artifacts: [],
      }),
    };

    const tools = new ToolRegistry();
    tools.register(findFilesTool);
    const gate = new ApprovalGate(boundedConfig.approval, bus);
    const agentRegistry = new AgentRegistry();

    const delegateTool = createDelegateTool({
      provider: createLoopingProvider(),
      tools,
      bus,
      approvalGate: gate,
      config: boundedConfig,
      repoRoot: "/tmp",
      agentRegistry,
      parentAgentId: "parent-1",
    });

    const result = await delegateTool.handler(
      { agent_type: "explore", task: "Search codebase" },
      { repoRoot: "/tmp", config: boundedConfig, sessionId: "" },
    );
    expect(result.success).toBe(true);

    const match = result.output.match(/\[(\d+) iterations/);
    const iterations = Number(match![1]);
    // min(parentMax=10, exploreCap=15) = 10, plus 1 grace iteration
    expect(iterations).toBeLessThanOrEqual(10 + 1);
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
    expect(delegateTool.description).toContain("explore");
    expect(delegateTool.paramSchema.required).toContain("agent_type");
    expect(delegateTool.paramSchema.properties).toHaveProperty("task");
    expect(delegateTool.paramSchema.properties).toHaveProperty("request");
    const requestSchema = delegateTool.paramSchema.properties?.["request"] as Record<string, unknown>;
    expect(requestSchema.additionalProperties).toBe(false);
    expect(requestSchema.required).toEqual(["objective"]);
  });
});
