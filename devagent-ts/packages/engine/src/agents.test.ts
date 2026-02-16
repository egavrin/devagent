import { describe, it, expect, beforeEach } from "vitest";
import { AgentRegistry, runAgent } from "./agents.js";
import type { AgentDefinition, AgentRunOptions } from "./agents.js";
import type {
  LLMProvider,
  StreamChunk,
  DevAgentConfig,
} from "@devagent/core";
import {
  AgentType,
  EventBus,
  ApprovalGate,
  ApprovalMode,
  MessageRole,
} from "@devagent/core";
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
      autoApprovePlan: false,
      autoApproveCode: false,
      autoApproveShell: false,
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

function makeReadOnlyTool(): ToolSpec {
  return {
    name: "read_file",
    description: "Read a file",
    category: "readonly",
    paramSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
    resultSchema: { type: "object" },
    handler: async (params) => ({
      success: true,
      output: `Content of ${params["path"] as string}`,
      error: null,
      artifacts: [],
    }),
  };
}

function makeMutatingTool(): ToolSpec {
  return {
    name: "write_file",
    description: "Write a file",
    category: "mutating",
    paramSchema: { type: "object", properties: { path: { type: "string" } }, required: ["path"] },
    resultSchema: { type: "object" },
    handler: async () => ({
      success: true,
      output: "Written",
      error: null,
      artifacts: [],
    }),
  };
}

// ─── AgentRegistry Tests ────────────────────────────────────

describe("AgentRegistry", () => {
  let registry: AgentRegistry;

  beforeEach(() => {
    registry = new AgentRegistry();
  });

  it("has all three built-in agent types", () => {
    expect(registry.has(AgentType.GENERAL)).toBe(true);
    expect(registry.has(AgentType.REVIEWER)).toBe(true);
    expect(registry.has(AgentType.ARCHITECT)).toBe(true);
  });

  it("returns correct definition for General agent", () => {
    const def = registry.get(AgentType.GENERAL);
    expect(def.name).toBe("General");
    expect(def.defaultMode).toBe("act");
    expect(def.allowedToolCategories).toContain("mutating");
  });

  it("returns correct definition for Reviewer agent", () => {
    const def = registry.get(AgentType.REVIEWER);
    expect(def.name).toBe("Reviewer");
    expect(def.defaultMode).toBe("plan");
    expect(def.allowedToolCategories).toEqual(["readonly"]);
  });

  it("returns correct definition for Architect agent", () => {
    const def = registry.get(AgentType.ARCHITECT);
    expect(def.name).toBe("Architect");
    expect(def.defaultMode).toBe("plan");
    expect(def.allowedToolCategories).toEqual(["readonly"]);
  });

  it("lists all definitions", () => {
    const defs = registry.list();
    expect(defs.length).toBe(3);
    const types = defs.map((d) => d.type);
    expect(types).toContain(AgentType.GENERAL);
    expect(types).toContain(AgentType.REVIEWER);
    expect(types).toContain(AgentType.ARCHITECT);
  });

  it("allows registering custom agent types", () => {
    const custom: AgentDefinition = {
      type: AgentType.GENERAL, // Overrides default
      name: "CustomGeneral",
      description: "Custom general agent",
      systemPromptTemplate: "Custom prompt for {{repoRoot}}",
      defaultMode: "act",
      allowedToolCategories: ["readonly"],
    };
    registry.register(custom);
    expect(registry.get(AgentType.GENERAL).name).toBe("CustomGeneral");
  });
});

// ─── runAgent Tests ─────────────────────────────────────────

describe("runAgent", () => {
  let bus: EventBus;
  let config: DevAgentConfig;
  let agentRegistry: AgentRegistry;

  beforeEach(() => {
    bus = new EventBus();
    config = makeConfig();
    agentRegistry = new AgentRegistry();
  });

  it("runs a General agent with a simple query", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "Hello from General agent" },
        { type: "done", content: "" },
      ],
    ]);

    const tools = new ToolRegistry();
    tools.register(makeReadOnlyTool());
    tools.register(makeMutatingTool());

    const gate = new ApprovalGate(config.approval, bus);
    const options: AgentRunOptions = {
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp/test",
      parentId: null,
      agentId: "agent-1",
    };

    const result = await runAgent(AgentType.GENERAL, "Hello", options, agentRegistry);
    expect(result.agentId).toBe("agent-1");
    expect(result.agentType).toBe(AgentType.GENERAL);
    expect(result.result.iterations).toBe(1);
    expect(result.result.aborted).toBe(false);
  });

  it("Reviewer agent only has read-only tools", async () => {
    const provider = createMockProvider([
      // Reviewer tries to use write_file — should fail as unknown tool
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts"}',
          toolCallId: "call_0",
          toolName: "write_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Cannot write in review mode" },
        { type: "done", content: "" },
      ],
    ]);

    const tools = new ToolRegistry();
    tools.register(makeReadOnlyTool());
    tools.register(makeMutatingTool());

    const gate = new ApprovalGate(config.approval, bus);
    const options: AgentRunOptions = {
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp/test",
      parentId: "parent-1",
      agentId: "agent-2",
    };

    const result = await runAgent(AgentType.REVIEWER, "Review code", options, agentRegistry);
    // write_file should be rejected — not in Reviewer's tool set
    const toolMessages = result.result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(1);
    expect(toolMessages[0]!.content).toContain("Error: Unknown tool");
  });

  it("Reviewer agent can use read-only tools", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "src/main.ts"}',
          toolCallId: "call_0",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "File looks good" },
        { type: "done", content: "" },
      ],
    ]);

    const tools = new ToolRegistry();
    tools.register(makeReadOnlyTool());
    tools.register(makeMutatingTool());

    const gate = new ApprovalGate(config.approval, bus);
    const options: AgentRunOptions = {
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/tmp/test",
      parentId: null,
      agentId: "agent-3",
    };

    const result = await runAgent(AgentType.REVIEWER, "Review main.ts", options, agentRegistry);
    const toolMessages = result.result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(1);
    expect(toolMessages[0]!.content).toBe("Content of src/main.ts");
  });

  it("substitutes repoRoot in system prompt", async () => {
    const messages: Array<{ role: string; content: string | null }> = [];
    const provider: LLMProvider = {
      id: "mock",
      async *chat(msgs): AsyncIterable<StreamChunk> {
        for (const m of msgs) {
          messages.push({ role: String(m.role), content: m.content });
        }
        yield { type: "text", content: "OK" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const tools = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const options: AgentRunOptions = {
      provider,
      tools,
      bus,
      approvalGate: gate,
      config,
      repoRoot: "/home/user/project",
      parentId: null,
      agentId: "agent-4",
    };

    await runAgent(AgentType.ARCHITECT, "Plan the architecture", options, agentRegistry);

    // System prompt should contain the actual repoRoot
    const systemMsg = messages.find((m) => m.role === "system");
    expect(systemMsg).toBeDefined();
    expect(systemMsg!.content).toContain("/home/user/project");
    expect(systemMsg!.content).not.toContain("{{repoRoot}}");
  });
});
