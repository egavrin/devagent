import { describe, it, expect, beforeEach } from "vitest";
import { TaskLoop } from "./task-loop.js";
import type {
  LLMProvider,
  Message,
  ToolSpec,
  StreamChunk,
  DevAgentConfig,
} from "@devagent/core";
import {
  EventBus,
  ApprovalGate,
  ApprovalMode,
  MessageRole,
  ProviderError,
} from "@devagent/core";
import { ToolRegistry } from "@devagent/tools";

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

// ─── Helpers ────────────────────────────────────────────────

function makeConfig(overrides?: Partial<DevAgentConfig>): DevAgentConfig {
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
    ...overrides,
  };
}

function makeEchoTool(): ToolSpec {
  return {
    name: "echo",
    description: "Echo the input",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        text: { type: "string" },
      },
      required: ["text"],
    },
    resultSchema: { type: "object" },
    handler: async (params) => ({
      success: true,
      output: `Echo: ${params["text"] as string}`,
      error: null,
      artifacts: [],
    }),
  };
}

describe("TaskLoop", () => {
  let bus: EventBus;
  let config: DevAgentConfig;

  beforeEach(() => {
    bus = new EventBus();
    config = makeConfig();
  });

  it("runs a simple text response", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "Hello, " },
        { type: "text", content: "world!" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "You are a test assistant.",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Hi!");
    expect(result.iterations).toBe(1);
    expect(result.aborted).toBe(false);

    // Check that final message was emitted
    const assistantMessages = result.messages.filter(
      (m) => m.role === MessageRole.ASSISTANT,
    );
    expect(assistantMessages.length).toBeGreaterThan(0);
    expect(assistantMessages[assistantMessages.length - 1]!.content).toBe(
      "Hello, world!",
    );
  });

  it("executes tool calls and feeds results back", async () => {
    const provider = createMockProvider([
      // First response: tool call
      [
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Second response: final text
      [
        { type: "text", content: "Got echo result" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "You are a test assistant.",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Run the echo tool");
    expect(result.iterations).toBe(2);

    // Should have tool result in messages
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(1);
    expect(toolMessages[0]!.content).toBe("Echo: test");
  });

  it("handles tool errors gracefully", async () => {
    const failingTool: ToolSpec = {
      name: "fail",
      description: "Always fails",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => {
        throw new Error("Intentional failure");
      },
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: "{}",
          toolCallId: "call_0",
          toolName: "fail",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "I see the tool failed" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(failingTool);

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Call the fail tool");
    expect(result.iterations).toBe(2);

    // Error should be surfaced in tool message
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages[0]!.content).toContain("Error: Intentional failure");
  });

  it("enforces budget limit", async () => {
    // Provider that always returns tool calls (infinite loop)
    let callCount = 0;
    const provider: LLMProvider = {
      id: "infinite",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        yield {
          type: "tool_call",
          content: '{"text": "loop"}',
          toolCallId: `call_${callCount}`,
          toolName: "echo",
        };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const limitedConfig = makeConfig({
      budget: { ...config.budget, maxIterations: 3 },
    });
    const gate = new ApprovalGate(limitedConfig.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config: limitedConfig,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    await expect(loop.run("Loop forever")).rejects.toThrow(
      "Max iterations",
    );
  });

  it("respects plan mode (readonly tools only)", async () => {
    const writeTool: ToolSpec = {
      name: "write",
      description: "Write tool",
      category: "mutating",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "written",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: "{}",
          toolCallId: "call_0",
          toolName: "write",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Cannot write in plan mode" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    registry.register(writeTool);

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      mode: "plan",
    });

    // In plan mode, "write" tool won't be in available tools
    // so it should result in "Unknown tool" error
    const result = await loop.run("Write something");
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages[0]!.content).toContain("Error: Unknown tool");
  });

  it("emits events during execution", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"text": "hi"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const gate = new ApprovalGate(config.approval, bus);
    const events: string[] = [];
    bus.on("message:user", () => events.push("user"));
    bus.on("tool:before", () => events.push("tool:before"));
    bus.on("tool:after", () => events.push("tool:after"));
    bus.on("message:assistant", () => events.push("assistant"));

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    await loop.run("Echo something");

    expect(events).toContain("user");
    expect(events).toContain("tool:before");
    expect(events).toContain("tool:after");
    expect(events).toContain("assistant");
  });

  it("can switch modes", () => {
    const provider = createMockProvider([]);
    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      mode: "plan",
    });

    expect(loop.getMode()).toBe("plan");
    loop.setMode("act");
    expect(loop.getMode()).toBe("act");
  });

  it("supports multi-turn conversation (reuse loop)", async () => {
    const provider = createMockProvider([
      // Turn 1
      [
        { type: "text", content: "Answer to turn 1" },
        { type: "done", content: "" },
      ],
      // Turn 2
      [
        { type: "text", content: "Answer to turn 2" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "You are a test assistant.",
      repoRoot: "/tmp",
    });

    // Turn 1
    const r1 = await loop.run("First question");
    expect(r1.iterations).toBe(1);

    // Reset for turn 2 — keeps message history
    loop.resetIterations();

    // Turn 2
    const r2 = await loop.run("Second question");
    expect(r2.iterations).toBe(1);

    // Messages should contain both turns
    const messages = loop.getMessages();
    const userMessages = messages.filter((m) => m.role === MessageRole.USER);
    expect(userMessages.length).toBe(2);
    expect(userMessages[0]!.content).toBe("First question");
    expect(userMessages[1]!.content).toBe("Second question");

    const assistantMessages = messages.filter(
      (m) => m.role === MessageRole.ASSISTANT,
    );
    expect(assistantMessages.length).toBe(2);
  });

  it("resetIterations resets counter and aborted flag", () => {
    const provider = createMockProvider([]);
    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    loop.abort();
    loop.resetIterations();
    expect(loop.getIterations()).toBe(0);
  });

  // ─── Provider Error Retry Tests ──────────────────────────────

  it("propagates ProviderError when provider throws", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "failing",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        throw new ProviderError("Connection refused");
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    await expect(loop.run("hello")).rejects.toThrow(ProviderError);
    // 1 initial + 3 retries = 4 total calls
    expect(callCount).toBe(4);
  });

  it("retries on ProviderError and succeeds on second attempt", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "flaky",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount === 1) {
          throw new ProviderError("Temporary failure");
        }
        yield { type: "text", content: "Success after retry" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("hello");
    expect(callCount).toBe(2);
    expect(result.iterations).toBe(1);

    const assistantMsgs = result.messages.filter(
      (m) => m.role === MessageRole.ASSISTANT,
    );
    expect(assistantMsgs[assistantMsgs.length - 1]!.content).toBe(
      "Success after retry",
    );
  });

  it("throws after exhausting all retry attempts", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "always-failing",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        throw new ProviderError("Persistent failure");
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    await expect(loop.run("hello")).rejects.toThrow("Persistent failure");
    expect(callCount).toBe(4); // 1 initial + 3 retries
  });

  it("does not retry non-ProviderError exceptions", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "type-error",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        throw new TypeError("Cannot read property 'x' of undefined");
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    await expect(loop.run("hello")).rejects.toThrow(TypeError);
    expect(callCount).toBe(1); // No retry
  });

  it("emits PROVIDER_RETRY error events during retries", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "retry-events",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount <= 2) {
          throw new ProviderError("Transient error");
        }
        yield { type: "text", content: "Finally works" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);

    const retryEvents: Array<{ message: string; code: string; fatal: boolean }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "PROVIDER_RETRY") {
        retryEvents.push(event as { message: string; code: string; fatal: boolean });
      }
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("hello");
    expect(callCount).toBe(3);
    expect(retryEvents.length).toBe(2);
    expect(retryEvents[0]!.code).toBe("PROVIDER_RETRY");
    expect(retryEvents[0]!.fatal).toBe(false);
    expect(retryEvents[1]!.code).toBe("PROVIDER_RETRY");
    expect(result.iterations).toBe(1);
  });
});
