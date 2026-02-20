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

  it("enforces budget limit with grace iteration", async () => {
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

    // Now resolves with budget_exceeded status instead of throwing,
    // because the loop injects a grace iteration for summary
    const result = await loop.run("Loop forever");
    expect(result.status).toBe("budget_exceeded");
    expect(result.iterations).toBeGreaterThanOrEqual(3);
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

  it("returns success status on normal completion", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "All done!" },
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
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Hello");
    expect(result.status).toBe("success");
    expect(result.lastText).toBe("All done!");
  });

  it("retries with summary request on empty response after tool calls", async () => {
    const provider = createMockProvider([
      // First: tool call
      [
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Second: empty response (no text, no tool calls)
      [{ type: "done", content: "" }],
      // Third: response after summary request injection
      [
        { type: "text", content: "Here is the summary" },
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
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Do the thing");
    expect(result.status).toBe("success");
    expect(result.lastText).toBe("Here is the summary");
  });

  it("returns empty_response when summary retry also produces no text", async () => {
    const provider = createMockProvider([
      // First: tool call
      [
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Second: empty response
      [{ type: "done", content: "" }],
      // Third: still empty after summary request
      [{ type: "done", content: "" }],
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
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Do the thing");
    expect(result.status).toBe("empty_response");
  });

  it("tracks lastText across tool call iterations", async () => {
    const provider = createMockProvider([
      // First: text + tool call
      [
        { type: "text", content: "Let me check..." },
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Second: final text
      [
        { type: "text", content: "Final answer" },
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
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Do something");
    expect(result.status).toBe("success");
    expect(result.lastText).toBe("Final answer");
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

  // ─── Doom Loop Detection Tests ─────────────────────────────

  it("detects doom loop when same failing tool called 3 times with identical args", async () => {
    const failingTool: ToolSpec = {
      name: "run_command",
      description: "Run a command",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: false,
        output: "",
        error: "Command exited with code 1",
        artifacts: [],
      }),
    };

    let callCount = 0;
    const provider: LLMProvider = {
      id: "doom-loop",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount <= 3) {
          // Same tool call with identical args 3 times
          yield {
            type: "tool_call",
            content: '{"cmd": "es2panda --input test.ets"}',
            toolCallId: `call_${callCount}`,
            toolName: "run_command",
          };
          yield { type: "done", content: "" };
        } else {
          // After doom loop warning, LLM responds with text
          yield { type: "text", content: "The command keeps failing. Let me try a different approach." };
          yield { type: "done", content: "" };
        }
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(failingTool);

    const gate = new ApprovalGate(config.approval, bus);

    const doomLoopEvents: Array<{ code: string }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "DOOM_LOOP") {
        doomLoopEvents.push(event as { code: string });
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

    const result = await loop.run("Run the compiler");

    // Loop should NOT have broken — LLM got to respond
    expect(result.status).toBe("success");
    expect(result.lastText).toContain("different approach");

    // Doom loop event should have been emitted
    expect(doomLoopEvents.length).toBe(1);

    // System message with doom loop warning should be in messages
    const systemMessages = result.messages.filter(
      (m) => m.role === MessageRole.SYSTEM && m.content?.includes("doom loop") || m.content?.includes("same arguments"),
    );
    expect(systemMessages.length).toBeGreaterThan(0);
  });

  it("does not trigger doom loop for different arguments", async () => {
    let callCount = 0;
    const failingTool: ToolSpec = {
      name: "run_command",
      description: "Run a command",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: false,
        output: "",
        error: "Command exited with code 1",
        artifacts: [],
      }),
    };

    const provider: LLMProvider = {
      id: "varied-args",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount <= 3) {
          // Same tool but different args each time
          yield {
            type: "tool_call",
            content: JSON.stringify({ cmd: `attempt_${callCount}` }),
            toolCallId: `call_${callCount}`,
            toolName: "run_command",
          };
          yield { type: "done", content: "" };
        } else {
          yield { type: "text", content: "Done trying" };
          yield { type: "done", content: "" };
        }
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(failingTool);

    const gate = new ApprovalGate(config.approval, bus);

    const doomLoopEvents: Array<{ code: string }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "DOOM_LOOP") {
        doomLoopEvents.push(event as { code: string });
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

    const result = await loop.run("Try different approaches");
    expect(result.status).toBe("success");

    // No doom loop because args are different each time
    expect(doomLoopEvents.length).toBe(0);
  });

  it("resets doom loop state on successful tool call", async () => {
    let callCount = 0;
    const conditionalTool: ToolSpec = {
      name: "run_command",
      description: "Run a command",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async (params) => {
        const cmd = params["cmd"] as string;
        if (cmd === "good") {
          return { success: true, output: "OK", error: null, artifacts: [] };
        }
        return { success: false, output: "", error: "Failed", artifacts: [] };
      },
    };

    const provider: LLMProvider = {
      id: "reset-doom",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount === 1) {
          // First: fail with args "bad"
          yield { type: "tool_call", content: '{"cmd": "bad"}', toolCallId: "c1", toolName: "run_command" };
          yield { type: "done", content: "" };
        } else if (callCount === 2) {
          // Second: fail again with same args
          yield { type: "tool_call", content: '{"cmd": "bad"}', toolCallId: "c2", toolName: "run_command" };
          yield { type: "done", content: "" };
        } else if (callCount === 3) {
          // Third: succeed with different args — resets doom loop tracking
          yield { type: "tool_call", content: '{"cmd": "good"}', toolCallId: "c3", toolName: "run_command" };
          yield { type: "done", content: "" };
        } else if (callCount === 4) {
          // Fourth: fail again with "bad" — counter starts fresh
          yield { type: "tool_call", content: '{"cmd": "bad"}', toolCallId: "c4", toolName: "run_command" };
          yield { type: "done", content: "" };
        } else if (callCount === 5) {
          // Fifth: fail again with "bad"
          yield { type: "tool_call", content: '{"cmd": "bad"}', toolCallId: "c5", toolName: "run_command" };
          yield { type: "done", content: "" };
        } else {
          yield { type: "text", content: "Gave up" };
          yield { type: "done", content: "" };
        }
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(conditionalTool);

    const gate = new ApprovalGate(config.approval, bus);

    const doomLoopEvents: Array<{ code: string }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "DOOM_LOOP") {
        doomLoopEvents.push(event as { code: string });
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

    const result = await loop.run("Test reset");
    expect(result.status).toBe("success");

    // No doom loop because success at call 3 reset the counter
    // After reset: only 2 consecutive "bad" calls (c4, c5), not 3
    expect(doomLoopEvents.length).toBe(0);
  });

  // ─── Memory Integration Tests ──────────────────────────────

  it("extracts doom-loop lesson into memory when memoryStore provided", async () => {
    const { MemoryStore } = await import("@devagent/core");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const { mkdirSync, rmSync } = await import("node:fs");
    const { randomUUID } = await import("node:crypto");

    const tmpDir = join(tmpdir(), `devagent-test-memory-${randomUUID()}`);
    mkdirSync(tmpDir, { recursive: true });
    const memoryStore = new MemoryStore({ dbPath: join(tmpDir, "memory.db") });

    try {
      const failingTool: ToolSpec = {
        name: "run_command",
        description: "Run a command",
        category: "readonly",
        paramSchema: { type: "object" },
        resultSchema: { type: "object" },
        handler: async () => ({
          success: false,
          output: "",
          error: "Command exited with code 1",
          artifacts: [],
        }),
      };

      let callCount = 0;
      const provider: LLMProvider = {
        id: "doom-memory",
        async *chat(): AsyncIterable<StreamChunk> {
          callCount++;
          if (callCount <= 3) {
            yield {
              type: "tool_call",
              content: '{"cmd": "es2panda --input test.ets"}',
              toolCallId: `call_${callCount}`,
              toolName: "run_command",
            };
            yield { type: "done", content: "" };
          } else {
            yield { type: "text", content: "Switching approach." };
            yield { type: "done", content: "" };
          }
        },
        abort() {},
      };

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
        memoryStore,
      });

      await loop.run("Run the compiler");

      // Check that a doom-loop lesson was extracted
      const memories = memoryStore.search({ category: "mistake", query: "doom-loop" });
      expect(memories.length).toBeGreaterThanOrEqual(1);
      expect(memories[0]!.key).toContain("doom-loop-run_command");
      expect(memories[0]!.content).toContain("repeatedly");
    } finally {
      memoryStore.close();
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it("calls applyDecay on memoryStore at session end", async () => {
    const { MemoryStore } = await import("@devagent/core");
    const { tmpdir } = await import("node:os");
    const { join } = await import("node:path");
    const { mkdirSync, rmSync } = await import("node:fs");
    const { randomUUID } = await import("node:crypto");

    const tmpDir = join(tmpdir(), `devagent-test-decay-${randomUUID()}`);
    mkdirSync(tmpDir, { recursive: true });
    const memoryStore = new MemoryStore({ dbPath: join(tmpDir, "memory.db") });

    try {
      // Store a memory so decay has something to work on
      memoryStore.store("pattern", "test-decay", "This is a test pattern");

      const provider = createMockProvider([
        [
          { type: "text", content: "Done" },
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
        systemPrompt: "Test",
        repoRoot: "/tmp",
        memoryStore,
      });

      const result = await loop.run("Hello");
      expect(result.status).toBe("success");

      // Verify memory still exists (applyDecay shouldn't remove fresh memories)
      const memories = memoryStore.search({ category: "pattern" });
      expect(memories.length).toBe(1);
      expect(memories[0]!.key).toBe("test-decay");
    } finally {
      memoryStore.close();
      rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  // ─── Checkpoint + Double-Check Tests ───────────────────────

  it("creates checkpoint after mutating tool success", async () => {
    const createCalls: Array<{ description: string; toolName: string }> = [];
    const mockCheckpointManager = {
      create(description: string, toolName: string) {
        createCalls.push({ description, toolName });
        return { id: "cp-0", commitHash: "abc", description, timestamp: Date.now(), toolName };
      },
      init() {},
      list() { return []; },
      diff() { return ""; },
      restore() { return false; },
    } as any;

    const writeTool: ToolSpec = {
      name: "write_file",
      description: "Write a file",
      category: "mutating",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "Written",
        error: null,
        artifacts: ["/tmp/test.ts"],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts", "content": "hello"}',
          toolCallId: "call_0",
          toolName: "write_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "File written." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
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
      checkpointManager: mockCheckpointManager,
    });

    const result = await loop.run("Write a file");
    expect(result.status).toBe("success");
    expect(createCalls.length).toBe(1);
    expect(createCalls[0]!.toolName).toBe("write_file");
    expect(createCalls[0]!.description).toContain("/tmp/test.ts");
  });

  it("injects system message when double-check fails after mutating tool", async () => {
    const mockDoubleCheck = {
      isEnabled: () => true,
      async check() {
        return {
          passed: false,
          diagnosticErrors: ["/tmp/test.ts: Unexpected token"],
          testOutput: null,
          testPassed: null,
        };
      },
      formatResults(result: any) {
        return `Diagnostic errors (${result.diagnosticErrors.length}):\n` +
          result.diagnosticErrors.map((e: string) => `  - ${e}`).join("\n");
      },
    } as any;

    const writeTool: ToolSpec = {
      name: "write_file",
      description: "Write a file",
      category: "mutating",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "Written",
        error: null,
        artifacts: ["/tmp/test.ts"],
      }),
    };

    const provider = createMockProvider([
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
        { type: "text", content: "I see the error, fixing..." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
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
      doubleCheck: mockDoubleCheck,
    });

    const result = await loop.run("Write a file");
    expect(result.status).toBe("success");

    // Check that a VALIDATION FAILED system message was injected
    const validationMessages = result.messages.filter(
      (m) => m.role === MessageRole.SYSTEM && m.content?.includes("VALIDATION FAILED"),
    );
    expect(validationMessages.length).toBe(1);
    expect(validationMessages[0]!.content).toContain("Unexpected token");
  });

  it("skips checkpoint for readonly tools", async () => {
    const createCalls: Array<{ description: string }> = [];
    const mockCheckpointManager = {
      create(description: string, toolName: string) {
        createCalls.push({ description });
        return null;
      },
      init() {},
      list() { return []; },
      diff() { return ""; },
      restore() { return false; },
    } as any;

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"text": "hello"}',
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
    registry.register(makeEchoTool()); // "readonly" category
    const gate = new ApprovalGate(config.approval, bus);

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      checkpointManager: mockCheckpointManager,
    });

    const result = await loop.run("Echo something");
    expect(result.status).toBe("success");

    // Checkpoint should NOT have been created for readonly tool
    expect(createCalls.length).toBe(0);
  });

  // ─── Session Resume (initialMessages) Tests ─────────────────

  it("initializes with provided messages when initialMessages given", async () => {
    const previousMessages: Message[] = [
      { role: MessageRole.SYSTEM, content: "You are a test assistant from a previous session." },
      { role: MessageRole.USER, content: "Previous question" },
      { role: MessageRole.ASSISTANT, content: "Previous answer" },
    ];

    const provider = createMockProvider([
      [
        { type: "text", content: "Continuing from where we left off." },
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
      systemPrompt: "This should be ignored when initialMessages provided.",
      repoRoot: "/tmp",
      initialMessages: previousMessages,
    });

    const result = await loop.run("Continue the conversation");
    expect(result.status).toBe("success");

    const messages = result.messages;

    // First message should be the system prompt from previous session, NOT the new one
    expect(messages[0]!.role).toBe(MessageRole.SYSTEM);
    expect(messages[0]!.content).toBe("You are a test assistant from a previous session.");

    // Previous conversation should be preserved
    expect(messages[1]!.role).toBe(MessageRole.USER);
    expect(messages[1]!.content).toBe("Previous question");
    expect(messages[2]!.role).toBe(MessageRole.ASSISTANT);
    expect(messages[2]!.content).toBe("Previous answer");

    // New user query should follow
    expect(messages[3]!.role).toBe(MessageRole.USER);
    expect(messages[3]!.content).toBe("Continue the conversation");

    // Assistant response
    const assistantMsgs = messages.filter(
      (m) => m.role === MessageRole.ASSISTANT,
    );
    expect(assistantMsgs.length).toBe(2); // Previous + new
    expect(assistantMsgs[1]!.content).toBe("Continuing from where we left off.");
  });

  it("defaults to system prompt when no initialMessages", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "Hello!" },
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
      systemPrompt: "Fresh system prompt.",
      repoRoot: "/tmp",
      // No initialMessages — should use systemPrompt
    });

    const result = await loop.run("Hi");
    expect(result.status).toBe("success");

    const messages = result.messages;

    // First message should be the system prompt
    expect(messages[0]!.role).toBe(MessageRole.SYSTEM);
    expect(messages[0]!.content).toBe("Fresh system prompt.");

    // Followed by user query
    expect(messages[1]!.role).toBe(MessageRole.USER);
    expect(messages[1]!.content).toBe("Hi");
  });

  it("handles empty initialMessages array by falling back to system prompt", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "Fresh start." },
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
      systemPrompt: "Default prompt.",
      repoRoot: "/tmp",
      initialMessages: [], // Empty array — should fall back to system prompt
    });

    const result = await loop.run("Hello");
    expect(result.status).toBe("success");

    // Should use systemPrompt since initialMessages is empty
    expect(result.messages[0]!.role).toBe(MessageRole.SYSTEM);
    expect(result.messages[0]!.content).toBe("Default prompt.");
  });

  // ─── Parallel Execution Tests ─────────────────────────────

  it("executes multiple readonly tool calls in parallel", async () => {
    // Track execution timeline to verify concurrency
    const executionLog: Array<{ tool: string; event: "start" | "end"; time: number }> = [];
    const baseTime = Date.now();

    const makeSlowReadonly = (name: string, delayMs: number, output: string): ToolSpec => ({
      name,
      description: `Slow ${name}`,
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => {
        executionLog.push({ tool: name, event: "start", time: Date.now() - baseTime });
        await new Promise((r) => setTimeout(r, delayMs));
        executionLog.push({ tool: name, event: "end", time: Date.now() - baseTime });
        return { success: true, output, error: null, artifacts: [] };
      },
    });

    const provider = createMockProvider([
      [
        // LLM returns 3 readonly tool calls at once
        { type: "tool_call", content: '{}', toolCallId: "c1", toolName: "slow_a" },
        { type: "tool_call", content: '{}', toolCallId: "c2", toolName: "slow_b" },
        { type: "tool_call", content: '{}', toolCallId: "c3", toolName: "slow_c" },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "All done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeSlowReadonly("slow_a", 100, "result_a"));
    registry.register(makeSlowReadonly("slow_b", 100, "result_b"));
    registry.register(makeSlowReadonly("slow_c", 100, "result_c"));

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

    const start = Date.now();
    const result = await loop.run("Run all three");
    const elapsed = Date.now() - start;

    expect(result.status).toBe("success");

    // All 3 tool results should be in messages
    const toolMessages = result.messages.filter((m) => m.role === MessageRole.TOOL);
    expect(toolMessages).toHaveLength(3);
    expect(toolMessages[0]!.content).toBe("result_a");
    expect(toolMessages[1]!.content).toBe("result_b");
    expect(toolMessages[2]!.content).toBe("result_c");

    // Parallel execution: 3 × 100ms tools should complete in ~100-150ms, not 300ms+
    // Be generous with the threshold to avoid flaky tests
    expect(elapsed).toBeLessThan(500); // Sequential would take 300ms+

    // Verify tools started roughly at the same time (parallel)
    const starts = executionLog.filter((e) => e.event === "start");
    expect(starts).toHaveLength(3);
    const maxStartDiff = Math.max(...starts.map((s) => s.time)) - Math.min(...starts.map((s) => s.time));
    expect(maxStartDiff).toBeLessThan(50); // All started within 50ms of each other
  });

  it("executes mutating tool calls sequentially even when mixed with readonly", async () => {
    const executionOrder: string[] = [];

    const makeTrackedTool = (name: string, category: "readonly" | "mutating"): ToolSpec => ({
      name,
      description: `Tracked ${name}`,
      category,
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => {
        executionOrder.push(name);
        return { success: true, output: `${name} done`, error: null, artifacts: [] };
      },
    });

    const provider = createMockProvider([
      [
        // LLM returns: readonly, readonly, mutating, readonly
        { type: "tool_call", content: '{}', toolCallId: "c1", toolName: "read_a" },
        { type: "tool_call", content: '{}', toolCallId: "c2", toolName: "read_b" },
        { type: "tool_call", content: '{}', toolCallId: "c3", toolName: "write_c" },
        { type: "tool_call", content: '{}', toolCallId: "c4", toolName: "read_d" },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeTrackedTool("read_a", "readonly"));
    registry.register(makeTrackedTool("read_b", "readonly"));
    registry.register(makeTrackedTool("write_c", "mutating"));
    registry.register(makeTrackedTool("read_d", "readonly"));

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

    const result = await loop.run("Do everything");
    expect(result.status).toBe("success");

    // All 4 tool results should appear
    const toolMessages = result.messages.filter((m) => m.role === MessageRole.TOOL);
    expect(toolMessages).toHaveLength(4);

    // Mutating tool must execute AFTER the first two readonly tools complete
    // and BEFORE the last readonly tool
    const writeIdx = executionOrder.indexOf("write_c");
    const readDIdx = executionOrder.indexOf("read_d");
    expect(writeIdx).toBeLessThan(readDIdx);

    // read_a and read_b should both execute before write_c
    const readAIdx = executionOrder.indexOf("read_a");
    const readBIdx = executionOrder.indexOf("read_b");
    expect(readAIdx).toBeLessThan(writeIdx);
    expect(readBIdx).toBeLessThan(writeIdx);
  });

  it("preserves tool result order matching callIds for parallel execution", async () => {
    const provider = createMockProvider([
      [
        { type: "tool_call", content: '{"text":"first"}', toolCallId: "c1", toolName: "echo" },
        { type: "tool_call", content: '{"text":"second"}', toolCallId: "c2", toolName: "echo" },
        { type: "tool_call", content: '{"text":"third"}', toolCallId: "c3", toolName: "echo" },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Results received." },
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
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Echo three things");
    const toolMsgs = result.messages.filter((m) => m.role === MessageRole.TOOL);
    expect(toolMsgs).toHaveLength(3);

    // Results must match the original callId order
    expect(toolMsgs[0]!.toolCallId).toBe("c1");
    expect(toolMsgs[0]!.content).toBe("Echo: first");
    expect(toolMsgs[1]!.toolCallId).toBe("c2");
    expect(toolMsgs[1]!.content).toBe("Echo: second");
    expect(toolMsgs[2]!.toolCallId).toBe("c3");
    expect(toolMsgs[2]!.content).toBe("Echo: third");
  });

  // ─── Tool Fatigue Detection ────────────────────────────────

  it("detects tool fatigue after 5 different-arg failures of same tool", async () => {
    const failingTool: ToolSpec = {
      name: "run_command",
      description: "Run a command",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: false,
        output: "",
        error: "Command exited with code 1",
        artifacts: [],
      }),
    };

    let callCount = 0;
    const provider: LLMProvider = {
      id: "fatigue-test",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount <= 5) {
          // Same tool, DIFFERENT args each time
          yield {
            type: "tool_call",
            content: JSON.stringify({ command: `attempt_${callCount}` }),
            toolCallId: `call_${callCount}`,
            toolName: "run_command",
          };
          yield { type: "done", content: "" };
        } else {
          yield {
            type: "text",
            content: "Switching to different approach.",
          };
          yield { type: "done", content: "" };
        }
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(failingTool);

    const gate = new ApprovalGate(config.approval, bus);
    const fatigueEvents: Array<{ code: string }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "TOOL_FATIGUE") {
        fatigueEvents.push(event as { code: string });
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

    const result = await loop.run("Try many approaches");
    expect(result.status).toBe("success");

    // Tool fatigue should have fired after 5 different-arg failures
    expect(fatigueEvents.length).toBe(1);

    // Escalated warning should be in messages
    const fatigueMessages = result.messages.filter(
      (m) =>
        m.role === MessageRole.SYSTEM &&
        m.content?.includes("ESCALATED WARNING"),
    );
    expect(fatigueMessages.length).toBe(1);
    expect(fatigueMessages[0]!.content).toContain("run_command");
  });

  it("resets tool fatigue counter on success", async () => {
    let callCount = 0;
    const conditionalTool: ToolSpec = {
      name: "run_command",
      description: "Run a command",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async (params) => {
        const cmd = params["command"] as string;
        if (cmd === "good") {
          return { success: true, output: "OK", error: null, artifacts: [] };
        }
        return {
          success: false,
          output: "",
          error: "Failed",
          artifacts: [],
        };
      },
    };

    const provider: LLMProvider = {
      id: "fatigue-reset",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount <= 3) {
          // 3 failures
          yield {
            type: "tool_call",
            content: JSON.stringify({ command: `bad_${callCount}` }),
            toolCallId: `c${callCount}`,
            toolName: "run_command",
          };
          yield { type: "done", content: "" };
        } else if (callCount === 4) {
          // Success — resets counter
          yield {
            type: "tool_call",
            content: '{"command": "good"}',
            toolCallId: "c4",
            toolName: "run_command",
          };
          yield { type: "done", content: "" };
        } else if (callCount <= 7) {
          // 3 more failures (total < 5 since reset)
          yield {
            type: "tool_call",
            content: JSON.stringify({ command: `bad_again_${callCount}` }),
            toolCallId: `c${callCount}`,
            toolName: "run_command",
          };
          yield { type: "done", content: "" };
        } else {
          yield { type: "text", content: "Done" };
          yield { type: "done", content: "" };
        }
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(conditionalTool);
    const gate = new ApprovalGate(config.approval, bus);

    const fatigueEvents: Array<{ code: string }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "TOOL_FATIGUE") {
        fatigueEvents.push(event as { code: string });
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

    const result = await loop.run("Test fatigue reset");
    expect(result.status).toBe("success");

    // No fatigue: 3 failures, then success resets, then 3 more (never hits 5)
    expect(fatigueEvents.length).toBe(0);
  });

  it("does not repeat tool fatigue warning for same tool", async () => {
    const failingTool: ToolSpec = {
      name: "run_command",
      description: "Run a command",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: false,
        output: "",
        error: "Failed",
        artifacts: [],
      }),
    };

    let callCount = 0;
    const provider: LLMProvider = {
      id: "fatigue-no-repeat",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount <= 8) {
          yield {
            type: "tool_call",
            content: JSON.stringify({ command: `try_${callCount}` }),
            toolCallId: `c${callCount}`,
            toolName: "run_command",
          };
          yield { type: "done", content: "" };
        } else {
          yield { type: "text", content: "Gave up" };
          yield { type: "done", content: "" };
        }
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(failingTool);
    const gate = new ApprovalGate(config.approval, bus);

    const fatigueEvents: Array<{ code: string }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "TOOL_FATIGUE") {
        fatigueEvents.push(event as { code: string });
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

    const result = await loop.run("Fail many times");
    expect(result.status).toBe("success");

    // Fatigue warning fires once at count=5, not again at 6,7,8
    expect(fatigueEvents.length).toBe(1);
  });

  // ─── Plan Call Coalescing ──────────────────────────────────

  it("coalesces multiple update_plan calls, executing only the last", async () => {
    let planCallCount = 0;
    const planTool: ToolSpec = {
      name: "update_plan",
      description: "Update plan",
      category: "workflow",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => {
        planCallCount++;
        return {
          success: true,
          output: `Plan updated (call #${planCallCount})`,
          error: null,
          artifacts: [],
        };
      },
    };

    const provider = createMockProvider([
      [
        // LLM emits 3 consecutive update_plan calls
        {
          type: "tool_call",
          content: '{"steps": "step1"}',
          toolCallId: "p1",
          toolName: "update_plan",
        },
        {
          type: "tool_call",
          content: '{"steps": "step2"}',
          toolCallId: "p2",
          toolName: "update_plan",
        },
        {
          type: "tool_call",
          content: '{"steps": "step3_final"}',
          toolCallId: "p3",
          toolName: "update_plan",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Plan is set." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(planTool);
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

    const result = await loop.run("Create a plan");
    expect(result.status).toBe("success");

    // Only the LAST plan call should have been executed
    expect(planCallCount).toBe(1);

    // All 3 tool call IDs should have corresponding TOOL messages
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(3);

    // First two should be "skipped" synthetic responses
    expect(toolMessages[0]!.content).toContain("Skipped");
    expect(toolMessages[0]!.toolCallId).toBe("p1");
    expect(toolMessages[1]!.content).toContain("Skipped");
    expect(toolMessages[1]!.toolCallId).toBe("p2");

    // Third should be the real execution
    expect(toolMessages[2]!.content).toContain("Plan updated");
    expect(toolMessages[2]!.toolCallId).toBe("p3");
  });

  it("does not coalesce non-plan tool calls", async () => {
    let echoCount = 0;
    const echoTool: ToolSpec = {
      name: "echo",
      description: "Echo",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async (params) => {
        echoCount++;
        return {
          success: true,
          output: `Echo #${echoCount}: ${params["text"]}`,
          error: null,
          artifacts: [],
        };
      },
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"text":"a"}',
          toolCallId: "e1",
          toolName: "echo",
        },
        {
          type: "tool_call",
          content: '{"text":"b"}',
          toolCallId: "e2",
          toolName: "echo",
        },
        {
          type: "tool_call",
          content: '{"text":"c"}',
          toolCallId: "e3",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "All echoed." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(echoTool);
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

    const result = await loop.run("Echo three things");
    expect(result.status).toBe("success");

    // All 3 echo calls should have been executed (no coalescing)
    expect(echoCount).toBe(3);
  });
});
