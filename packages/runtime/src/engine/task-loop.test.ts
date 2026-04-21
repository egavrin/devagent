import { resolve } from "node:path";
import {
  it,
  expect,
  beforeEach,
  beforeAll,
  vi,
} from "vitest";

import { SessionState } from "./session-state.js";
import { TaskLoop } from "./task-loop.js";
import {
  createMockProvider,
  makeConfig,
  makeEchoTool,
} from "./task-loop.test-helpers.js";
import type {
  LLMProvider,
  Message,
  ToolSpec,
  StreamChunk,
} from "../core/index.js";
import {
  EventBus,
  ApprovalGate,
  MessageRole,
  ProviderError,
  loadModelRegistry,
} from "../core/index.js";
import { ToolRegistry } from "../tools/index.js";

let bus: EventBus;
let config: ReturnType<typeof makeConfig>;


beforeAll(() => {
    const modelsDir = resolve(import.meta.dirname ?? new URL(".", import.meta.url).pathname, "../../../../models");
    loadModelRegistry(undefined, [modelsDir]);
  });

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
  it("prepends pinned messages for a single run", async () => {
    const seenMessages: Message[][] = [];
    const provider: LLMProvider = {
      id: "capture",
      async *chat(messages): AsyncIterable<StreamChunk> {
        seenMessages.push([...messages]);
        yield { type: "text", content: "Done" };
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
      systemPrompt: "You are a test assistant.",
      repoRoot: "/tmp",
    });

    await loop.run("Inspect the diff", {
      prependedMessages: [
        {
          role: MessageRole.USER,
          content: "Pre-loaded diff",
          pinned: true,
        },
      ],
    });

    expect(seenMessages).toHaveLength(1);
    expect(seenMessages[0]).toMatchObject([
      { role: MessageRole.SYSTEM, content: "You are a test assistant." },
      { role: MessageRole.USER, content: "Pre-loaded diff", pinned: true },
      { role: MessageRole.USER, content: "Inspect the diff" },
    ]);
  });
  it("does not leak prepended messages into later runs", async () => {
    const seenMessages: Message[][] = [];
    const provider: LLMProvider = {
      id: "capture",
      async *chat(messages): AsyncIterable<StreamChunk> {
        seenMessages.push([...messages]);
        yield { type: "text", content: "Done" };
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
      systemPrompt: "You are a test assistant.",
      repoRoot: "/tmp",
    });

    await loop.run("Inspect the diff", {
      prependedMessages: [
        {
          role: MessageRole.USER,
          content: "Pre-loaded diff",
          pinned: true,
        },
      ],
    });

    await loop.run("Second turn");

    expect(seenMessages).toHaveLength(2);
    expect(seenMessages[0]?.map((message) => message.content)).toContain("Pre-loaded diff");
    expect(seenMessages[1]?.map((message) => message.content)).not.toContain("Pre-loaded diff");
  });
  it("updates the root system prompt without dropping prior history", async () => {
    const seenMessages: Message[][] = [];
    const provider: LLMProvider = {
      id: "capture",
      async *chat(messages): AsyncIterable<StreamChunk> {
        seenMessages.push([...messages]);
        yield { type: "text", content: `run-${seenMessages.length}` };
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
      systemPrompt: "Prompt A",
      repoRoot: "/tmp",
    });

    await loop.run("first turn");
    loop.updateSystemPrompt("Prompt B");
    loop.resetIterations();
    await loop.run("second turn");

    expect(seenMessages).toHaveLength(2);
    expect(seenMessages[0]?.[0]).toMatchObject({ role: MessageRole.SYSTEM, content: "Prompt A" });
    expect(seenMessages[1]?.[0]).toMatchObject({ role: MessageRole.SYSTEM, content: "Prompt B" });
    expect(seenMessages[1]?.some((message) => message.role === MessageRole.USER && message.content === "first turn")).toBe(true);
    expect(loop.getMessages()[0]).toMatchObject({ role: MessageRole.SYSTEM, content: "Prompt B" });
  });
  it("injects seeded session state before the first provider call", async () => {
    const sessionState = new SessionState({ persist: false });
    sessionState.recordReadonlyCoverage("read_file", "src/already-read.ts");

    const seenMessageContents: string[] = [];
    const provider: LLMProvider = {
      id: "capture",
      async *chat(messages): AsyncIterable<StreamChunk> {
        seenMessageContents.push(...messages.map((m) => m.content ?? ""));
        yield { type: "text", content: "Done" };
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
      systemPrompt: "You are a test assistant.",
      repoRoot: "/tmp",
      sessionState,
      injectSessionStateOnFirstTurn: true,
    });

    await loop.run("Inspect the repo");

    expect(seenMessageContents.some((content) => content.includes("[SESSION STATE"))).toBe(true);
    expect(seenMessageContents.some((content) => content.includes("src/already-read.ts"))).toBe(true);
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
  it("rejects namespaced tool call names with guidance", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "functions.echo",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "done" },
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

    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(1);
    expect(toolMessages[0]!.content).toContain("Error: Unknown tool: functions.echo");
    expect(toolMessages[0]!.content).toContain('Try "echo"');
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
  it("emits iteration:start with correct count per LLM round-trip", async () => {
    // Provider returns 2 parallel tool calls in response 1, then a final text.
    // iteration:start should fire twice (once per LLM call), not three times.
    const provider = createMockProvider([
      // Response 1: two parallel tool calls in the same response
      [
        {
          type: "tool_call",
          content: '{"text": "a"}',
          toolCallId: "call_a",
          toolName: "echo",
        },
        {
          type: "tool_call",
          content: '{"text": "b"}',
          toolCallId: "call_b",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Response 2: final text
      [
        { type: "text", content: "Done" },
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

    const iterationEvents: Array<{ iteration: number; maxIterations: number }> = [];
    bus.on("iteration:start", (event) => {
      iterationEvents.push({ iteration: event.iteration, maxIterations: event.maxIterations });
    });

    const result = await loop.run("Run both tools");

    // 2 LLM round-trips, NOT 3 (two tool calls happen in the same iteration)
    expect(result.iterations).toBe(2);
    expect(iterationEvents).toHaveLength(2);
    expect(iterationEvents[0]!.iteration).toBe(1);
    expect(iterationEvents[1]!.iteration).toBe(2);
    expect(iterationEvents[0]!.maxIterations).toBe(config.budget.maxIterations);
  });
  it("respects plan mode (readonly + state tools only)", async () => {
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

    // In plan mode, "write" (mutating) tool won't be in available tools
    // so it should result in "Unknown tool" error
    const result = await loop.run("Write something");
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages[0]!.content).toContain("Error: Unknown tool");
  });
  it("allows state tools in plan mode", async () => {
    const stateTool: ToolSpec = {
      name: "update_plan",
      description: "State tool",
      category: "state",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "plan updated",
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
          toolName: "update_plan",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Plan updated" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    registry.register(stateTool);

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

    const result = await loop.run("Create a plan");
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    // State tool should execute successfully in plan mode
    expect(toolMessages[0]!.content).toContain("plan updated");
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
  it("compacts context before the provider call and applies response headroom", async () => {
    const observedCalls: ReadonlyArray<Message>[] = [];
    const provider: LLMProvider = {
      id: "observe-preflight",
      async *chat(messages: ReadonlyArray<Message>): AsyncIterable<StreamChunk> {
        observedCalls.push(messages);
        yield { type: "text", content: "Compacted first" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const truncateAsync = vi.fn(
      async (messages: ReadonlyArray<Message>, maxTokens: number) => ({
        messages: [messages[0]!, messages[messages.length - 1]!],
        truncated: true,
        removedCount: messages.length - 2,
        estimatedTokens: Math.max(0, maxTokens - 1),
      }),
    );
    const fakeContextManager = {
      truncateAsync,
      truncate: vi.fn(),
    };

    // Budget must be large enough that after compaction + real token
    // estimation of the 2 remaining messages, we still fit. The
    // fake's estimatedTokens is ignored for the budget check (the
    // loop re-estimates tokens from actual message content after
    // injecting session state).
    const compactingConfig = makeConfig({
      budget: {
        ...config.budget,
        maxContextTokens: 200,
        responseHeadroom: 10,
      },
      context: {
        ...config.context,
        triggerRatio: 0.5,
        keepRecentMessages: 1,
      },
    });

    const gate = new ApprovalGate(compactingConfig.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: new ToolRegistry(),
      bus,
      approvalGate: gate,
      config: compactingConfig,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      initialMessages: [
        { role: MessageRole.SYSTEM, content: "S".repeat(300) },
        { role: MessageRole.USER, content: "Original task " + "U".repeat(180) },
        { role: MessageRole.ASSISTANT, content: "Answer 1" + "A".repeat(120) },
      ],
      contextManager: fakeContextManager as unknown as import("@devagent/runtime").ContextManager,
    });

    const result = await loop.run("New question");
    expect(result.status).toBe("success");
    expect(truncateAsync).toHaveBeenCalledTimes(1);
    expect(truncateAsync.mock.calls[0]?.[1]).toBe(190); // 200 maxContext - 10 headroom
    // observedCalls includes the main chat call + compaction quality judge call
    expect(observedCalls.length).toBeGreaterThanOrEqual(1);
    expect(observedCalls[0]!.length).toBeLessThan(4);
  });
  it("fails fast when context compaction throws before provider call", async () => {
    let providerCalled = false;
    const provider: LLMProvider = {
      id: "never-called",
      async *chat(): AsyncIterable<StreamChunk> {
        providerCalled = true;
        yield { type: "text", content: "Should not happen" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const fakeContextManager = {
      truncateAsync: vi.fn(async () => {
        throw new Error("compact explode");
      }),
      truncate: vi.fn(),
    };

    const compactingConfig = makeConfig({
      budget: {
        ...config.budget,
        maxContextTokens: 30,
        responseHeadroom: 5,
      },
      context: {
        ...config.context,
        triggerRatio: 0.1,
        keepRecentMessages: 1,
      },
    });

    const gate = new ApprovalGate(compactingConfig.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: new ToolRegistry(),
      bus,
      approvalGate: gate,
      config: compactingConfig,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      initialMessages: [
        { role: MessageRole.SYSTEM, content: "S".repeat(300) },
        { role: MessageRole.USER, content: "Task " + "U".repeat(220) },
      ],
      contextManager: fakeContextManager as unknown as import("@devagent/runtime").ContextManager,
    });

    await expect(loop.run("Question")).rejects.toThrow("compact explode");
    expect(providerCalled).toBe(false);
  });
  it("retries immediately with forced compaction on provider context overflow", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "overflow-on-large-context",
      async *chat(messages: ReadonlyArray<Message>): AsyncIterable<StreamChunk> {
        callCount++;
        if (messages.length > 2) {
          throw new ProviderError("maximum context length exceeded");
        }
        yield { type: "text", content: "Recovered after compaction" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const truncateAsync = vi.fn(
      async (messages: ReadonlyArray<Message>) => ({
        messages: [messages[0]!, messages[messages.length - 1]!],
        truncated: true,
        removedCount: messages.length - 2,
        estimatedTokens: 1,
      }),
    );
    const fakeContextManager = {
      truncateAsync,
      truncate: vi.fn(),
    };

    const overflowConfig = makeConfig({
      budget: {
        ...config.budget,
        maxContextTokens: 200,
        responseHeadroom: 0,
      },
      context: {
        ...config.context,
        triggerRatio: 1,
        keepRecentMessages: 20,
      },
    });

    const gate = new ApprovalGate(overflowConfig.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: new ToolRegistry(),
      bus,
      approvalGate: gate,
      config: overflowConfig,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      initialMessages: [
        { role: MessageRole.SYSTEM, content: "sys" },
        { role: MessageRole.USER, content: "task" },
        { role: MessageRole.ASSISTANT, content: "answer" },
      ],
      contextManager: fakeContextManager as unknown as import("@devagent/runtime").ContextManager,
    });

    const result = await loop.run("follow up");
    expect(result.status).toBe("success");
    expect(result.lastText).toBe("Recovered after compaction");
    // callCount includes: 1 overflow + 1 recovery + 1 compaction quality judge = 3
    expect(callCount).toBeGreaterThanOrEqual(2);
    expect(truncateAsync).toHaveBeenCalledTimes(1);
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
      // Turn 1: text response (text-only = done, no judge)
      [
        { type: "text", content: "Answer to turn 1" },
        { type: "done", content: "" },
      ],
      // Turn 2: text response
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
