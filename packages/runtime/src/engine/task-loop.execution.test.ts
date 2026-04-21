import { resolve } from "node:path";
import {
  it,
  expect,
  beforeEach,
  beforeAll,
} from "vitest";

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
  ProviderTlsCertificateError,
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

  it("surfaces TLS remediation after exhausting certificate verification retries", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "tls-always-failing",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        throw new ProviderTlsCertificateError("MockProvider", "unable to verify the first certificate");
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

    const runPromise = loop.run("hello");
    await expect(runPromise).rejects.toThrow("certificate verification failed");
    await expect(runPromise).rejects.toThrow("NODE_EXTRA_CA_CERTS");
    expect(callCount).toBe(4);
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
            content: '{"cmd": "tsc --noEmit test.ts"}',
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

  // ─── Double-Check Tests ────────────────────────────────────
  it("appends validation errors inline with tool result when double-check fails", async () => {
    const mockDoubleCheck = {
      isEnabled: () => true,
      async captureBaseline() { return {}; },
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
    const toolAfterEvents: Array<{ readonly result: { readonly metadata?: Record<string, unknown> } }> = [];
    bus.on("tool:after", (event) => {
      toolAfterEvents.push(event);
    });

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
    expect(result.status).toBe("empty_response");

    // Validation errors should be appended inline to tool result (not a separate SYSTEM message)
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(1);
    expect(toolMessages[0]!.content).toContain("Written");
    expect(toolMessages[0]!.content).toContain("VALIDATION ERRORS");
    expect(toolMessages[0]!.content).toContain("Unexpected token");
    expect(toolAfterEvents).toHaveLength(1);
    expect(toolAfterEvents[0]!.result.metadata?.["validationResult"]).toEqual(
      expect.objectContaining({
        passed: false,
        diagnosticErrors: ["/tmp/test.ts: Unexpected token"],
        testPassed: null,
      }),
    );

    // No separate SYSTEM validation message should exist
    const systemValidationMessages = result.messages.filter(
      (m) => m.role === MessageRole.SYSTEM && m.content?.includes("VALIDATION"),
    );
    expect(systemValidationMessages.length).toBe(0);
  });
  it("does not accept final text response while double-check failures are unresolved", async () => {
    let checkCalls = 0;
    const mockDoubleCheck = {
      isEnabled: () => true,
      async captureBaseline() { return {}; },
      async check() {
        checkCalls++;
        if (checkCalls === 1) {
          return {
            passed: false,
            diagnosticErrors: ["/tmp/test.ts: Unexpected token"],
            testOutput: null,
            testPassed: null,
          };
        }
        return {
          passed: true,
          diagnosticErrors: [],
          testOutput: null,
          testPassed: null,
        };
      },
      formatResults(result: any) {
        if (result.passed) return "Double-check: All validations passed.";
        return "Diagnostic errors (1):\n  - /tmp/test.ts: Unexpected token";
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
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts"}',
          toolCallId: "call_1",
          toolName: "write_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Now fixed." },
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
    expect(result.iterations).toBe(4);

    const finalAssistant = [...result.messages]
      .reverse()
      .find((m) => m.role === MessageRole.ASSISTANT && (!m.toolCalls || m.toolCalls.length === 0));
    expect(finalAssistant?.content).toBe("Now fixed.");

    const unresolvedValidationMessage = result.messages.find(
      (m) =>
        m.role === MessageRole.SYSTEM &&
        m.content?.includes("Double-check still failing"),
    );
    expect(unresolvedValidationMessage).toBeDefined();
  });
  it("exits loop when double-check keeps failing and LLM only produces text (no infinite loop)", async () => {
    // Regression: unresolvedDoubleCheckFailure nudge fires once then clears,
    // so the second text-only response exits immediately.
    const mockDoubleCheck = {
      isEnabled: () => true,
      async captureBaseline() { return {}; },
      async check() {
        // Always fail — simulates persistent LSP errors (e.g., pre-existing AOSP errors)
        return {
          passed: false,
          diagnosticErrors: ["/tmp/file.cpp: error: unknown type"],
          testOutput: null,
          testPassed: null,
        };
      },
      formatResults() {
        return "Diagnostic errors (1):\n  - /tmp/file.cpp: error: unknown type";
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
        artifacts: ["/tmp/file.cpp"],
      }),
    };

    // LLM: one tool call, then only text responses
    const provider = createMockProvider([
      [
        { type: "tool_call", content: '{"path": "/tmp/file.cpp"}', toolCallId: "c0", toolName: "write_file" },
        { type: "done", content: "" },
      ],
      // Text-only: triggers double-check nudge (one-shot, clears flag)
      [{ type: "text", content: "I see errors, let me analyze..." }, { type: "done", content: "" }],
      // Text-only: double-check flag cleared, exits as final answer
      [{ type: "text", content: "Here are my conclusions." }, { type: "done", content: "" }],
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

    const result = await loop.run("Fix the file");

    // Must terminate — nudge once then exit
    expect(result.status).toBe("success");
    // 1 (tool call) + 2 (nudge) + 3 (exit) = 3 iterations
    expect(result.iterations).toBe(3);
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
  it("executes multiple explore delegate calls in parallel when emitted in one turn", async () => {
    const executionLog: Array<{ callId: string; event: "start" | "end"; time: number }> = [];
    const batchIds = new Set<string>();
    const batchSizes = new Set<number>();
    const baseTime = Date.now();

    const provider = createMockProvider([
      [
        { type: "tool_call", content: '{"agent_type":"explore","request":{"objective":"docs lane"}}', toolCallId: "d1", toolName: "delegate" },
        { type: "tool_call", content: '{"agent_type":"explore","request":{"objective":"runtime lane"}}', toolCallId: "d2", toolName: "delegate" },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Delegates complete." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register({
      name: "delegate",
      description: "Mock delegate",
      category: "workflow",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async (params, context) => {
        const callId = String((params["request"] as Record<string, unknown>)["objective"] ?? "unknown");
        if (context.batchId) batchIds.add(context.batchId);
        if (context.batchSize) batchSizes.add(context.batchSize);
        executionLog.push({ callId, event: "start", time: Date.now() - baseTime });
        await new Promise((resolve) => setTimeout(resolve, 100));
        executionLog.push({ callId, event: "end", time: Date.now() - baseTime });
        return {
          success: true,
          output: `${callId} complete`,
          error: null,
          artifacts: [],
        };
      },
    });

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Investigations should be decomposed into evidence lanes.",
      repoRoot: "/tmp",
    });

    const start = Date.now();
    const result = await loop.run("Investigate contradictions across docs and runtime");
    const elapsed = Date.now() - start;

    expect(result.status).toBe("success");
    const toolMessages = result.messages.filter((m) => m.role === MessageRole.TOOL);
    expect(toolMessages).toHaveLength(2);
    expect(toolMessages[0]!.content).toBe("docs lane complete");
    expect(toolMessages[1]!.content).toBe("runtime lane complete");
    expect(elapsed).toBeLessThan(300);

    const starts = executionLog.filter((entry) => entry.event === "start");
    expect(starts).toHaveLength(2);
    const maxStartDiff = Math.max(...starts.map((entry) => entry.time)) - Math.min(...starts.map((entry) => entry.time));
    expect(maxStartDiff).toBeLessThan(50);
    expect(batchIds.size).toBe(1);
    expect(batchSizes).toEqual(new Set([2]));
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
  it("emits message:tool events for coalesced skipped calls", async () => {
    const planTool: ToolSpec = {
      name: "update_plan",
      description: "Update plan",
      category: "workflow",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "Plan updated",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
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

    // Collect message:tool events from the bus
    const toolEvents: Array<{ content: string; toolCallId: string }> = [];
    bus.on("message:tool", (evt: { content: string; toolCallId: string }) => {
      toolEvents.push(evt);
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

    await loop.run("Create a plan");

    // All 3 tool call IDs should have message:tool events
    expect(toolEvents.length).toBe(3);
    // Skipped calls should emit with "Skipped" content
    expect(toolEvents[0]!.toolCallId).toBe("p1");
    expect(toolEvents[0]!.content).toContain("Skipped");
    expect(toolEvents[1]!.toolCallId).toBe("p2");
    expect(toolEvents[1]!.content).toContain("Skipped");
    // Executed call should emit with real output
    expect(toolEvents[2]!.toolCallId).toBe("p3");
    expect(toolEvents[2]!.content).toContain("Plan updated");
  });
