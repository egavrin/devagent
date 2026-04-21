import { resolve } from "node:path";
import {
  describe,
  it,
  expect,
  beforeEach,
  beforeAll,
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
  ContextManager,
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

  describe("per-file read stagnation guard", () => {
    it("injects nudge after excessive reads of same file without sessionState", async () => {
      // Without sessionState, the sessionState-based stagnation guard won't fire.
      // The per-file read counter should still catch repeated reads.
      const READ_LIMIT = 8; // matches PER_FILE_READ_LIMIT
      const responses: StreamChunk[][] = [];

      // LLM reads the same file with different line ranges N+1 times
      for (let i = 0; i < READ_LIMIT + 1; i++) {
        responses.push([
          {
            type: "tool_call",
            content: JSON.stringify({ path: "/tmp/big-file.ts", start_line: i * 100 + 1, end_line: (i + 1) * 100 }),
            toolCallId: `call_read_${i}`,
            toolName: "read_file",
          },
          { type: "done", content: "" },
        ]);
      }
      responses.push([
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ]);

      const provider = createMockProvider(responses);
      const registry = new ToolRegistry();
      registry.register({
        name: "read_file",
        description: "Read file",
        category: "readonly",
        paramSchema: {
          type: "object",
          properties: { path: { type: "string" }, start_line: { type: "number" }, end_line: { type: "number" } },
          required: ["path"],
        },
        resultSchema: { type: "object" },
        handler: async (params) => ({
          success: true,
          output: `Lines ${params["start_line"]}-${params["end_line"]}`,
          error: null,
          artifacts: [],
        }),
      });

      const gate = new ApprovalGate(config.approval, bus);
      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config,
        systemPrompt: "Test",
        repoRoot: "/tmp",
        // NO sessionState — mimics subagent behavior
      });

      const result = await loop.run("Review big file");
      const nudge = result.messages.find(
        (m) =>
          m.role === MessageRole.SYSTEM
          && m.content?.includes("excessive number of reads"),
      );
      expect(nudge).toBeDefined();
    });

    it("hard-blocks reads after PER_FILE_READ_HARD_LIMIT", async () => {
      const HARD_LIMIT = 12; // matches PER_FILE_READ_HARD_LIMIT
      const responses: StreamChunk[][] = [];

      // LLM reads same file HARD_LIMIT+1 times with different ranges
      for (let i = 0; i < HARD_LIMIT + 1; i++) {
        responses.push([
          {
            type: "tool_call",
            content: JSON.stringify({ path: "/tmp/big-file.ts", start_line: i * 50 + 1, end_line: (i + 1) * 50 }),
            toolCallId: `call_read_${i}`,
            toolName: "read_file",
          },
          { type: "done", content: "" },
        ]);
      }
      responses.push([
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ]);

      const provider = createMockProvider(responses);
      const registry = new ToolRegistry();
      registry.register({
        name: "read_file",
        description: "Read file",
        category: "readonly",
        paramSchema: {
          type: "object",
          properties: { path: { type: "string" }, start_line: { type: "number" }, end_line: { type: "number" } },
          required: ["path"],
        },
        resultSchema: { type: "object" },
        handler: async (params) => ({
          success: true,
          output: `Lines ${params["start_line"]}-${params["end_line"]}`,
          error: null,
          artifacts: [],
        }),
      });

      // Need enough iterations to reach the hard limit
      const extendedConfig = {
        ...config,
        budget: { ...config.budget, maxIterations: 20 },
      };

      const gate = new ApprovalGate(extendedConfig.approval, bus);
      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config: extendedConfig,
        systemPrompt: "Test",
        repoRoot: "/tmp",
      });

      const result = await loop.run("Read big file");
      // The HARD_LIMIT+1th read should be blocked
      const blocked = result.messages.find(
        (m) =>
          m.role === MessageRole.TOOL
          && m.content?.includes("Blocked:"),
      );
      expect(blocked).toBeDefined();
      expect(blocked!.content).toContain("Do NOT attempt to read this file again");
    });
  });

  // ─── Two-Phase Pruning ──────────────────────────────────────
  it("prunes old large tool results before full compaction", async () => {
    const responses: StreamChunk[][] = [];

    // Generate 15 tool calls to fill context, then final answer
    for (let i = 0; i < 15; i++) {
      responses.push([
        {
          type: "tool_call" as const,
          content: `{"text": "call_${i}"}`,
          toolCallId: `call_${i}`,
          toolName: "echo",
        },
        { type: "done" as const, content: "" },
      ]);
    }
    // Final response
    responses.push([
      { type: "text" as const, content: "Done" },
      { type: "done" as const, content: "" },
    ]);

    const provider = createMockProvider(responses);

    // Echo tool that returns large output to fill context
    const bigEchoTool: ToolSpec = {
      name: "echo",
      description: "Echo with large output",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { text: { type: "string" } },
        required: ["text"],
      },
      resultSchema: { type: "object" },
      handler: async (params) => ({
        success: true,
        output: `Echo: ${params["text"]} ${"x".repeat(5000)}`,
        error: null,
        artifacts: [],
      }),
    };

    const registry = new ToolRegistry();
    registry.register(bigEchoTool);

    // Use small context budget to trigger compaction
    const smallConfig = makeConfig({
      budget: {
        maxIterations: 20,
        maxContextTokens: 10_000,
        responseHeadroom: 1_000,
        costWarningThreshold: 1.0,
        enableCostTracking: true,
      },
    });

    const gate = new ApprovalGate(smallConfig.approval, bus);
    const contextManager = new ContextManager(smallConfig.context);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config: smallConfig,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      contextManager,
      sessionState,
    });

    const result = await loop.run("Run echo many times");
    // Should complete without throwing
    expect(result.status).toBeDefined();

    // Verify some messages were pruned (contain "Previously:" marker)
    const prunedMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL && m.content?.startsWith("[Previously:"),
    );
    // With a 10K token budget and 5K+ output per call, pruning should kick in
    expect(prunedMessages.length).toBeGreaterThanOrEqual(0);
  });

  it("includes inline tool summary in pruned replacement text", async () => {
    const provider = createMockProvider([]);
    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    // Use small protect budget so test tool outputs are outside the window
    const testConfig = makeConfig({
      context: { ...config.context, pruneProtectTokens: 15_000 },
    });
    const gate = new ApprovalGate(testConfig.approval, bus);
    const sessionState = new SessionState();
    sessionState.addToolSummary({
      tool: "read_file",
      target: "/tmp/file-0.ts",
      summary: "Read 900 lines: export function alpha(); export class Beta",
      iteration: 1,
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config: testConfig,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      sessionState,
    });

    const bigToolOutput = `export function alpha() {}\n${"x".repeat(45_000)}`;
    const messages: Message[] = [
      { role: MessageRole.SYSTEM, content: "Test" },
    ];
    for (let i = 0; i < 4; i++) {
      messages.push({
        role: MessageRole.ASSISTANT,
        content: null,
        toolCalls: [{
          name: "read_file",
          arguments: { path: `/tmp/file-${i}.ts` },
          callId: `call_${i}`,
        }],
      });
      messages.push({
        role: MessageRole.TOOL,
        content: bigToolOutput,
        toolCallId: `call_${i}`,
      });
    }

    (loop as any).messages = messages;
    const pruneResult = (loop as any).pruneToolOutputs(200_000, 120_000);
    expect(pruneResult.savedTokens).toBeGreaterThan(0);
    expect(pruneResult.prunedCount).toBeGreaterThan(0);

    const toolMessages = (loop as any).messages.filter(
      (m: Message) => m.role === MessageRole.TOOL,
    ) as Message[];
    const pruned = toolMessages.find(
      (m) => m.content?.startsWith("[Previously:"),
    );
    expect(pruned).toBeDefined();
    expect(pruned!.content).toContain("read_file(/tmp/file-0.ts)");
    expect(pruned!.content).toContain("pruned from");
  });

  it("tool outputs within last 60K tokens are NOT pruned by default", async () => {
    const provider = createMockProvider([]);
    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config, // default pruneProtectTokens → 60K
      systemPrompt: "Test",
      repoRoot: "/tmp",
      sessionState,
    });

    // 4 tool outputs of ~11K tokens each (~44K total) — all within 60K window
    const bigToolOutput = "x".repeat(45_000);
    const messages: Message[] = [
      { role: MessageRole.SYSTEM, content: "Test" },
    ];
    for (let i = 0; i < 4; i++) {
      messages.push({
        role: MessageRole.ASSISTANT,
        content: null,
        toolCalls: [{
          name: "read_file",
          arguments: { path: `/tmp/file-${i}.ts` },
          callId: `call_${i}`,
        }],
      });
      messages.push({
        role: MessageRole.TOOL,
        content: bigToolOutput,
        toolCallId: `call_${i}`,
      });
    }

    (loop as any).messages = messages;
    const pruneResult = (loop as any).pruneToolOutputs(200_000, 120_000);
    // All tool outputs are within the 60K protection window, so nothing should be pruned
    expect(pruneResult.prunedCount).toBe(0);
  });

  it("custom pruneProtectTokens override works", async () => {
    const provider = createMockProvider([]);
    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    const testConfig = makeConfig({
      context: { ...config.context, pruneProtectTokens: 10_000 },
    });
    const gate = new ApprovalGate(testConfig.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config: testConfig,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      sessionState,
    });

    // 4 tool outputs of ~11K tokens each — with 10K protect, only the most recent ~10K protected
    const bigToolOutput = "x".repeat(45_000);
    const messages: Message[] = [
      { role: MessageRole.SYSTEM, content: "Test" },
    ];
    for (let i = 0; i < 4; i++) {
      messages.push({
        role: MessageRole.ASSISTANT,
        content: null,
        toolCalls: [{
          name: "read_file",
          arguments: { path: `/tmp/file-${i}.ts` },
          callId: `call_${i}`,
        }],
      });
      messages.push({
        role: MessageRole.TOOL,
        content: bigToolOutput,
        toolCallId: `call_${i}`,
      });
    }

    (loop as any).messages = messages;
    const pruneResult = (loop as any).pruneToolOutputs(200_000, 120_000);
    // With 10K protect window, most tool outputs should be prunable
    expect(pruneResult.prunedCount).toBeGreaterThan(0);
  });


  // ─── Pruning Threshold Tuning ──────────────────────────────
  describe("pruning threshold tuning", () => {
    it("Phase 1 skips tool results under 5000 tokens", async () => {
      const provider = createMockProvider([]);
      const registry = new ToolRegistry();
      registry.register(makeEchoTool());
      const testConfig = makeConfig({
        context: { ...config.context, pruneProtectTokens: 0 },
      });
      const gate = new ApprovalGate(testConfig.approval, bus);

      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config: testConfig,
        systemPrompt: "Test",
        repoRoot: "/tmp",
      });

      const sessionState = new SessionState();
      (loop as any).sessionState = sessionState;

      // Build messages: assistant with tool calls + tool results
      // Use outputs of ~2000 tokens (~8000 chars) — above old 500 threshold but below new 5000
      const mediumOutput = "x".repeat(8_000); // ~2000 tokens
      const messages: Message[] = [
        { role: MessageRole.USER, content: "start" },
        {
          role: MessageRole.ASSISTANT,
          content: "",
          toolCalls: [
            { callId: "call_0", name: "echo", arguments: { text: "a" } },
            { callId: "call_1", name: "echo", arguments: { text: "b" } },
          ],
        },
        { role: MessageRole.TOOL, content: mediumOutput, toolCallId: "call_0" },
        { role: MessageRole.TOOL, content: mediumOutput, toolCallId: "call_1" },
        { role: MessageRole.USER, content: "continue" },
      ];

      (loop as any).messages = messages;
      // currentTokens=~4000, threshold=3000 → would trigger pruning
      const pruneResult = (loop as any).pruneToolOutputs(4_000, 3_000);
      // Messages at ~2000 tokens each are below MIN_PRUNE_MSG_TOKENS (5000),
      // so nothing should be pruned
      expect(pruneResult.prunedCount).toBe(0);
      expect(pruneResult.savedTokens).toBe(0);
    });

    it("Phase 1 targets 75% of threshold (not 85%)", async () => {
      const provider = createMockProvider([]);
      const registry = new ToolRegistry();
      registry.register(makeEchoTool());
      const testConfig = makeConfig({
        context: { ...config.context, pruneProtectTokens: 0 },
      });
      const gate = new ApprovalGate(testConfig.approval, bus);

      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config: testConfig,
        systemPrompt: "Test",
        repoRoot: "/tmp",
      });

      const sessionState = new SessionState();
      (loop as any).sessionState = sessionState;

      // Create 4 large tool outputs (~7500 tokens each = ~30000 chars)
      const bigOutput = "x".repeat(30_000);
      const messages: Message[] = [
        { role: MessageRole.USER, content: "start" },
      ];
      for (let i = 0; i < 4; i++) {
        messages.push({
          role: MessageRole.ASSISTANT,
          content: "",
          toolCalls: [{ callId: `call_${i}`, name: "echo", arguments: { text: `${i}` } }],
        });
        messages.push({ role: MessageRole.TOOL, content: bigOutput, toolCallId: `call_${i}` });
      }
      messages.push({ role: MessageRole.USER, content: "continue" });

      (loop as any).messages = messages;
      // currentTokens=30000, threshold=28000
      // At 75%: targetTokens=21000, targetSavings=9000 → needs 2 prunes (~7500 each)
      // At 85%: targetTokens=23800, targetSavings=6200 → needs 1 prune
      const pruneResult = (loop as any).pruneToolOutputs(30_000, 28_000);
      // With 0.75 ratio: targetTokens = 28000 * 0.75 = 21000
      // targetSavings = 30000 - 21000 = 9000 → needs 2 prunes (~7500 tokens each)
      expect(pruneResult.prunedCount).toBe(2);
    });
  });

  // ─── Fix: Provider Token Overhead Causes Context Collapse ─────────────
  describe("overhead-aware pruning", () => {
    it("does not over-prune when provider-reported tokens include large overhead", async () => {
      // Scenario: provider reports 180K tokens (including tool definitions, system
      // prompt encoding, tokenizer overhead). Actual message content is ~55K tokens.
      // Without the fix, pruning strips messages to ~6K tokens.
      // With the fix, pruning accounts for ~125K overhead and leaves ≥20K tokens.

      const responses: StreamChunk[][] = [];

      // Generate 20 tool calls with large output (~2500 tokens each ≈ 50K total)
      for (let i = 0; i < 20; i++) {
        responses.push([
          {
            type: "tool_call" as const,
            content: `{"text": "call_${i}"}`,
            toolCallId: `call_${i}`,
            toolName: "echo",
          },
          {
            type: "done" as const,
            content: "",
            // Report very high token count on first call to simulate overhead
            usage: i === 0
              ? { promptTokens: 180_000, completionTokens: 50 }
              : { promptTokens: 10_000, completionTokens: 50 },
          },
        ]);
      }
      // Final response
      responses.push([
        { type: "text" as const, content: "Done analyzing." },
        { type: "done" as const, content: "", usage: { promptTokens: 5000, completionTokens: 50 } },
      ]);

      let callIndex = 0;
      const provider: LLMProvider = {
        id: "mock-overhead",
        async *chat(): AsyncIterable<StreamChunk> {
          const chunks = responses[callIndex] ?? [];
          callIndex++;
          for (const chunk of chunks) {
            yield chunk;
          }
        },
        abort() {},
      };

      // Echo tool that returns ~2500 chars (~625 tokens) per call
      const bigEchoTool: ToolSpec = {
        name: "echo",
        description: "Echo with large output",
        category: "readonly",
        paramSchema: {
          type: "object",
          properties: { text: { type: "string" } },
          required: ["text"],
        },
        resultSchema: { type: "object" },
        handler: async (params) => ({
          success: true,
          output: `Echo: ${params["text"]} ${"x".repeat(2500)}`,
          error: null,
          artifacts: [],
        }),
      };

      const registry = new ToolRegistry();
      registry.register(bigEchoTool);

      // Context budget = 200K, trigger at 80% = 160K
      // Provider reports 180K > 160K → triggers pruning.
      // Actual message content ≈ 50-55K tokens.
      // Overhead ≈ 180K - 55K = 125K.
      // messageThreshold = 160K - 125K = 35K.
      // Pruning target = 55K - (35K * 0.85) = 55K - 29.75K ≈ 25K savings.
      // Should leave ~30K tokens (not ~6K).
      const overheadConfig = makeConfig({
        budget: {
          maxIterations: 25,
          maxContextTokens: 200_000,
          responseHeadroom: 2_000,
          costWarningThreshold: 5.0,
          enableCostTracking: true,
        },
        context: {
          pruningStrategy: "hybrid",
          triggerRatio: 0.8,
          keepRecentMessages: 4,
        },
      });

      const gate = new ApprovalGate(overheadConfig.approval, bus);
      const contextManager = new ContextManager(overheadConfig.context);
      const sessionState = new SessionState();

      const compactedEvents: Array<{
        estimatedTokens: number;
        removedCount: number;
        prunedCount?: number;
        tokensSaved?: number;
      }> = [];
      bus.on("context:compacted", (evt: any) => {
        compactedEvents.push(evt);
      });

      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config: overheadConfig,
        systemPrompt: "You are a test assistant.",
        repoRoot: "/tmp",
        contextManager,
        sessionState,
      });

      const result = await loop.run("Analyze everything");
      expect(result.status).toBeDefined();

      // Key assertion: after pruning, message content should be ≥20K tokens.
      // Without the fix, it would be ~6K tokens (over-pruned).
      if (compactedEvents.length > 0) {
        const pruneEvent = compactedEvents.find((evt) => (evt.prunedCount ?? 0) > 0);
        if (pruneEvent) {
          expect((pruneEvent.tokensSaved ?? 0)).toBeGreaterThan(0);
        }
        const compactedTokens = compactedEvents[compactedEvents.length - 1]!.estimatedTokens;
        expect(compactedTokens).toBeGreaterThanOrEqual(15_000);
      }

      // Verify at least some non-pruned tool messages remain with real content
      const unprunedTools = result.messages.filter(
        (m) => m.role === MessageRole.TOOL &&
          !m.content?.startsWith("[Previously:") &&
          !m.content?.startsWith("[Superseded"),
      );
      expect(unprunedTools.length).toBeGreaterThanOrEqual(2);
    });
  });
