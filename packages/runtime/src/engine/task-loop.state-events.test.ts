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
  ToolSpec,
  StreamChunk,
  ToolContext,
} from "../core/index.js";
import {
  AgentType,
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

  it("records artifacts in session state for successful mutating tools", async () => {
    const writeTool: ToolSpec = {
      name: "write_file",
      description: "Write a file",
      category: "mutating",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "Written src/new.ts",
        error: null,
        artifacts: ["/repo/src/new.ts"],
      }),
    };

    const sessionState = new SessionState();
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "src/new.ts", "content": "export const value = 1;"}',
          toolCallId: "call_0",
          toolName: "write_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done." },
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
      repoRoot: "/repo",
      sessionState,
    });

    await loop.run("Write a file");

    expect(sessionState.getModifiedFiles()).toContain("/repo/src/new.ts");
  });

  it("records artifacts in session state when success=false but artifacts non-empty (batch partial write)", async () => {
    // Simulate batch replace_in_file with partial failure:
    // success=false (some pairs failed) but artifacts=[filePath] (file was written)
    const partialTool: ToolSpec = {
      name: "replace_in_file",
      description: "Replace in file",
      category: "mutating",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: false,
        output: "Applied 2 replacement(s) in src/datashare.rs:\n  ✓ ok\n  ✗ fail",
        error: "Some replacements failed",
        artifacts: ["/repo/src/datashare.rs"],
      }),
    };

    const sessionState = new SessionState();

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "src/datashare.rs", "replacements": [{"search":"a","replace":"b"}]}',
          toolCallId: "call_0",
          toolName: "replace_in_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(partialTool);
    const gate = new ApprovalGate(config.approval, bus);

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/repo",
      sessionState,
    });

    await loop.run("Replace descriptors");

    // Even though success=false, the file WAS modified (artifacts non-empty)
    const modified = sessionState.getModifiedFiles();
    expect(modified).toContain("/repo/src/datashare.rs");

    // Tool summary should also be recorded
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
  });

  it("does not record external artifacts as modified files and keys fetch_url summaries by URL", async () => {
    const fetchTool: ToolSpec = {
      name: "fetch_url",
      description: "Fetch a URL",
      category: "external",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "Saved binary response",
        error: null,
        artifacts: ["/tmp/devagent-fetch-url/session/download.bin"],
      }),
    };

    const sessionState = new SessionState();
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"url": "https://example.com/report.pdf", "save_binary": true}',
          toolCallId: "call_0",
          toolName: "fetch_url",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(fetchTool);
    const gate = new ApprovalGate(config.approval, bus);

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/repo",
      sessionState,
    });

    await loop.run("Download a report");

    expect(sessionState.getModifiedFiles()).toEqual([]);
    const summaries = sessionState.getToolSummaries();
    expect(summaries).toHaveLength(1);
    expect(summaries[0]!.tool).toBe("fetch_url");
    expect(summaries[0]!.target).toBe("https://example.com/report.pdf");
  });



  // ─── Bug fix: emit bus events for tool call + tool result messages ───────
  it("emits message:assistant with toolCalls when LLM returns tool calls", async () => {
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
        { type: "text", content: "Echo result." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    const gate = new ApprovalGate(config.approval, bus);

    const assistantEvents: Array<{
      content: string;
      partial: boolean;
      toolCalls?: ReadonlyArray<{ name: string; callId: string }>;
    }> = [];
    bus.on("message:assistant", (event) => {
      assistantEvents.push(event);
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/repo",
    });

    await loop.run("Say hello");

    // Should have at least one assistant event with toolCalls
    const withToolCalls = assistantEvents.filter(
      (e) => !e.partial && e.toolCalls && e.toolCalls.length > 0,
    );
    expect(withToolCalls.length).toBeGreaterThanOrEqual(1);
    expect(withToolCalls[0]!.toolCalls![0]!.name).toBe("echo");
    expect(withToolCalls[0]!.toolCalls![0]!.callId).toBe("call_0");
  });

  it("emits message:tool after each tool execution", async () => {
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
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    const gate = new ApprovalGate(config.approval, bus);

    const toolEvents: Array<{
      role: string;
      content: string;
      toolCallId: string;
    }> = [];
    bus.on("message:tool", (event) => {
      toolEvents.push(event);
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/repo",
    });

    await loop.run("Say hello");

    expect(toolEvents.length).toBe(1);
    expect(toolEvents[0]!.role).toBe("tool");
    expect(toolEvents[0]!.toolCallId).toBe("call_0");
    expect(toolEvents[0]!.content).toContain("hello");
  });

  it("emits typed file edit previews on tool:after when a tool returns metadata.fileEdits", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path":"src/new.ts","content":"export const x = 1;\\n"}',
          toolCallId: "call_0",
          toolName: "write_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register({
      name: "write_file",
      description: "Mock write file",
      category: "mutating",
      paramSchema: { type: "object", properties: {}, required: [] },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "Wrote file",
        error: null,
        artifacts: ["src/new.ts"],
        metadata: {
          fileEdits: [
            {
              path: "src/new.ts",
              kind: "create",
              additions: 1,
              deletions: 0,
              unifiedDiff: "--- /dev/null\n+++ b/src/new.ts\n@@ -0,0 +1,1 @@\n+export const x = 1;",
              truncated: false,
              before: "",
              after: "export const x = 1;\n",
            },
            {
              path: "src/extra-1.ts",
              kind: "create",
              additions: 1,
              deletions: 0,
              unifiedDiff: "--- /dev/null\n+++ b/src/extra-1.ts\n@@ -0,0 +1,1 @@\n+1",
              truncated: false,
            },
            {
              path: "src/extra-2.ts",
              kind: "create",
              additions: 1,
              deletions: 0,
              unifiedDiff: "--- /dev/null\n+++ b/src/extra-2.ts\n@@ -0,0 +1,1 @@\n+2",
              truncated: false,
            },
            {
              path: "src/extra-3.ts",
              kind: "create",
              additions: 1,
              deletions: 0,
              unifiedDiff: "--- /dev/null\n+++ b/src/extra-3.ts\n@@ -0,0 +1,1 @@\n+3",
              truncated: false,
            },
          ],
        },
      }),
    });
    const gate = new ApprovalGate(config.approval, bus);

    const toolAfterEvents: Array<{
      readonly name: string;
      readonly fileEdits?: ReadonlyArray<{
        readonly path: string;
        readonly additions: number;
        readonly before?: string;
        readonly after?: string;
        readonly structuredDiff?: { readonly hunks: ReadonlyArray<unknown> };
      }>;
      readonly fileEditHiddenCount?: number;
    }> = [];
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
      repoRoot: "/repo",
    });

    await loop.run("Create a file");

    expect(toolAfterEvents).toHaveLength(1);
    expect(toolAfterEvents[0]?.fileEdits).toHaveLength(3);
    expect(toolAfterEvents[0]?.fileEdits?.[0]).toEqual(
      expect.objectContaining({
        path: "src/new.ts",
        additions: 1,
        before: "",
        after: "export const x = 1;\n",
        structuredDiff: expect.objectContaining({
          hunks: expect.any(Array),
        }),
      }),
    );
    expect(toolAfterEvents[0]?.fileEditHiddenCount).toBe(1);
  });

  it("emits summary-only delegate tool messages while keeping full delegate output in loop history", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"agent_type":"explore","request":{"objective":"Inspect docs lane","laneLabel":"docs/spec","scope":"docs","constraints":[],"exclusions":[],"successCriteria":[],"parentContext":"Need focused evidence"}}',
          toolCallId: "call_0",
          toolName: "delegate",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register({
      name: "delegate",
      description: "Mock delegate",
      category: "workflow",
      paramSchema: { type: "object", properties: {}, required: [] },
      resultSchema: { type: "object", properties: {} },
      async handler() {
        return {
          success: true,
          output: "Subagent (explore) completed [6 iterations, 100+20 tokens]:\n\n{\"answer\":\"full child payload\"}",
          error: null,
          artifacts: [],
          metadata: {
            agentMeta: {
              agentId: "root-sub-1",
              parentId: "root",
              depth: 1,
              agentType: AgentType.EXPLORE,
            },
            delegateSummary: {
              agentId: "root-sub-1",
              agentType: "explore",
              laneLabel: "docs/spec",
              durationMs: 4500,
              iterations: 6,
              quality: {
                score: 0.81,
                completeness: "partial",
              },
            },
          },
        };
      },
    });
    const gate = new ApprovalGate(config.approval, bus);

    const toolEvents: Array<{
      role: string;
      content: string;
      toolCallId: string;
      toolName?: string;
      summaryOnly?: boolean;
    }> = [];
    bus.on("message:tool", (event) => {
      toolEvents.push(event);
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/repo",
    });

    const result = await loop.run("Delegate the docs lane");

    expect(toolEvents).toHaveLength(1);
    expect(toolEvents[0]!.toolName).toBe("delegate");
    expect(toolEvents[0]!.summaryOnly).toBe(true);
    expect(toolEvents[0]!.content).toContain("root-sub-1");
    expect(toolEvents[0]!.content).toContain("docs/spec");
    expect(toolEvents[0]!.content).not.toContain("full child payload");

    const toolMessages = result.messages.filter((message) => message.role === MessageRole.TOOL);
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0]!.content).toContain("full child payload");
  });

  it("includes agent ownership on child message events", async () => {
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
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const receivedContexts: ToolContext[] = [];
    const registry = new ToolRegistry();
    registry.register({
      ...makeEchoTool(),
      handler: async (params, context) => {
        receivedContexts.push(context);
        return {
          success: true,
          output: String(params["text"] ?? ""),
          error: null,
          artifacts: [],
        };
      },
    });
    const gate = new ApprovalGate(config.approval, bus);

    const userEvents: Array<{ agentId?: string; parentAgentId?: string | null }> = [];
    const assistantEvents: Array<{ agentId?: string; parentAgentId?: string | null }> = [];
    const toolEvents: Array<{ agentId?: string; parentAgentId?: string | null }> = [];
    bus.on("message:user", (event) => {
      userEvents.push(event);
    });
    bus.on("message:assistant", (event) => {
      assistantEvents.push(event);
    });
    bus.on("message:tool", (event) => {
      toolEvents.push(event);
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/repo",
      agentContext: {
        agentId: "root-sub-1",
        parentAgentId: "root",
        depth: 1,
        agentType: AgentType.EXPLORE,
      },
    });

    await loop.run("Say hello");

    expect(userEvents.some((event) => event.agentId === "root-sub-1")).toBe(true);
    expect(assistantEvents.some((event) => event.agentId === "root-sub-1")).toBe(true);
    expect(toolEvents.some((event) => event.agentId === "root-sub-1")).toBe(true);
    expect(toolEvents.every((event) => event.parentAgentId === "root")).toBe(true);
    expect(receivedContexts[0]?.agentId).toBe("root-sub-1");
    expect(receivedContexts[0]?.parentAgentId).toBe("root");
    expect(receivedContexts[0]?.agentType).toBe(AgentType.EXPLORE);
  });



  // ─── Bug fix: wire cost record persistence ───────────────────────────────
  it("emits cost:update with usage from done chunks and accumulates in result", async () => {
    // Provider returns usage data in the done chunk
    const provider: LLMProvider = {
      id: "mock",
      async *chat(): AsyncIterable<StreamChunk> {
        yield { type: "text", content: "Hello world" };
        yield {
          type: "done",
          content: "",
          usage: { promptTokens: 100, completionTokens: 50 },
        };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);

    const costEvents: Array<{
      inputTokens: number;
      outputTokens: number;
      totalCost: number;
      model: string;
    }> = [];
    bus.on("cost:update", (event) => {
      costEvents.push(event);
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/repo",
    });

    const result = await loop.run("Say hello");

    // Bus event emitted
    expect(costEvents.length).toBe(1);
    expect(costEvents[0]!.inputTokens).toBe(100);
    expect(costEvents[0]!.outputTokens).toBe(50);

    // Result cost accumulated
    expect(result.cost.inputTokens).toBe(100);
    expect(result.cost.outputTokens).toBe(50);
  });

  it("computes totalCost from pricing table for known models", async () => {
    const provider: LLMProvider = {
      id: "mock",
      async *chat(): AsyncIterable<StreamChunk> {
        yield { type: "text", content: "Hello" };
        yield {
          type: "done",
          content: "",
          usage: { promptTokens: 1_000_000, completionTokens: 500_000 },
        };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    // Use a known model with defined pricing (unique to anthropic.toml)
    const pricedConfig = makeConfig({
      provider: "anthropic",
      model: "claude-sonnet-4-20250514",
    });
    const gate = new ApprovalGate(pricedConfig.approval, bus);

    const costEvents: Array<{ totalCost: number }> = [];
    bus.on("cost:update", (event) => {
      costEvents.push(event);
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config: pricedConfig,
      systemPrompt: "Test",
      repoRoot: "/repo",
    });

    const result = await loop.run("Say hello");

    // claude-sonnet-4: $3/M input + $15/M output
    // Expected: (1M * 3 + 500K * 15) / 1M = 3.0 + 7.5 = 10.50
    expect(costEvents[0]!.totalCost).toBeCloseTo(10.50);
    expect(result.cost.totalCost).toBeCloseTo(10.50);
  });

  it("accumulates cost across multiple LLM calls", async () => {
    let callIndex = 0;
    const provider: LLMProvider = {
      id: "mock",
      async *chat(): AsyncIterable<StreamChunk> {
        if (callIndex === 0) {
          callIndex++;
          yield {
            type: "tool_call",
            content: '{"text": "hi"}',
            toolCallId: "call_0",
            toolName: "echo",
          };
          yield {
            type: "done",
            content: "",
            usage: { promptTokens: 200, completionTokens: 30 },
          };
        } else {
          yield { type: "text", content: "Done." };
          yield {
            type: "done",
            content: "",
            usage: { promptTokens: 300, completionTokens: 40 },
          };
        }
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    const gate = new ApprovalGate(config.approval, bus);

    const costEvents: Array<{
      inputTokens: number;
      outputTokens: number;
    }> = [];
    bus.on("cost:update", (event) => {
      costEvents.push(event);
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/repo",
    });

    const result = await loop.run("Echo hi");

    // Two cost events (one per LLM call)
    expect(costEvents.length).toBe(2);

    // Total accumulated cost
    expect(result.cost.inputTokens).toBe(500); // 200 + 300
    expect(result.cost.outputTokens).toBe(70); // 30 + 40
  });

  describe("token-aware compaction", () => {
    it("triggers compaction attempt from actual provider tokens even when char estimate is below threshold", async () => {
      // Budget: 1000 tokens, trigger at 80% = 800 (effective budget = 950)
      // The char-based estimate will be low, but the provider reports 850 prompt tokens.
      // This should cause maybeCompactContext to attempt compaction (context:compacting fires).
      const compactionConfig = makeConfig({
        budget: {
          maxIterations: 5,
          maxContextTokens: 1000,
          responseHeadroom: 50,
          costWarningThreshold: 1.0,
          enableCostTracking: true,
        },
        context: {
          pruningStrategy: "sliding_window",
          triggerRatio: 0.8,
          keepRecentMessages: 2,
        },
      });

      let callIndex = 0;
      const provider: LLMProvider = {
        id: "mock",
        async *chat(): AsyncIterable<StreamChunk> {
          if (callIndex === 0) {
            callIndex++;
            yield {
              type: "tool_call",
              content: '{"text": "hi"}',
              toolCallId: "call_0",
              toolName: "echo",
            };
            yield {
              type: "done",
              content: "",
              // Report 850 tokens — above 80% of 950 effective budget
              usage: { promptTokens: 850, completionTokens: 20 },
            };
          } else {
            yield { type: "text", content: "Done." };
            yield {
              type: "done",
              content: "",
              usage: { promptTokens: 200, completionTokens: 10 },
            };
          }
        },
        abort() {},
      };

      const registry = new ToolRegistry();
      registry.register(makeEchoTool());
      const gate = new ApprovalGate(compactionConfig.approval, bus);
      const contextManager = new ContextManager(compactionConfig.context);

      let compactingFired = false;
      bus.on("context:compacting", () => {
        compactingFired = true;
      });

      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config: compactionConfig,
        systemPrompt: "Test",
        repoRoot: "/repo",
        contextManager,
      });

      await loop.run("Test compaction");

      // The key assertion: compaction was *attempted* because provider-reported
      // tokens (850) exceeded the threshold (760), even though the char-based
      // estimate alone would not have triggered it.
      expect(compactingFired).toBe(true);
    });
  });
  it("continues when LLM produces text-only but plan has incomplete steps", async () => {
    // Simulate: LLM sets up a plan via update_plan, then produces a
    // text-only "progress update" response. The loop should detect the
    // incomplete plan and auto-continue instead of exiting.
    const provider = createMockProvider([
      // Iteration 1: tool call to update_plan
      [
        {
          type: "tool_call",
          content: JSON.stringify({
            steps: JSON.stringify([
              { description: "Read files", status: "in_progress" },
              { description: "Analyze code", status: "pending" },
              { description: "Report findings", status: "pending" },
            ]),
          }),
          toolName: "update_plan",
          toolCallId: "call_plan_1",
        },
        { type: "done", content: "" },
      ],
      // Iteration 2: text-only "progress update" (no tool calls)
      [
        { type: "text", content: "Progress: I've set up the plan. Next I'll read the files." },
        { type: "done", content: "" },
      ],
      // Iteration 3: LLM resumes with tool calls after continuation prompt
      [
        {
          type: "tool_call",
          content: JSON.stringify({ text: "reading files" }),
          toolName: "echo",
          toolCallId: "call_echo_1",
        },
        { type: "done", content: "" },
      ],
      // Iteration 4: LLM completes the plan
      [
        {
          type: "tool_call",
          content: JSON.stringify({
            steps: JSON.stringify([
              { description: "Read files", status: "completed" },
              { description: "Analyze code", status: "completed" },
              { description: "Report findings", status: "completed" },
            ]),
          }),
          toolName: "update_plan",
          toolCallId: "call_plan_2",
        },
        { type: "done", content: "" },
      ],
      // Iteration 5: final text response (plan is now complete → exits)
      [
        { type: "text", content: "Here are my findings: everything looks good." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    const sessionState = new SessionState();
    const planTool = (await import("./plan-tool.js")).createPlanTool(bus, () => sessionState);
    registry.register(planTool);
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
      sessionState,
    });

    const result = await loop.run("Review the code");
    // Should have continued past the text-only response
    // 1 (plan setup) + 2 (text continuation) + 3 (echo) + 4 (plan complete) + 5 (final text)
    expect(result.iterations).toBe(5);
  });

  it("exits normally when LLM produces text-only and plan is complete", async () => {
    const provider = createMockProvider([
      // Iteration 1: tool call to update_plan (all completed)
      [
        {
          type: "tool_call",
          content: JSON.stringify({
            steps: JSON.stringify([
              { description: "Read files", status: "completed" },
              { description: "Analyze", status: "completed" },
            ]),
          }),
          toolName: "update_plan",
          toolCallId: "call_plan_1",
        },
        { type: "done", content: "" },
      ],
      // Iteration 2: final text response
      [
        { type: "text", content: "All done! Here is the review." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    const sessionState = new SessionState();
    const planTool = (await import("./plan-tool.js")).createPlanTool(bus, () => sessionState);
    registry.register(planTool);
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
    });

    const result = await loop.run("Review the code");
    // Should exit after text-only (plan is complete)
    expect(result.iterations).toBe(2);
  });

  it("limits plan nudge to a single attempt to prevent infinite loops", async () => {
    // LLM keeps producing text-only responses with an incomplete plan.
    // Plan nudge fires once, then the next text-only exits immediately.
    const responses: StreamChunk[][] = [
      // Iteration 1: set up plan
      [
        {
          type: "tool_call",
          content: JSON.stringify({
            steps: JSON.stringify([
              { description: "Step 1", status: "in_progress" },
              { description: "Step 2", status: "pending" },
            ]),
          }),
          toolName: "update_plan",
          toolCallId: "call_plan_1",
        },
        { type: "done", content: "" },
      ],
      // Iteration 2: text-only → plan nudge fires (single shot)
      [
        { type: "text", content: "Still working on step 1..." },
        { type: "done", content: "" },
      ],
      // Iteration 3: text-only → planNudgeUsed=true, exits as final
      [
        { type: "text", content: "Here are my partial results." },
        { type: "done", content: "" },
      ],
    ];

    const provider = createMockProvider(responses);
    const registry = new ToolRegistry();
    const sessionState = new SessionState();
    const planTool = (await import("./plan-tool.js")).createPlanTool(bus, () => sessionState);
    registry.register(planTool);
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
    });

    const result = await loop.run("Review the code");
    // 1 (plan setup) + 2 (text → nudge) + 3 (text → exit)
    expect(result.iterations).toBe(3);
  });

  it("text-only response without plan exits immediately (no judge)", async () => {
    // Without a plan and without completion judge, text-only = done.
    const provider = createMockProvider([
      // Iteration 1: tool call
      [
        {
          type: "tool_call",
          content: '{"text": "checking"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Iteration 2: text-only → exits immediately
      [
        { type: "text", content: "Here are my findings: everything looks correct." },
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

    const result = await loop.run("Check the code");
    expect(result.iterations).toBe(2);
    expect(result.status).toBe("success");
  });

  it("records tool summaries for readonly tools after execution", async () => {
    const provider = createMockProvider([
      // First response: call read_file (readonly)
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts"}',
          toolCallId: "call_read1",
          toolName: "read_file",
        },
        { type: "done", content: "", usage: { promptTokens: 100, completionTokens: 50 } },
      ],
      // Second response: text-only → finish
      [
        { type: "text", content: "Done reviewing." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    // Register a read_file-like readonly tool
    registry.register({
      name: "read_file",
      description: "Read file contents",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
      },
      resultSchema: { type: "object" },
      handler: async (_params) => ({
        success: true,
        output: "const x = 42;\nexport default x;",
        error: null,
        artifacts: [],
      }),
    });

    const gate = new ApprovalGate(config.approval, bus);
    const ss = new SessionState();
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      sessionState: ss,
      repoRoot: "/tmp",
    });

    await loop.run("Read the file");

    const summaries = ss.getToolSummaries();
    expect(summaries.length).toBe(1);
    expect(summaries[0]!.tool).toBe("read_file");
    expect(summaries[0]!.target).toBe("/tmp/test.ts");
  });

  it("records git_diff path targets as modified files for review continuity", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path":"packages/runtime/src/core/config.ts","staged":true}',
          toolCallId: "call_diff1",
          toolName: "git_diff",
        },
        { type: "done", content: "", usage: { promptTokens: 100, completionTokens: 50 } },
      ],
      [
        { type: "text", content: "Done reviewing." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register({
      name: "git_diff",
      description: "Show git diff",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: {
          path: { type: "string" },
          staged: { type: "boolean" },
        },
        required: [],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "diff --git a/a.ts b/a.ts\n@@ -1 +1 @@\n-a\n+b",
        error: null,
        artifacts: [],
      }),
    });

    const gate = new ApprovalGate(config.approval, bus);
    const ss = new SessionState();
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      sessionState: ss,
      repoRoot: "/tmp",
    });

    await loop.run("Review uncommitted changes");

    expect(ss.getModifiedFiles()).toEqual(["packages/runtime/src/core/config.ts"]);
    expect(ss.getReadonlyCoverage().get("git_diff")).toEqual(["packages/runtime/src/core/config.ts"]);
  });

  it("records only execute_tool_script final stdout in model history", async () => {
    let scriptHandlerCalls = 0;
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: JSON.stringify({
            script: "const diff = await tools.git_diff({ path: 'src/a.ts' }); print('changed=' + diff.success);",
          }),
          toolCallId: "call_script_1",
          toolName: "execute_tool_script",
        },
        { type: "done", content: "", usage: { promptTokens: 100, completionTokens: 50 } },
      ],
      [
        { type: "text", content: "Done reviewing." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register({
      name: "execute_tool_script",
      description: "Execute readonly script",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { script: { type: "string" } },
        required: ["script"],
      },
      resultSchema: { type: "object" },
      handler: async () => {
        scriptHandlerCalls++;
        return {
          success: true,
          output: "changed=true",
          error: null,
          artifacts: [],
        };
      },
    });

    const gate = new ApprovalGate(config.approval, bus);
    const ss = new SessionState();
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      sessionState: ss,
      repoRoot: "/tmp",
    });

    const result = await loop.run("Review changes");
    expect(scriptHandlerCalls).toBe(1);
    const toolMessages = result.messages.filter((m) => m.role === MessageRole.TOOL);
    expect(toolMessages).toHaveLength(1);
    expect(toolMessages[0]!.content).toContain("changed=true");
    expect(toolMessages[0]!.content).not.toContain("diff --git");
    expect(ss.getToolSummaries().filter((s) => s.tool === "git_diff")).toHaveLength(0);
    expect(ss.getToolSummaries().filter((s) => s.tool === "execute_tool_script")).toHaveLength(1);
  });

  it("includes readonly summaries in session state system message", async () => {
    const ss = new SessionState();
    ss.addToolSummary({
      tool: "read_file",
      target: "/src/main.ts",
      summary: "Read 45 lines",
      iteration: 1,
    });
    ss.addToolSummary({
      tool: "git_diff",
      target: "HEAD",
      summary: "5 files changed",
      iteration: 2,
    });

    const msg = ss.toSystemMessage("full");
    expect(msg).toBeDefined();
    expect(msg).toContain("Recent activity");
    expect(msg).toContain("read_file");
    expect(msg).toContain("git_diff");
  });
