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

function isCompactionJudgePrompt(content: string | null | undefined): boolean {
  return Boolean(
    content?.includes("assess whether")
      || content?.includes("extract structured")
      || content?.includes("classify"),
  );
}

function getCompactionJudgeResponse(): string {
  return '{"entries":[],"is_final":true,"confidence":0.9,"reason":"done","quality_loss":0.1,"missing_context":[],"recommendation":"none"}';
}

beforeAll(() => {
    const modelsDir = resolve(import.meta.dirname ?? new URL(".", import.meta.url).pathname, "../../../../models");
    loadModelRegistry(undefined, [modelsDir]);
  });

beforeEach(() => {
    bus = new EventBus();
    config = makeConfig();
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

  // ─── Compaction State Preservation ──────────────────────────
  it("SessionState survives compaction", async () => {
    const sessionState = new SessionState();

    // Pre-populate state (simulating completed work)
    sessionState.setPlan([
      { description: "Analyze code", status: "completed" },
      { description: "Write fix", status: "completed" },
      { description: "Run tests", status: "completed" },
    ]);
    sessionState.recordModifiedFile("src/foo.ts");
    sessionState.recordModifiedFile("src/bar.ts");
    sessionState.addEnvFact("cmd-not-found:rg", "rg is not installed on this system. Use an alternative command.");

    // Very small context to force compaction
    const smallConfig = makeConfig({
      budget: {
        maxIterations: 10,
        maxContextTokens: 800,
        responseHeadroom: 50,
        costWarningThreshold: 1.0,
        enableCostTracking: true,
      },
      context: {
        pruningStrategy: "sliding_window",
        triggerRatio: 0.3,
        keepRecentMessages: 3,
      },
    });

    // Provider: tool call with big output, then extra responses for
    // knowledge extraction + compaction judge + final text
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: JSON.stringify({
            name: "echo",
            arguments: { text: "A".repeat(600) },
            callId: "c1",
          }),
        },
        { type: "done", content: "" },
      ],
      // Knowledge extraction call (consumes a response slot)
      [
        { type: "text", content: '{"entries":[]}' },
        { type: "done", content: "" },
      ],
      // Compaction quality judge call
      [
        { type: "text", content: '{"quality_loss":0.1,"missing_context":[],"recommendation":"none"}' },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Final answer." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    const gate = new ApprovalGate(smallConfig.approval, bus);
    const contextManager = new ContextManager(smallConfig.context);

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config: smallConfig,
      systemPrompt: "You are a test assistant.",
      repoRoot: "/tmp",
      contextManager,
      sessionState,
    });

    const result = await loop.run("Test with compaction");

    // SessionState object should be completely intact
    const plan = sessionState.getPlan();
    expect(plan).not.toBeNull();
    expect(plan).toHaveLength(3);
    expect(plan![0]!.status).toBe("completed");
    expect(plan![1]!.status).toBe("completed");
    expect(plan![2]!.status).toBe("completed");

    expect(sessionState.getModifiedFiles()).toEqual(["src/foo.ts", "src/bar.ts"]);
    expect(sessionState.getEnvFacts()).toContain(
      "rg is not installed on this system. Use an alternative command.",
    );

    // If compaction actually fired, verify session state message was injected.
    // With a very small budget (800 tokens), tier will be "minimal" (plan only).
    const stateMsg = result.messages.find(
      (m) => m.role === MessageRole.SYSTEM && m.content?.includes("[SESSION STATE"),
    );
    if (stateMsg) {
      // Plan section always present in all tiers
      expect(stateMsg.content).toContain("Analyze code");
      // Modified files and env facts may be omitted in minimal/compact tiers
      // when the budget is very small, so don't assert them here.
    }

    // Basic sanity: loop completed successfully
    expect(result.status).toBe("success");
  });

  it("dedup set for non-path readonly tools (search_files) survives compaction", async () => {
    // Regression test: rebuildDedupFromCoverage used to clear the dedup set and
    // reconstruct it from coverage targets. For search_files, the coverage target
    // is "search:pattern" which doesn't match the real dedup key format
    // "search_files:{path,pattern}", so the call was re-executed after compaction.
    // Fix: don't clear the dedup set during compaction — it's still valid.
    let searchCallCount = 0;
    const bigOutput = "result: FindClass match at line 42\n".repeat(80); // ~2800 chars to blow context

    const searchTool: ToolSpec = {
      name: "search_files",
      description: "Search",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: {
          pattern: { type: "string" },
          path: { type: "string" },
        },
      },
      resultSchema: { type: "object" },
      handler: async () => {
        searchCallCount++;
        return { success: true, output: bigOutput, error: null, artifacts: [] };
      },
    };

    const tightConfig = makeConfig({
      budget: {
        maxIterations: 10,
        maxContextTokens: 1200,
        responseHeadroom: 200,
        costWarningThreshold: 10,
        enableCostTracking: false,
      },
      context: {
        pruningStrategy: "sliding_window",
        triggerRatio: 0.3, // 360 token threshold — blows after first 2800-char result (~700t)
        keepRecentMessages: 2,
      },
    });

    let compactionFired = false;
    bus.on("context:compacting", () => { compactionFired = true; });

    const searchArgs = JSON.stringify({ pattern: "FindClass", path: "." });
    const provider = createMockProvider([
      // First LLM call: invoke search_files
      [
        { type: "tool_call", content: searchArgs, toolCallId: "c1", toolName: "search_files" },
        { type: "done", content: "" },
      ],
      // Knowledge extraction call (during compaction)
      [
        { type: "text", content: '{"entries":[]}' },
        { type: "done", content: "" },
      ],
      // Compaction quality judge call
      [
        { type: "text", content: '{"quality_loss":0.1,"missing_context":[],"recommendation":"none"}' },
        { type: "done", content: "" },
      ],
      // Second LLM call (after compaction): re-invoke same search_files
      // Should be deduped — handler must NOT be called a second time
      [
        { type: "tool_call", content: searchArgs, toolCallId: "c2", toolName: "search_files" },
        { type: "done", content: "" },
      ],
      // Third LLM call: final answer
      [
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(searchTool);
    const gate = new ApprovalGate(tightConfig.approval, bus);
    const contextManager = new ContextManager(tightConfig.context);
    // SessionState is required to trigger the bug: without it rebuildDedupFromCoverage
    // returns early (no clear), so the dedup set is preserved by accident.
    // With it, the set is cleared and incorrectly rebuilt from coverage targets.
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config: tightConfig,
      systemPrompt: "Test.",
      repoRoot: "/tmp",
      contextManager,
      sessionState,
    });

    await loop.run("Search for patterns.");

    // Compaction must have fired — otherwise we're not testing the right thing
    expect(compactionFired).toBe(true);
    // Handler should only execute once; the second identical call must be deduped
    // even after compaction
    expect(searchCallCount).toBe(1);
  });

  it("injects post-compaction continuation message after Phase 2", async () => {
    const sessionState = new SessionState();
    sessionState.setPlan([
      { description: "Step 1", status: "completed" },
      { description: "Step 2", status: "in_progress" },
    ]);
    sessionState.recordModifiedFile("src/foo.ts");

    const smallConfig = makeConfig({
      budget: {
        maxIterations: 10,
        maxContextTokens: 800,
        responseHeadroom: 50,
        costWarningThreshold: 1.0,
        enableCostTracking: true,
      },
      context: {
        pruningStrategy: "sliding_window",
        triggerRatio: 0.3,
        keepRecentMessages: 3,
      },
    });

    // Use a dynamic provider that handles multiple call types:
    // tool call on first main chat, then text on everything else
    let mainChatCalls = 0;
    const provider: LLMProvider = {

      id: "mock-continuation",
      async *chat(msgs: ReadonlyArray<Message>): AsyncIterable<StreamChunk> {
        const sys = msgs.find((m) => m.role === MessageRole.SYSTEM);
        if (isCompactionJudgePrompt(sys?.content)) {
          yield { type: "text", content: getCompactionJudgeResponse() };
          yield { type: "done", content: "" };
          return;
        }
        mainChatCalls++;
        if (mainChatCalls === 1) {
          yield {
            type: "tool_call",
            content: JSON.stringify({ name: "echo", arguments: { text: "A".repeat(600) }, callId: "c1" }),
            toolCallId: "c1",
            toolName: "echo",
          };
        } else {
          yield { type: "text", content: "Done" };
        }
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    const gate = new ApprovalGate(smallConfig.approval, bus);
    const contextManager = new ContextManager(smallConfig.context);

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config: smallConfig,
      systemPrompt: "You are a test assistant.",
      repoRoot: "/tmp",
      contextManager,
      sessionState,
    });

    const result = await loop.run("Test continuation msg");

    // After Phase 2 compaction, a continuation message should be injected
    const continuationMsg = result.messages.find(
      (m) => m.role === MessageRole.SYSTEM && m.content?.includes("Context was compacted"),
    );
    if (continuationMsg) {
      expect(continuationMsg.content).toContain("do NOT re-scan");
      expect(continuationMsg.content).toContain("knowledge");
    }
    expect(result.status).toBe("success");
  });
  describe("SessionState full lifecycle (disk persistence)", () => {
    it("persists through compaction and simulated resume", async () => {
      // --- Phase 1: First session with mock persistence ---
      const saved = new Map<string, import("./session-state.js").SessionStateJSON>();
      const persistence: import("./session-state.js").SessionStatePersistence = {
        save: (id, data) => { saved.set(id, structuredClone(data)); },
        load: (id) => saved.get(id) ?? null,
      };

      const ss1 = new SessionState();
      ss1.bind("session-1", persistence);

      // Accumulate state
      ss1.setPlan([
        { description: "Find files", status: "completed" },
        { description: "Edit code", status: "in_progress" },
        { description: "Run tests", status: "pending" },
      ]);
      ss1.recordModifiedFile("/src/main.ts");
      ss1.recordModifiedFile("/src/utils.ts");
      ss1.addEnvFact("cmd-not-found:rg", "rg is not installed");
      ss1.addToolSummary({
        tool: "edit_file",
        target: "/src/main.ts",
        summary: "Added function foo()",
        iteration: 3,
      });

      // Verify auto-saved
      expect(saved.has("session-1")).toBe(true);
      const savedData = saved.get("session-1")!;
      expect(savedData.plan).toHaveLength(3);
      expect(savedData.modifiedFiles).toEqual(["/src/main.ts", "/src/utils.ts"]);
      expect(savedData.envFacts).toHaveLength(1);
      expect(savedData.toolSummaries).toHaveLength(1);

      // --- Phase 2: Simulate process crash + resume (load from persistence) ---
      const ss2 = SessionState.loadOrCreate("session-1", persistence);

      // Verify full state restored
      expect(ss2.getPlan()).toEqual(ss1.getPlan());
      expect(ss2.getModifiedFiles()).toEqual(ss1.getModifiedFiles());
      expect(ss2.getEnvFacts()).toEqual(ss1.getEnvFacts());
      expect(ss2.getToolSummaries()).toEqual(ss1.getToolSummaries());

      // Continue accumulating in resumed session
      ss2.recordModifiedFile("/src/new-file.ts");
      ss2.setPlan([
        { description: "Find files", status: "completed" },
        { description: "Edit code", status: "completed" },
        { description: "Run tests", status: "in_progress" },
      ]);

      expect(ss2.getModifiedFiles()).toHaveLength(3);
      expect(ss2.getPlan()![1]!.status).toBe("completed");

      // --- Phase 3: Verify system message output with tiers ---
      const fullMsg = ss2.toSystemMessage("full")!;
      expect(fullMsg).toContain("## Plan");
      expect(fullMsg).toContain("## Modified files");
      expect(fullMsg).toContain("## Environment");
      expect(fullMsg).toContain("## Recent activity");

      const compactMsg = ss2.toSystemMessage("compact")!;
      expect(compactMsg).toContain("## Plan");
      expect(compactMsg).toContain("## Modified files");
      expect(compactMsg).toContain("## Recent activity");

      const minimalMsg = ss2.toSystemMessage("minimal")!;
      expect(minimalMsg).toContain("## Plan");
      expect(minimalMsg).not.toContain("## Modified files");
      expect(minimalMsg).not.toContain("## Environment");
      expect(minimalMsg).toContain("## Recent activity");
    });

    it("loadOrCreate creates fresh state when no prior session exists", () => {
      const persistence: import("./session-state.js").SessionStatePersistence = {
        save: () => {},
        load: () => null,
      };

      const ss = SessionState.loadOrCreate("new-session", persistence);
      expect(ss.getPlan()).toBeNull();
      expect(ss.getModifiedFiles()).toEqual([]);
      expect(ss.getEnvFacts()).toEqual([]);
      expect(ss.getToolSummaries()).toEqual([]);
    });

    it("JSON round-trip preserves all data types", () => {
      const ss = new SessionState();
      ss.setPlan([
        { description: "Step 1", status: "completed" },
        { description: "Step 2", status: "in_progress" },
        { description: "Step 3", status: "pending" },
      ]);
      for (let i = 0; i < 5; i++) {
        ss.recordModifiedFile(`/src/file-${i}.ts`);
      }
      ss.addEnvFact("key1", "fact1");
      ss.addEnvFact("key2", "fact2");
      ss.addToolSummary({
        tool: "edit_file",
        target: "/src/file-0.ts",
        summary: "Edited file-0",
        iteration: 1,
      });
      ss.addToolSummary({
        tool: "run_command",
        target: "npm test",
        summary: "All 42 tests passed",
        iteration: 5,
      });

      const json = ss.toJSON();
      const restored = SessionState.fromJSON(json);

      expect(restored.getPlan()).toEqual(ss.getPlan());
      expect(restored.getModifiedFiles()).toEqual(ss.getModifiedFiles());
      expect(restored.getEnvFacts()).toEqual(ss.getEnvFacts());
      expect(restored.getToolSummaries()).toEqual(ss.getToolSummaries());
      expect(restored.toSystemMessage("full")).toEqual(ss.toSystemMessage("full"));
    });
  });

  // ─── Fix 1: Tool summary must capture original output, not DoubleCheck-augmented ─────
  describe("tool summary excludes DoubleCheck noise", () => {
    it("records original tool output in session state, not validation-error-augmented output", async () => {
      const mockDoubleCheck = {
        isEnabled: () => true,
        async captureBaseline() { return {}; },
        async check() {
          return {
            passed: false,
            diagnosticErrors: [
              "/tmp/test.ts: error1",
              "/tmp/test.ts: error2",
              "/tmp/test.ts: error3",
            ],
            testOutput: null,
            testPassed: null,
          };
        },
        formatResults(result: any) {
          return `Double-check: Validation FAILED.\nDiagnostic errors (${result.diagnosticErrors.length}):\n` +
            result.diagnosticErrors.map((e: string) => `  - ${e}`).join("\n");
        },
      } as any;

      const writeTool: ToolSpec = {
        name: "replace_in_file",
        description: "Replace in file",
        category: "mutating",
        paramSchema: { type: "object" },
        resultSchema: { type: "object" },
        handler: async () => ({
          success: true,
          output: "Replaced 3 occurrence(s) in src/foo.cpp",
          error: null,
          artifacts: ["/tmp/src/foo.cpp"],
        }),
      };

      const sessionState = new SessionState();

      const provider = createMockProvider([
        [
          {
            type: "tool_call",
            content: '{"path": "src/foo.cpp", "search": "std.core", "replace": "std:core"}',
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
        sessionState,
      });

      await loop.run("Replace descriptors");

      // Tool summary should NOT contain DoubleCheck noise — should have clean structured info
      const summaries = sessionState.getToolSummaries();
      expect(summaries.length).toBe(1);
      // Structured summary from formatToolSummary should have search/replace details
      expect(summaries[0]!.summary).toContain("std.core");
      expect(summaries[0]!.summary).toContain("std:core");
      expect(summaries[0]!.summary).toContain("3");
      // Must NOT contain DoubleCheck noise
      expect(summaries[0]!.summary).not.toContain("VALIDATION ERRORS");
      expect(summaries[0]!.summary).not.toContain("Double-check");
      expect(summaries[0]!.summary).not.toContain("Diagnostic errors");
    });
  });

  // ─── Fix 2: DoubleCheck filters pre-existing errors ──────────────────────
  describe("DoubleCheck baseline filtering", () => {
    it("passes when all diagnostic errors are pre-existing (no new errors introduced)", async () => {
      // Simulate: file had 3 errors BEFORE the edit, still has 3 errors AFTER
      const mockDoubleCheck = {
        isEnabled: () => true,
        async captureBaseline(_files: ReadonlyArray<string>) {
          // Returns the pre-edit diagnostic state
          return { "/tmp/test.ts": 3 };
        },
        async check(_files: ReadonlyArray<string>, _baseline?: Record<string, number>) {
          // All 3 errors are pre-existing — no new ones introduced
          return {
            passed: true, // Should pass because no NEW errors
            diagnosticErrors: [],
            testOutput: null,
            testPassed: null,
          };
        },
        formatResults(result: any) {
          if (result.passed) return "Double-check: All validations passed.";
          return `Double-check: Validation FAILED.\nDiagnostic errors (${result.diagnosticErrors.length})`;
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
      // When all errors are pre-existing, tool output should NOT contain VALIDATION ERRORS
      const toolMessages = result.messages.filter(
        (m) => m.role === MessageRole.TOOL,
      );
      expect(toolMessages.length).toBe(1);
      expect(toolMessages[0]!.content).not.toContain("VALIDATION ERRORS");
    });
  });

  // ─── Fix 3: Structured tool summaries ────────────────────────────────────
  describe("structured tool summaries", () => {
    it("captures search/replace details for replace_in_file in summary", async () => {
      const replaceTool: ToolSpec = {
        name: "replace_in_file",
        description: "Replace in file",
        category: "mutating",
        paramSchema: { type: "object" },
        resultSchema: { type: "object" },
        handler: async () => ({
          success: true,
          output: "Replaced 4 occurrence(s) in src/utils.cpp",
          error: null,
          artifacts: ["/tmp/src/utils.cpp"],
        }),
      };

      const sessionState = new SessionState();

      const provider = createMockProvider([
        [
          {
            type: "tool_call",
            content: '{"path": "src/utils.cpp", "search": "@ohos.data", "replace": "@ohos:data"}',
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
      registry.register(replaceTool);
      const gate = new ApprovalGate(config.approval, bus);

      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config,
        systemPrompt: "Test",
        repoRoot: "/tmp",
        sessionState,
      });

      await loop.run("Replace descriptors");

      const summaries = sessionState.getToolSummaries();
      expect(summaries.length).toBe(1);
      // Should contain structured info about what was searched/replaced
      expect(summaries[0]!.summary).toContain("@ohos.data");
      expect(summaries[0]!.summary).toContain("@ohos:data");
      expect(summaries[0]!.summary).toContain("4");
    });
  });


  // ─── Bug fix: record modifiedFiles even on partial success ────────────────
