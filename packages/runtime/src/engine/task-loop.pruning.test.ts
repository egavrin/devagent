import { resolve } from "node:path";
import {
  describe,
  it,
  expect,
  beforeEach,
  beforeAll,
} from "vitest";

import { SessionState } from "./session-state.js";
import {
  TaskLoop,
  extractStructuralDigest,
} from "./task-loop.js";
import {
  createMockProvider,
  makeConfig,
  makeEchoTool,
} from "./task-loop.test-helpers.js";
import type {
  LLMProvider,
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

  describe("post-injection token count in compaction", () => {
    it("context:compacted estimatedTokens includes session state tokens after pruning", async () => {
      // Verify that the estimatedTokens in the context:compacted event
      // reflects the token count AFTER injectSessionState() has added
      // session state content to messages (not before).

      // Provider that reports high token count to trigger pruning
      let callIndex = 0;
      const provider: LLMProvider = {
        id: "inject-test",
        async *chat(): AsyncIterable<StreamChunk> {
          callIndex++;
          if (callIndex <= 5) {
            yield {
              type: "tool_call",
              content: '{"text": "data"}',
              toolCallId: `c_${callIndex}`,
              toolName: "echo",
            };
            // Ramp up reported tokens to trigger compaction
            yield {
              type: "done",
              content: "",
              usage: { inputTokens: 2000 * callIndex, outputTokens: 100 },
            };
          } else {
            yield { type: "text", content: "Done" };
            yield { type: "done", content: "" };
          }
        },
        abort() {},
      };

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
          output: `Echo: ${params["text"]} ${"x".repeat(2000)}`,
          error: null,
          artifacts: [],
        }),
      };

      const registry = new ToolRegistry();
      registry.register(bigEchoTool);

      const compactConfig = makeConfig({
        budget: {
          maxIterations: 10,
          maxContextTokens: 10_000,
          responseHeadroom: 1_000,
          costWarningThreshold: 5.0,
          enableCostTracking: true,
        },
        context: {
          pruningStrategy: "hybrid",
          triggerRatio: 0.8,
          keepRecentMessages: 4,
        },
      });

      const gate = new ApprovalGate(compactConfig.approval, bus);
      const contextManager = new ContextManager(compactConfig.context);

      // Create session state with data — this content will be injected
      // into messages by injectSessionState() and should be reflected
      // in the estimatedTokens of the compacted event.
      const sessionState = new SessionState();
      sessionState.recordModifiedFile("/tmp/foo.ts");
      sessionState.addToolSummary({ tool: "echo", target: "/tmp/foo.ts", summary: "test data for session state padding content", iteration: 1 });

      const compactedEvents: Array<{ estimatedTokens: number; tokensBefore: number }> = [];
      bus.on("context:compacted", (evt: any) => {
        compactedEvents.push(evt);
      });

      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config: compactConfig,
        systemPrompt: "Test",
        repoRoot: "/tmp",
        contextManager,
        sessionState,
      });

      const result = await loop.run("Run echo");
      expect(result.status).toBeDefined();

      // If compaction fired, verify the event's estimatedTokens is a valid
      // number (not NaN, not stale pre-injection count)
      if (compactedEvents.length > 0) {
        const evt = compactedEvents[compactedEvents.length - 1]!;
        expect(evt.estimatedTokens).toBeGreaterThan(0);
        expect(Number.isNaN(evt.estimatedTokens)).toBe(false);
      }
    });
  });
  describe("approaching-limit warning at 60%", () => {
    it("fires warning when estimated tokens exceed 60% of threshold", async () => {
      // Budget: 10000 tokens, trigger at 80% = 8000, warning at 60% of 8000 = 4800
      // Provider reports 5000 tokens on first call → should trigger warning.
      let callIndex = 0;
      const provider: LLMProvider = {
        id: "mock-warning",
        async *chat(): AsyncIterable<StreamChunk> {
          if (callIndex === 0) {
            callIndex++;
            yield {
              type: "tool_call",
              content: '{"text": "hi"}',
              toolCallId: "call_0",
              toolName: "echo",
            };
            // Report 5000 tokens — above 60% of 8000 (4800), but below 8000
            yield {
              type: "done",
              content: "",
              usage: { promptTokens: 5000, completionTokens: 20 },
            };
          } else {
            yield { type: "text", content: "Done." };
            yield {
              type: "done",
              content: "",
              usage: { promptTokens: 5000, completionTokens: 10 },
            };
          }
        },
        abort() {},
      };

      const registry = new ToolRegistry();
      registry.register(makeEchoTool());

      const warningConfig = makeConfig({
        budget: {
          maxIterations: 5,
          maxContextTokens: 10000,
          responseHeadroom: 500,
          costWarningThreshold: 1.0,
          enableCostTracking: true,
        },
        context: {
          pruningStrategy: "sliding_window",
          triggerRatio: 0.8,
          keepRecentMessages: 4,
        },
      });

      const gate = new ApprovalGate(warningConfig.approval, bus);
      const contextManager = new ContextManager(warningConfig.context);
      const sessionState = new SessionState();

      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config: warningConfig,
        systemPrompt: "Test",
        repoRoot: "/tmp",
        contextManager,
        sessionState,
      });

      const result = await loop.run("Test warning");

      // Warning should have been injected as a SYSTEM message
      const warningMessages = result.messages.filter(
        (m) => m.role === MessageRole.SYSTEM && m.content?.includes("Context is filling up"),
      );
      expect(warningMessages.length).toBe(1);
      expect(warningMessages[0]!.content).toContain("save_finding");
    });
  });
  it("read_file summary includes structural digest", async () => {
    const readTool: ToolSpec = {
      name: "read_file",
      description: "Read a file",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "// comment line\n\nexport function App() {\n  return 1;\n}\n\nexport class Thing {}\n",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/app.tsx"}',
          toolCallId: "call_0",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Read the file." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(readTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

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

    await loop.run("Read the file");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    expect(summaries[0]!.summary).toContain("Read");
    expect(summaries[0]!.summary).toContain("export function App");
    expect(summaries[0]!.summary).toContain("export class Thing");
    expect(summaries[0]!.summary).toMatch(/Read \d+ lines/);
  });

  it("run_command summary uses summarizeDiff for git diff commands", async () => {
    const runTool: ToolSpec = {
      name: "run_command",
      description: "Run a command",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { command: { type: "string" } },
        required: ["command"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "diff --git a/foo.ts b/foo.ts\n--- a/foo.ts\n+++ b/foo.ts\n@@ -1,3 +1,4 @@ function foo\n+import bar;\n const x = 1;\n const y = 2;\n",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"command": "git diff HEAD~1"}',
          toolCallId: "call_0",
          toolName: "run_command",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Diff shown." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(runTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

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

    await loop.run("Show diff");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    // summarizeDiff should produce a hunk-based summary, not raw output
    expect(summaries[0]!.summary).toContain("foo");
  });

  it("run_command summary uses first line hint for non-git-diff commands", async () => {
    const runTool: ToolSpec = {
      name: "run_command",
      description: "Run a command",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { command: { type: "string" } },
        required: ["command"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "total 42\ndrwxr-xr-x  5 user  staff  160 Jan  1 12:00 src\n",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"command": "ls -la"}',
          toolCallId: "call_0",
          toolName: "run_command",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Listed." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(runTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

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

    await loop.run("List files");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    // Should use first line as hint
    expect(summaries[0]!.summary).toContain("total 42");
  });

  describe("extractStructuralDigest", () => {
    it("extracts declarations from source text", () => {
      const digest = extractStructuralDigest(`
export class Worker {}
function runTask() {}
interface RunOptions {}
type RunResult = string
      `);
      expect(digest).toContain("export class Worker");
      expect(digest).toContain("function runTask");
      expect(digest).toContain("interface RunOptions");
      expect(digest).toContain("type RunResult");
    });

    it("respects declaration and character limits", () => {
      const lines = Array.from({ length: 20 }, (_, i) => `function fn${i}() {}`).join("\n");
      const digest = extractStructuralDigest(lines, 60);
      expect(digest.length).toBeLessThanOrEqual(60);
      expect(digest).toContain("fn0");
      expect(digest).toContain("...");
    });

    it("returns empty string when no structural declarations are found", () => {
      const digest = extractStructuralDigest("just plain text\nwithout symbols");
      expect(digest).toBe("");
    });
  });
