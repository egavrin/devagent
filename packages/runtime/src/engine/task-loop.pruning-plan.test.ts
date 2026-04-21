import { resolve } from "node:path";
import {
  describe,
  it,
  expect,
  beforeEach,
  beforeAll,
} from "vitest";

import {
  TaskLoop,
  summarizeDiff,
  truncateToolOutput,
} from "./task-loop.js";
import {
  createMockProvider,
  makeConfig,
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
  lookupModelPricing,
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

  describe("summarizeDiff", () => {
    it("extracts function names from hunk headers", () => {
      const diff = `diff --git a/src/auth.ts b/src/auth.ts
--- a/src/auth.ts
+++ b/src/auth.ts
@@ -10,6 +10,9 @@ function handleAuth(req: Request)
+    const token = req.headers.get("Authorization");
+    if (!token) {
+      throw new Error("Missing token");
@@ -50,3 +53,7 @@ export function validateToken(token: string)
+    const decoded = jwt.verify(token, secret);
+    return decoded;
+    // TODO: add expiry check
+    // TODO: add refresh logic`;

      const result = summarizeDiff(diff);
      expect(result).toContain("handleAuth");
      expect(result).toContain("validateToken");
      expect(result).toContain("+7/-0");
    });

    it("handles diffstat format", () => {
      const diff = " 3 files changed, 10 insertions(+), 5 deletions(-)";
      const result = summarizeDiff(diff);
      expect(result).toBe("3 files changed, +10, -5");
    });

    it("handles diff with no hunk headers", () => {
      const diff = "some output\nwithout any hunks\njust text lines";
      const result = summarizeDiff(diff);
      expect(result).toContain("diff: 3 lines");
    });

    it("counts additions and deletions correctly", () => {
      const diff = `@@ -1,3 +1,4 @@ class MyClass
-    old line
+    new line 1
+    new line 2`;
      const result = summarizeDiff(diff);
      expect(result).toContain("+2/-1");
    });

    it("limits to 4 most significant hunks", () => {
      const hunks = Array.from({ length: 6 }, (_, i) =>
        `@@ -${i * 10},5 +${i * 10},5 @@ function fn${i}()\n+line ${i}`,
      ).join("\n");
      const result = summarizeDiff(hunks);
      expect(result).toContain("+6/-0");
      expect(result).toContain("+2 more");
    });
  });
  describe("model pricing via registry", () => {
    it("returns pricing for known models", () => {
      const pricing = lookupModelPricing("claude-sonnet-4-20250514");
      expect(pricing).toBeDefined();
      expect(pricing!.inputPricePerMillion).toBe(3);
      expect(pricing!.outputPricePerMillion).toBe(15);
    });

    it("returns undefined for unknown models", () => {
      expect(lookupModelPricing("unknown-model-xyz")).toBeUndefined();
    });
  });

  // ─── Tool Output Truncation ─────────────────────────────────
  describe("truncateToolOutput", () => {
    it("passes through short output unchanged", () => {
      const short = "hello world\nline 2";
      expect(truncateToolOutput(short)).toBe(short);
    });

    it("truncates long output with head+tail strategy", () => {
      // Create output with 500 lines, each 100 chars → ~50K chars
      const lines = Array.from({ length: 500 }, (_, i) =>
        `line ${i}: ${"x".repeat(90)}`,
      );
      const longOutput = lines.join("\n");
      expect(longOutput.length).toBeGreaterThan(48_000);

      const result = truncateToolOutput(longOutput);
      expect(result.length).toBeLessThan(longOutput.length);
      expect(result).toContain("[... ");
      expect(result).toContain("lines truncated ...");

      // Should contain first 200 lines
      expect(result).toContain("line 0:");
      expect(result).toContain("line 199:");

      // Should contain last 100 lines
      expect(result).toContain("line 400:");
      expect(result).toContain("line 499:");

      // Should NOT contain middle lines
      expect(result).not.toContain("line 250:");
    });

    it("respects custom maxChars parameter", () => {
      const lines = Array.from({ length: 500 }, (_, i) =>
        `line ${i}: data`,
      );
      const output = lines.join("\n");
      // Use a small limit to force truncation
      const result = truncateToolOutput(output, 100);
      expect(result.length).toBeLessThan(output.length);
    });

    it("handles few-lines-but-long-chars case", () => {
      // 10 very long lines
      const lines = Array.from({ length: 10 }, () => "x".repeat(10_000));
      const output = lines.join("\n");
      const result = truncateToolOutput(output, 1000);
      // Should be truncated by char limit since lines count < head+tail
      expect(result).toContain("[... output truncated ...]");
    });

    it("enforces maxChars when head+tail lines are individually long", () => {
      // 500 lines × 500 chars each → head+tail would be ~150K chars without char guard
      const lines = Array.from({ length: 500 }, (_, i) =>
        `line ${i}: ${"A".repeat(490)}`,
      );
      const output = lines.join("\n");
      const result = truncateToolOutput(output); // default maxChars = 48_000
      expect(result.length).toBeLessThanOrEqual(48_000 + 30); // allow for truncation marker
      expect(result).toContain("[... output truncated ...]");
    });

    it("does not double-truncate when head+tail fits within maxChars", () => {
      // 500 lines × ~100 chars each → total ~50K (exceeds 48K → enters truncation)
      // But head(200)+tail(100) = 300 lines × ~100 = 30K (fits within 48K → no char truncation)
      const lines = Array.from({ length: 500 }, (_, i) =>
        `line ${i}: ${"B".repeat(90)}`,
      );
      const output = lines.join("\n");
      expect(output.length).toBeGreaterThan(48_000); // must exceed to enter truncation
      const result = truncateToolOutput(output);
      // Should have the line-truncation marker but NOT the char-truncation marker
      expect(result).toContain("lines truncated ...");
      expect(result).not.toContain("[... output truncated ...]");
      // Should contain head and tail lines
      expect(result).toContain("line 0:");
      expect(result).toContain("line 499:");
    });
  });

  // ─── File-Read Deduplication ─────────────────────────────────
  describe("midpoint briefing state reset", () => {
    it("clears toolResultIndices after midpoint so stale indices do not corrupt new messages", async () => {
      const readTool: ToolSpec = {
        name: "read_file",
        description: "Read file",
        category: "readonly",
        paramSchema: {
          type: "object",
          properties: { path: { type: "string" } },
          required: ["path"],
        },
        resultSchema: { type: "object" },
        handler: async (params) => ({
          success: true,
          output: `Contents of ${params["path"]}`,
          error: null,
          artifacts: [],
        }),
      };

      // Iteration 1: read_file → tracked at some index
      // Iteration 2: midpoint fires (interval=2), replaces messages
      // Iteration 3: read_file same path → should NOT corrupt a message at the old stale index
      let callIndex = 0;
      const provider: LLMProvider = {
        id: "midpoint-test",
        async *chat(): AsyncIterable<StreamChunk> {
          callIndex++;
          if (callIndex === 1) {
            yield {
              type: "tool_call",
              content: '{"path": "/tmp/test.ts"}',
              toolCallId: "r1",
              toolName: "read_file",
            };
            yield { type: "done", content: "" };
          } else if (callIndex === 2) {
            // Dummy tool call to bump iteration to 2 (triggers midpoint)
            yield {
              type: "tool_call",
              content: '{"path": "/tmp/other.ts"}',
              toolCallId: "r2",
              toolName: "read_file",
            };
            yield { type: "done", content: "" };
          } else if (callIndex === 3) {
            // After midpoint: read same file again
            yield {
              type: "tool_call",
              content: '{"path": "/tmp/test.ts"}',
              toolCallId: "r3",
              toolName: "read_file",
            };
            yield { type: "done", content: "" };
          } else {
            yield { type: "text", content: "All done" };
            yield { type: "done", content: "" };
          }
        },
        abort() {},
      };

      const registry = new ToolRegistry();
      registry.register(readTool);

      const midpointConfig = makeConfig({
        budget: { ...config.budget, maxIterations: 10 },
        context: { ...config.context, midpointBriefingInterval: 2 },
      });
      const gate = new ApprovalGate(midpointConfig.approval, bus);

      const midpointCallback = async (
        _messages: ReadonlyArray<Message>,
      ): Promise<{ continueMessages: ReadonlyArray<Message> } | null> => {
        // Return enough messages so a stale index falls within bounds.
        // The stale toolResultIndices entry for "read_file:/tmp/test.ts"
        // will point at some index from the old array. We pad with
        // enough messages (including TOOL-role ones) so a stale write
        // would corrupt one of these instead of being out-of-bounds.
        const padding: Message[] = Array.from({ length: 10 }, (_, i) => ({
          role: MessageRole.TOOL,
          content: `Preserved tool result ${i}`,
          toolCallId: `preserved_${i}`,
        }));
        return {
          continueMessages: [
            { role: MessageRole.SYSTEM, content: "Summarized context" },
            ...padding,
          ],
        };
      };

      const loop = new TaskLoop({
        provider,
        tools: registry,
        bus,
        approvalGate: gate,
        config: midpointConfig,
        systemPrompt: "Test",
        repoRoot: "/tmp",
        midpointCallback,
      });

      const result = await loop.run("Read files");
      expect(result.status).toBe("success");

      // After midpoint, all tool messages should have valid content
      // (no message should have been corrupted by stale index writes)
      const toolMessages = result.messages.filter(
        (m) => m.role === MessageRole.TOOL,
      );
      for (const tm of toolMessages) {
        // Stale index corruption would write "[Superseded..." into a wrong message
        expect(tm.content).not.toContain("[Superseded");
      }
    });
  });
  it("replaces older tool result for same read_file path", async () => {
    // LLM reads the same file twice, then gives a final answer
    const provider = createMockProvider([
      // First call: read_file
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts"}',
          toolCallId: "call_0",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      // Second call: read_file same path
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts"}',
          toolCallId: "call_1",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      // Final answer
      [
        { type: "text", content: "Done reviewing" },
        { type: "done", content: "" },
      ],
    ]);

    const readFileTool: ToolSpec = {
      name: "read_file",
      description: "Read a file",
      category: "readonly",
      paramSchema: { type: "object", properties: { path: { type: "string" } } },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "file content here: " + "x".repeat(100),
        error: null,
        artifacts: [],
      }),
    };

    const registry = new ToolRegistry();
    registry.register(readFileTool);

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

    const result = await loop.run("Read the file");
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(2);
    // First tool result should be replaced with superseded note
    expect(toolMessages[0]!.content).toContain("[Superseded");
    // Second tool result should use generic readonly dedup skip text
    expect(toolMessages[1]!.content).toContain("Skipped redundant readonly call read_file");
  });

  it("does not supersede read_file results for different line ranges of same file", async () => {
    // LLM reads lines 1-50, then lines 51-100 of same file, then answers
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts", "start_line": 1, "end_line": 50}',
          toolCallId: "call_0",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts", "start_line": 51, "end_line": 100}',
          toolCallId: "call_1",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done reviewing" },
        { type: "done", content: "" },
      ],
    ]);

    const readFileTool: ToolSpec = {
      name: "read_file",
      description: "Read a file",
      category: "readonly",
      paramSchema: { type: "object", properties: { path: { type: "string" }, start_line: { type: "number" }, end_line: { type: "number" } } },
      resultSchema: { type: "object" },
      handler: async (params) => ({
        success: true,
        output: `Lines ${params["start_line"]}-${params["end_line"]}`,
        error: null,
        artifacts: [],
      }),
    };

    const registry = new ToolRegistry();
    registry.register(readFileTool);

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

    const result = await loop.run("Read different ranges");
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(2);
    // Neither should be superseded — different line ranges
    expect(toolMessages[0]!.content).not.toContain("[Superseded");
    expect(toolMessages[1]!.content).not.toContain("[Superseded");
    // Both should have their actual content
    expect(toolMessages[0]!.content).toContain("Lines 1-50");
    expect(toolMessages[1]!.content).toContain("Lines 51-100");
  });

  it("still supersedes read_file results for same path without line ranges", async () => {
    // LLM reads same file twice with no line ranges — should supersede
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts"}',
          toolCallId: "call_0",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/test.ts"}',
          toolCallId: "call_1",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done" },
        { type: "done", content: "" },
      ],
    ]);

    const readFileTool: ToolSpec = {
      name: "read_file",
      description: "Read a file",
      category: "readonly",
      paramSchema: { type: "object", properties: { path: { type: "string" } } },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "file content",
        error: null,
        artifacts: [],
      }),
    };

    const registry = new ToolRegistry();
    registry.register(readFileTool);

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

    const result = await loop.run("Read same file");
    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(2);
    // First should be superseded
    expect(toolMessages[0]!.content).toContain("[Superseded");
  });

  it("invalidates readonly dedup cache after failed mutating tool with artifacts", async () => {
    let readCalls = 0;
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path":"/tmp/test.ts"}',
          toolCallId: "call_read_1",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      [
        {
          type: "tool_call",
          content: '{"path":"/tmp/test.ts","content":"new value"}',
          toolCallId: "call_write_1",
          toolName: "write_file",
        },
        { type: "done", content: "" },
      ],
      [
        {
          type: "tool_call",
          content: '{"path":"/tmp/test.ts"}',
          toolCallId: "call_read_2",
          toolName: "read_file",
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
      name: "read_file",
      description: "Read a file",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { path: { type: "string" } },
        required: ["path"],
      },
      resultSchema: { type: "object" },
      handler: async () => {
        readCalls++;
        return {
          success: true,
          output: `content-${readCalls}`,
          error: null,
          artifacts: [],
        };
      },
    });
    registry.register({
      name: "write_file",
      description: "Write file",
      category: "mutating",
      paramSchema: {
        type: "object",
        properties: {
          path: { type: "string" },
          content: { type: "string" },
        },
        required: ["path", "content"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: false,
        output: "partial write occurred before failure",
        error: "disk quota reached",
        artifacts: ["/tmp/test.ts"],
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
    });

    const result = await loop.run("Read then mutate then read");
    expect(readCalls).toBe(2);
    const toolMessages = result.messages.filter((m) => m.role === MessageRole.TOOL);
    const secondReadMessage = toolMessages.find((m) => m.toolCallId === "call_read_2");
    expect(secondReadMessage?.content).toContain("content-2");
  });
