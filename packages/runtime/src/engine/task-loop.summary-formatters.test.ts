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
  summarizeTestOutput,
} from "./task-loop.js";
import {
  createMockProvider,
  makeConfig,
  makeEchoTool,
} from "./task-loop.test-helpers.js";
import type {
  Message,
  ToolSpec,
} from "../core/index.js";
import {
  EventBus,
  ApprovalGate,
  MessageRole,
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

  it("search_files summary retains pattern, file paths, and match lines", async () => {
    const searchTool: ToolSpec = {
      name: "search_files",
      description: "Search files",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { pattern: { type: "string" }, path: { type: "string" } },
        required: ["pattern"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: [
          "Found 4 matches for \"createPlanTool(\" in 3 files:",
          "",
          "packages/runtime/src/engine/plan-tool.ts",
          "  42: export function createPlanTool(ctx: Context) {",
          "",
          "packages/runtime/src/engine/task-loop.ts",
          "  105: const tool = createPlanTool(this.context);",
          "  200: tools.push(createPlanTool(ctx));",
          "",
          "packages/runtime/src/engine/index.ts",
          '  12: export { createPlanTool } from "./plan-tool.js";',
        ].join("\n"),
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"pattern": "createPlanTool(", "path": "/tmp"}',
          toolCallId: "call_0",
          toolName: "search_files",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(searchTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Search");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    const s = summaries[0]!.summary;
    // Should retain the search pattern
    expect(s).toContain("createPlanTool(");
    // Should retain file paths
    expect(s).toContain("plan-tool.ts");
    expect(s).toContain("task-loop.ts");
    expect(s).toContain("index.ts");
    // Should retain match lines
    expect(s).toContain("export function createPlanTool");
  });

  it("find_files summary retains glob pattern and file paths", async () => {
    const findTool: ToolSpec = {
      name: "find_files",
      description: "Find files",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { pattern: { type: "string" } },
        required: ["pattern"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: [
          "Found 4 files matching \"**/*.test.ts\":",
          "packages/runtime/src/core/config.test.ts",
          "packages/runtime/src/core/session.test.ts",
          "packages/runtime/src/engine/task-loop.test.ts",
          "packages/runtime/src/engine/session-state.test.ts",
        ].join("\n"),
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"pattern": "**/*.test.ts"}',
          toolCallId: "call_0",
          toolName: "find_files",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(findTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Find test files");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    const s = summaries[0]!.summary;
    // Should retain glob pattern
    expect(s).toContain("**/*.test.ts");
    // Should retain file paths
    expect(s).toContain("config.test.ts");
    expect(s).toContain("session-state.test.ts");
  });

  it("git_status summary retains status groups and file names", async () => {
    const gitStatusTool: ToolSpec = {
      name: "git_status",
      description: "Show git status",
      category: "readonly",
      paramSchema: { type: "object", properties: {} },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: [
          " M packages/runtime/src/core/config.ts",
          " M packages/runtime/src/core/session.ts",
          " M packages/runtime/src/engine/task-loop.ts",
          "A  packages/runtime/src/engine/finding-tool.ts",
          "A  packages/runtime/src/engine/session-state.ts",
          "?? tmp/debug.log",
        ].join("\n"),
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
          toolName: "git_status",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(gitStatusTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Status");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    const s = summaries[0]!.summary;
    // Should retain entry count
    expect(s).toContain("6");
    // Should retain status groups with file names
    expect(s).toContain("[M]");
    expect(s).toContain("config.ts");
    expect(s).toContain("[A]");
    expect(s).toContain("finding-tool.ts");
    expect(s).toContain("[?]");
    expect(s).toContain("debug.log");
  });

  it("run_command summary preserves head+tail for long output", async () => {
    const outputLines = Array.from({ length: 30 }, (_, i) => `output line ${i}`);
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
        output: outputLines.join("\n"),
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"command": "some-long-command"}',
          toolCallId: "call_0",
          toolName: "run_command",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(runTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Run command");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    const s = summaries[0]!.summary;
    // Should include the command
    expect(s).toContain("some-long-command");
    // Should include first lines (head)
    expect(s).toContain("output line 0");
    // Should include last lines (tail)
    expect(s).toContain("output line 29");
  });

  it("run_command summary extracts pass/fail for test output", async () => {
    const testOutput = [
      "PASS src/config.test.ts",
      "PASS src/session.test.ts",
      "FAIL src/task-loop.test.ts",
      "  ● search_files summary",
      "    Expected: contains 'pattern'",
      "    Received: '15 matches'",
      "",
      "Tests: 2 passed, 1 failed, 3 total",
    ].join("\n");

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
        output: testOutput,
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"command": "bun run test"}',
          toolCallId: "call_0",
          toolName: "run_command",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(runTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Run tests");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    const s = summaries[0]!.summary;
    // Should contain pass/fail counts
    expect(s).toContain("2 passed");
    expect(s).toContain("1 failed");
    // Should contain the failing test name
    expect(s).toContain("task-loop.test.ts");
  });

  it("diagnostics summary has severity counts and diagnostic lines", async () => {
    const diagOutput = [
      "packages/runtime/src/engine/task-loop.ts:42:10 - error TS2322: Type 'string' is not assignable to type 'number'.",
      "packages/runtime/src/engine/task-loop.ts:105:5 - error TS2339: Property 'foo' does not exist on type 'Bar'.",
      "packages/runtime/src/engine/task-loop.ts:88:3 - warning TS6133: Variable 'x' is declared but never used.",
    ].join("\n");

    const diagTool: ToolSpec = {
      name: "diagnostics",
      description: "Show diagnostics",
      category: "readonly",
      paramSchema: { type: "object", properties: {} },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: diagOutput,
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
          toolName: "diagnostics",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(diagTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Check");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    const s = summaries[0]!.summary;
    // Should contain severity counts
    expect(s).toMatch(/2\s*error/);
    expect(s).toMatch(/1\s*warning/);
    // Should retain diagnostic lines
    expect(s).toContain("TS2322");
    expect(s).toContain("TS2339");
    expect(s).toContain("TS6133");
  });

  it("symbols summary retains symbol list", async () => {
    const symbolsOutput = [
      "class SessionState (line 98)",
      "  method setPlan (line 179)",
      "  method getPlan (line 188)",
      "  method addToolSummary (line 286)",
      "interface SessionStateConfig (line 51)",
      "function extractEnvFact (line 623)",
    ].join("\n");

    const symbolsTool: ToolSpec = {
      name: "symbols",
      description: "Show symbols",
      category: "readonly",
      paramSchema: { type: "object", properties: { path: { type: "string" } } },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: symbolsOutput,
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/session-state.ts"}',
          toolCallId: "call_0",
          toolName: "symbols",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(symbolsTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Symbols");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    const s = summaries[0]!.summary;
    // Should retain symbol count
    expect(s).toContain("6 symbols");
    // Should retain symbol entries
    expect(s).toContain("class SessionState");
    expect(s).toContain("method setPlan");
    expect(s).toContain("function extractEnvFact");
  });

  it("read_file summary has richer structural digest (20 declarations, 8 snippet lines)", async () => {
    // Generate source with 25 declarations
    const decls = Array.from({ length: 25 }, (_, i) => `export function fn${i}() {\n  return ${i};\n}`);
    const source = decls.join("\n\n");

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
        output: source,
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"path": "/tmp/big.ts"}',
          toolCallId: "call_0",
          toolName: "read_file",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(readTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Read");
    const summaries = sessionState.getToolSummaries();
    expect(summaries.length).toBe(1);
    const s = summaries[0]!.summary;
    // With raised limit, should include more than old cap of 10 declarations
    // It should contain fn0 through at least fn14 (up to 20)
    expect(s).toContain("fn0");
    expect(s).toContain("fn14");
  });

  it("search_files and find_files get distinct summary targets", async () => {
    const searchTool: ToolSpec = {
      name: "search_files",
      description: "Search files",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { pattern: { type: "string" }, path: { type: "string" } },
        required: ["pattern"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "Found 2 matches for \"foo\" in 1 file:\nsrc/bar.ts\n  10: const foo = 1;",
        error: null,
        artifacts: [],
      }),
    };

    const findTool: ToolSpec = {
      name: "find_files",
      description: "Find files",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { pattern: { type: "string" } },
        required: ["pattern"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "Found 3 files matching \"*.ts\":\nsrc/a.ts\nsrc/b.ts\nsrc/c.ts",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"pattern": "foo"}',
          toolCallId: "call_0",
          toolName: "search_files",
        },
        { type: "done", content: "" },
      ],
      [
        {
          type: "tool_call",
          content: '{"pattern": "*.ts"}',
          toolCallId: "call_1",
          toolName: "find_files",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(searchTool);
    registry.register(findTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Search and find");
    const summaries = sessionState.getToolSummaries();
    // Both should be stored (distinct targets), not overwrite each other
    expect(summaries.length).toBe(2);
    // Verify targets are different
    expect(summaries[0]!.target).not.toBe(summaries[1]!.target);
    // Targets should include tool-specific prefixes
    expect(summaries.some(s => s.target.startsWith("search:"))).toBe(true);
    expect(summaries.some(s => s.target.startsWith("find:"))).toBe(true);
  });

  it("search_files with different patterns on the same path produces distinct summary targets", async () => {
    // Regression test: different patterns on the same path must yield separate SessionState
    // entries so the stagnation detector sees progress on each call.
    //
    // Previously getSummaryTarget returned the path arg directly for any tool
    // with a non-empty path, so search_files(pattern=A, path=P) and
    // search_files(pattern=B, path=P) both produced target=P. The second
    // overwrote the first → toolSummaries.length stayed constant across iterations
    // → the stagnation detector fired prematurely, cutting off directory discovery
    // before all ANI source directories had been scanned.
    const searchTool: ToolSpec = {
      name: "search_files",
      description: "Search files",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: { pattern: { type: "string" }, path: { type: "string" } },
        required: ["pattern"],
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "1 match(es):\nframeworks/js/ani/foo.cpp:10: match",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      [
        // First call: pattern A on path P
        {
          type: "tool_call",
          content: '{"pattern": "Find(Class|Namespace)", "path": "frameworks/js/ani"}',
          toolCallId: "call_0",
          toolName: "search_files",
        },
        { type: "done", content: "" },
      ],
      [
        // Second call: DIFFERENT pattern, SAME path — must NOT overwrite first
        {
          type: "tool_call",
          content: '{"pattern": "@ohos\\\\.[A-Za-z]+", "path": "frameworks/js/ani"}',
          toolCallId: "call_1",
          toolName: "search_files",
        },
        { type: "done", content: "" },
      ],
      [{ type: "text", content: "Done." }, { type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(searchTool);
    const gate = new ApprovalGate(config.approval, bus);
    const sessionState = new SessionState();

    const loop = new TaskLoop({
      provider, tools: registry, bus, approvalGate: gate, config,
      systemPrompt: "Test", repoRoot: "/tmp", sessionState,
    });

    await loop.run("Search with two different patterns in the same directory");
    const summaries = sessionState.getToolSummaries();
    // Both calls must produce distinct entries — second must NOT overwrite first
    expect(summaries.length).toBe(2);
    expect(summaries[0]!.target).not.toBe(summaries[1]!.target);
    // Coverage targets must also be distinct (stagnation detector tracks these)
    const coverage = sessionState.getReadonlyCoverage().get("search_files") ?? [];
    expect(coverage.length).toBe(2);
  });

  describe("summarizeTestOutput", () => {
    it("extracts pass/fail/skip counts from test output", () => {
      const output = "Tests: 10 passed, 2 failed, 1 skipped, 13 total";
      const result = summarizeTestOutput(output);
      expect(result).not.toBeNull();
      expect(result).toContain("10 passed");
      expect(result).toContain("2 failed");
    });

    it("extracts TypeScript error counts from tsc output", () => {
      const output = [
        "src/foo.ts(10,5): error TS2322: Type 'string' is not assignable to type 'number'.",
        "src/bar.ts(20,3): error TS2339: Property 'x' does not exist on type 'Y'.",
        "",
        "Found 2 errors in 2 files.",
      ].join("\n");
      const result = summarizeTestOutput(output);
      expect(result).not.toBeNull();
      expect(result).toContain("2 error");
      expect(result).toContain("TS2322");
    });

    it("extracts failing test names", () => {
      const output = [
        "PASS src/config.test.ts",
        "FAIL src/task-loop.test.ts",
        "  ● formatToolSummary > search summary",
        "    Expected: contains 'pattern'",
        "Tests: 1 passed, 1 failed, 2 total",
      ].join("\n");
      const result = summarizeTestOutput(output);
      expect(result).not.toBeNull();
      expect(result).toContain("FAIL");
      expect(result).toContain("task-loop.test.ts");
    });

    it("returns null for non-test output", () => {
      const output = "total 42\ndrwxr-xr-x  5 user  staff  160 Jan  1 12:00 src";
      const result = summarizeTestOutput(output);
      expect(result).toBeNull();
    });
  });
  it("git_diff with staged does NOT supersede unstaged git_diff", async () => {
    const gitDiffTool: ToolSpec = {
      name: "git_diff",
      description: "Show git diff",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: {
          path: { type: "string" },
          staged: { type: "boolean" },
          ref: { type: "string" },
        },
      },
      resultSchema: { type: "object" },
      handler: async (params) => ({
        success: true,
        output: params["staged"] ? "staged diff content" : "unstaged diff content",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      // First call: unstaged git_diff
      [
        {
          type: "tool_call",
          content: '{}',
          toolCallId: "call_0",
          toolName: "git_diff",
        },
        { type: "done", content: "" },
      ],
      // Second call: staged git_diff
      [
        {
          type: "tool_call",
          content: '{"staged": true}',
          toolCallId: "call_1",
          toolName: "git_diff",
        },
        { type: "done", content: "" },
      ],
      // Final response
      [
        { type: "text", content: "Both diffs shown." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(gitDiffTool);
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

    const result = await loop.run("Show both diffs");

    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    // Both should be preserved (not superseded)
    expect(toolMessages.length).toBe(2);
    // Unstaged should NOT be superseded
    expect(toolMessages[0]!.content).toContain("unstaged diff content");
    // Staged should also be preserved
    expect(toolMessages[1]!.content).toContain("staged diff content");
  });

  it("same git_diff with same staged flag skips repeated execution", async () => {
    const gitDiffTool: ToolSpec = {
      name: "git_diff",
      description: "Show git diff",
      category: "readonly",
      paramSchema: {
        type: "object",
        properties: {
          staged: { type: "boolean" },
        },
      },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "diff output",
        error: null,
        artifacts: [],
      }),
    };

    const provider = createMockProvider([
      // First call: staged git_diff
      [
        {
          type: "tool_call",
          content: '{"staged": true}',
          toolCallId: "call_0",
          toolName: "git_diff",
        },
        { type: "done", content: "" },
      ],
      // Second call: same staged git_diff
      [
        {
          type: "tool_call",
          content: '{"staged": true}',
          toolCallId: "call_1",
          toolName: "git_diff",
        },
        { type: "done", content: "" },
      ],
      // Final response
      [
        { type: "text", content: "Done." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(gitDiffTool);
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

    const result = await loop.run("Show diff twice");

    const toolMessages = result.messages.filter(
      (m) => m.role === MessageRole.TOOL,
    );
    expect(toolMessages.length).toBe(2);
    // First should be superseded
    expect(toolMessages[0]!.content).toContain("[Superseded");
    // Second should be a direct dedup skip
    expect(toolMessages[1]!.content).toContain("Skipped redundant git_diff");
  });


  // ─── Pinned Message Support ─────────────────────────────────
  it("git_diff results are auto-pinned", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call" as const,
          content: '{}',
          toolCallId: "call_diff",
          toolName: "git_diff",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Done" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register({
      name: "git_diff",
      description: "Show git diff",
      category: "readonly",
      paramSchema: { type: "object" },
      resultSchema: { type: "object" },
      handler: async () => ({
        success: true,
        output: "diff --git a/file.ts\n+new line",
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
    });

    const result = await loop.run("Show the diff");
    const toolMsgs = result.messages.filter((m) => m.role === MessageRole.TOOL);
    expect(toolMsgs.length).toBe(1);
    expect(toolMsgs[0]!.pinned).toBe(true);
  });

  it("auto-pinning stops after MAX_PINNED_DIFFS", async () => {
    const provider = createMockProvider([]);
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

    // Simulate 25 git_diff results via appendToolResult
    for (let i = 0; i < 25; i++) {
      (loop as any).appendToolResult(`call_${i}`, {
        success: true,
        output: `diff ${i}`,
        error: null,
        artifacts: [],
      }, "git_diff", {});
    }

    const toolMsgs = (loop as any).messages.filter((m: Message) => m.role === MessageRole.TOOL);
    const pinnedCount = toolMsgs.filter((m: Message) => m.pinned === true).length;
    expect(pinnedCount).toBe(20); // MAX_PINNED_DIFFS
    // Messages 21-25 should not be pinned
    expect(toolMsgs[20]!.pinned).toBeUndefined();
  });

  it("pinned messages survive pruneToolOutputs", async () => {
    const provider = createMockProvider([]);
    const registry = new ToolRegistry();
    registry.register(makeEchoTool());
    const testConfig = makeConfig({
      context: { ...config.context, pruneProtectTokens: 0 },
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

    const bigOutput = "x".repeat(45_000);
    const messages: Message[] = [
      { role: MessageRole.SYSTEM, content: "Test" },
      {
        role: MessageRole.ASSISTANT,
        content: null,
        toolCalls: [{ name: "git_diff", arguments: {}, callId: "call_pinned" }],
      },
      {
        role: MessageRole.TOOL,
        content: bigOutput,
        toolCallId: "call_pinned",
        pinned: true,
      },
      {
        role: MessageRole.ASSISTANT,
        content: null,
        toolCalls: [{ name: "read_file", arguments: { path: "/tmp/f.ts" }, callId: "call_unpinned" }],
      },
      {
        role: MessageRole.TOOL,
        content: bigOutput,
        toolCallId: "call_unpinned",
      },
    ];

    (loop as any).messages = messages;
    const pruneResult = (loop as any).pruneToolOutputs(200_000, 50_000);

    // Unpinned should be pruned, pinned should survive
    const toolMsgs = (loop as any).messages.filter((m: Message) => m.role === MessageRole.TOOL);
    const pinnedMsg = toolMsgs.find((m: Message) => m.pinned === true);
    expect(pinnedMsg).toBeDefined();
    expect(pinnedMsg!.content).toBe(bigOutput); // unchanged
    expect(pruneResult.prunedCount).toBe(1); // only the unpinned one
  });


  // ─── Prune Priority Ordering ─────────────────────────────────
  describe("prune priority ordering", () => {
    it("git_status pruned before git_diff at same age", async () => {
      const provider = createMockProvider([]);
      const registry = new ToolRegistry();
      registry.register(makeEchoTool());
      const testConfig = makeConfig({
        context: { ...config.context, pruneProtectTokens: 0 },
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

      // Use large outputs (~6250 tokens each) so 1 prune is enough
      const bigOutput = "x".repeat(25_000);
      const messages: Message[] = [
        { role: MessageRole.SYSTEM, content: "Test" },
        // git_diff result (priority 3)
        {
          role: MessageRole.ASSISTANT,
          content: null,
          toolCalls: [{ name: "git_diff", arguments: {}, callId: "call_diff" }],
        },
        { role: MessageRole.TOOL, content: bigOutput, toolCallId: "call_diff" },
        // git_status result (priority 0)
        {
          role: MessageRole.ASSISTANT,
          content: null,
          toolCalls: [{ name: "git_status", arguments: {}, callId: "call_status" }],
        },
        { role: MessageRole.TOOL, content: bigOutput, toolCallId: "call_status" },
      ];

      (loop as any).messages = messages;
      // currentTokens=~12500, threshold=10000, targetTokens=8500, targetSavings=4000
      const pruneResult = (loop as any).pruneToolOutputs(12_500, 10_000);
      expect(pruneResult.prunedCount).toBe(1);

      // git_status should be pruned (priority 0), git_diff should survive (priority 3)
      const toolMsgs = (loop as any).messages.filter((m: Message) => m.role === MessageRole.TOOL);
      const statusMsg = toolMsgs.find((m: Message) => m.toolCallId === "call_status");
      const diffMsg = toolMsgs.find((m: Message) => m.toolCallId === "call_diff");
      expect(statusMsg!.content).toContain("[Previously:");
      expect(diffMsg!.content).toBe(bigOutput);
    });

    it("find_files pruned before read_file at same age", async () => {
      const provider = createMockProvider([]);
      const registry = new ToolRegistry();
      registry.register(makeEchoTool());
      const testConfig = makeConfig({
        context: { ...config.context, pruneProtectTokens: 0 },
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

      // Use large outputs (~6250 tokens each) so 1 prune is enough to hit savings
      const bigOutput = "x".repeat(25_000);
      const messages: Message[] = [
        { role: MessageRole.SYSTEM, content: "Test" },
        // read_file result (priority 2)
        {
          role: MessageRole.ASSISTANT,
          content: null,
          toolCalls: [{ name: "read_file", arguments: { path: "/tmp/a.ts" }, callId: "call_read" }],
        },
        { role: MessageRole.TOOL, content: bigOutput, toolCallId: "call_read" },
        // find_files result (priority 0)
        {
          role: MessageRole.ASSISTANT,
          content: null,
          toolCalls: [{ name: "find_files", arguments: {}, callId: "call_find" }],
        },
        { role: MessageRole.TOOL, content: bigOutput, toolCallId: "call_find" },
      ];

      (loop as any).messages = messages;
      // currentTokens=~12500, threshold=10000, targetTokens=8500, targetSavings=4000
      const pruneResult = (loop as any).pruneToolOutputs(12_500, 10_000);
      expect(pruneResult.prunedCount).toBe(1);

      const toolMsgs = (loop as any).messages.filter((m: Message) => m.role === MessageRole.TOOL);
      const findMsg = toolMsgs.find((m: Message) => m.toolCallId === "call_find");
      const readMsg = toolMsgs.find((m: Message) => m.toolCallId === "call_read");
      expect(findMsg!.content).toContain("[Previously:");
      expect(readMsg!.content).toBe(bigOutput);
    });
  });

  // ─── Post-Compaction Re-read Storm Detection ─────────────
