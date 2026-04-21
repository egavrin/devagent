/**
 * Live integration tests for ToolScriptEngine + createToolScriptTool.
 *
 * These tests use the REAL builtin tool handlers against the actual repo
 * to verify end-to-end behavior. They run find_files, read_file, search_files,
 * git_status, git_diff with real filesystem and git operations.
 */

import { resolve } from "node:path";
import { describe, it, expect, beforeEach } from "vitest";

import { createToolScriptTool } from "./tool-script-tool.js";
import { ToolScriptEngine } from "./tool-script.js";
import { EventBus } from "../core/index.js";
import type { ToolContext } from "../core/index.js";
import { createDefaultToolRegistry } from "../tools/index.js";
import type { ToolRegistry } from "../tools/index.js";

// ─── Setup ──────────────────────────────────────────────────

// Point at the actual repo root
const REPO_ROOT = resolve(import.meta.dirname, "../../../..");

let registry: ToolRegistry;
let bus: EventBus;
let context: ToolContext;

beforeEach(() => {
  registry = createDefaultToolRegistry();
  bus = new EventBus();
  context = {
    repoRoot: REPO_ROOT,
    config: {} as never,
    sessionId: "live-test",
  };
});

// ─── Live Engine Tests ──────────────────────────────────────
it("find_files returns real files from the repo", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    steps: [
      {
        id: "find",
        tool: "find_files",
        args: { pattern: "packages/runtime/src/engine/tool-script*.ts" },
      },
    ],
  });

  expect(result.steps).toHaveLength(1);
  expect(result.steps[0]!.success).toBe(true);
  expect(result.steps[0]!.output).toContain("tool-script.ts");
  expect(result.steps[0]!.output).toContain("tool-script-tool.ts");
});

it("find_files → read_file chain using $stepId.lines[N]", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    steps: [
      {
        id: "find",
        tool: "find_files",
        args: { pattern: "packages/runtime/src/engine/index.ts" },
      },
      {
        id: "read",
        tool: "read_file",
        args: { path: "$find.lines[0]" },
      },
    ],
  });

  expect(result.steps).toHaveLength(2);
  expect(result.steps[0]!.success).toBe(true);
  // The found file should contain "index.ts"
  expect(result.steps[0]!.output).toContain("index.ts");

  expect(result.steps[1]!.success).toBe(true);
  // The file content should have our new exports
  expect(result.steps[1]!.output).toContain("ToolScriptEngine");
  expect(result.steps[1]!.output).toContain("createToolScriptTool");
});

it("search_files finds pattern in real source files", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    steps: [
      {
        id: "search",
        tool: "search_files",
        args: {
          pattern: "class ToolScriptEngine",
          path: "packages/runtime/src/engine",
          file_pattern: "**/*.ts",
        },
      },
    ],
  });

  expect(result.steps).toHaveLength(1);
  expect(result.steps[0]!.success).toBe(true);
  expect(result.steps[0]!.output).toContain("tool-script.ts");
  expect(result.steps[0]!.output).toContain("class ToolScriptEngine");
});

it("git_status returns clean repo info", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    steps: [{ id: "status", tool: "git_status", args: {} }],
  });

  expect(result.steps).toHaveLength(1);
  expect(result.steps[0]!.success).toBe(true);
  // Should return some git status text (branch, changes, etc.)
  expect(result.steps[0]!.output.length).toBeGreaterThan(0);
});

it("multi-tool batch: find → search → read in one script", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    steps: [
      {
        id: "find",
        tool: "find_files",
        args: { pattern: "packages/runtime/src/core/types.ts" },
      },
      {
        id: "search",
        tool: "search_files",
        args: {
          pattern: "ToolCategory",
          path: "packages/runtime/src/core",
          file_pattern: "**/types.ts",
          max_results: 5,
        },
      },
      {
        id: "read",
        tool: "read_file",
        args: { path: "$find.lines[0]", start_line: 1, end_line: 30 },
      },
    ],
  });

  expect(result.steps).toHaveLength(3);

  // find_files: should find the file
  expect(result.steps[0]!.success).toBe(true);
  expect(result.steps[0]!.output).toContain("types.ts");

  // search_files: should find ToolCategory references
  expect(result.steps[1]!.success).toBe(true);
  expect(result.steps[1]!.output).toContain("ToolCategory");

  // read_file: should read beginning of types.ts (using resolved ref)
  expect(result.steps[2]!.success).toBe(true);
  expect(result.steps[2]!.output.length).toBeGreaterThan(0);
});

it("emits real tool:before and tool:after events during live execution", async () => {
  const beforeNames: string[] = [];
  const afterNames: string[] = [];

  bus.on("tool:before", (e) => beforeNames.push(e.name));
  bus.on("tool:after", (e) => afterNames.push(e.name));

  const engine = new ToolScriptEngine({ registry, context, bus });

  await engine.execute({
    steps: [
      {
        id: "find",
        tool: "find_files",
        args: { pattern: "package.json", max_results: 5 },
      },
      { id: "status", tool: "git_status", args: {} },
    ],
  });

  expect(beforeNames).toEqual(["find_files", "git_status"]);
  expect(afterNames).toEqual(["find_files", "git_status"]);
});

it("handles read_file on nonexistent file gracefully (fail-forward)", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    steps: [
      {
        id: "read_missing",
        tool: "read_file",
        args: { path: "this/file/does/not/exist.ts" },
      },
      { id: "status", tool: "git_status", args: {} },
    ],
  });

  expect(result.steps).toHaveLength(2);
  // First step fails but doesn't abort the script
  expect(result.steps[0]!.success).toBe(false);
  // Second step still runs
  expect(result.steps[1]!.success).toBe(true);
});

it("reference from failed step produces error marker", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    steps: [
      {
        id: "bad",
        tool: "read_file",
        args: { path: "nonexistent.ts" },
      },
      {
        id: "use_bad",
        tool: "read_file",
        args: { path: "$bad" },
      },
    ],
  });

  expect(result.steps).toHaveLength(2);
  expect(result.steps[0]!.success).toBe(false);
  // The second step should have received an error marker as the path
  expect(result.steps[1]!.success).toBe(false);
});


// ─── Live Factory Tests ─────────────────────────────────────

describe("createToolScriptTool (live)", () => {
  it("full end-to-end via the tool handler with JSON string input", async () => {
    const tool = createToolScriptTool({ registry, bus });

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          {
            id: "find",
            tool: "find_files",
            args: { pattern: "packages/runtime/src/engine/tool-script.ts" },
          },
          {
            id: "read",
            tool: "read_file",
            args: { path: "$find.lines[0]", start_line: 1, end_line: 15 },
          },
        ]),
      },
      context,
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("=== Step find (find_files)");
    expect(result.output).toContain("tool-script.ts");
    expect(result.output).toContain("=== Step read (read_file)");
    expect(result.output).toContain("ToolScriptEngine");
    expect(result.output).toContain("2/2 steps succeeded");
  });

  it("rejects write_file via the tool handler", async () => {
    const tool = createToolScriptTool({ registry, bus });

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          {
            id: "w",
            tool: "write_file",
            args: { path: "hack.ts", content: "bad" },
          },
        ]),
      },
      context,
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Only readonly tools");
  });

  it("rejects run_command via the tool handler", async () => {
    const tool = createToolScriptTool({ registry, bus });

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          {
            id: "cmd",
            tool: "run_command",
            args: { command: "rm -rf /" },
          },
        ]),
      },
      context,
    );

    expect(result.success).toBe(false);
    // run_command is "mutating" category
    expect(result.error).toContain("Only readonly tools");
  });

  it("3-step real-world batch: find test files, search for pattern, read first match", async () => {
    const tool = createToolScriptTool({ registry, bus });

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          {
            id: "tests",
            tool: "find_files",
            args: { pattern: "packages/runtime/src/engine/*.test.ts", max_results: 10 },
          },
          {
            id: "grep",
            tool: "search_files",
            args: {
              pattern: "describe\\(",
              path: "packages/runtime/src/engine",
              file_pattern: "*.test.ts",
              max_results: 10,
            },
          },
          {
            id: "status",
            tool: "git_status",
            args: {},
          },
        ]),
      },
      context,
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("=== Step tests (find_files)");
    expect(result.output).toContain("=== Step grep (search_files)");
    expect(result.output).toContain("=== Step status (git_status)");
    expect(result.output).toContain("3/3 steps succeeded");
  });
});
