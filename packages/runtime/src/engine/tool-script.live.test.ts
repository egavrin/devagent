/**
 * Live integration tests for programmatic readonly tool scripts.
 */

import { resolve } from "node:path";
import { describe, it, expect, beforeEach } from "vitest";

import { createToolScriptTool } from "./tool-script-tool.js";
import { ToolScriptEngine } from "./tool-script.js";
import { EventBus } from "../core/index.js";
import type { ToolContext } from "../core/index.js";
import { createDefaultToolRegistry } from "../tools/index.js";
import type { ToolRegistry } from "../tools/index.js";

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

it("finds, reads, and summarizes real repo files without returning raw intermediate output", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    script: `
      const files = await tools.find_files({ pattern: "packages/runtime/src/engine/tool-script*.ts" });
      const first = files.output.split("\\n").find((line) => line.endsWith("tool-script.ts"));
      const read = await tools.read_file({ path: first });
      print(JSON.stringify({
        count: files.output.split("\\n").filter(Boolean).length,
        hasEngine: read.output.includes("ToolScriptEngine"),
      }));
    `,
  });

  expect(result.success).toBe(true);
  expect(JSON.parse(result.output.trim())).toEqual({ count: 4, hasEngine: true });
  expect(result.output).not.toContain("programmatic readonly tool scripts");
});

it("supports real search, git status, and parallel execution", async () => {
  const engine = new ToolScriptEngine({ registry, context, bus });

  const result = await engine.execute({
    script: `
      const [search, status] = await Promise.all([
        tools.search_files({
          pattern: "ToolCategory",
          path: "packages/runtime/src/core",
          file_pattern: "**/types.ts",
          max_results: 5,
        }),
        tools.git_status({}),
      ]);
      print(JSON.stringify({
        foundCategory: search.output.includes("ToolCategory"),
        hasStatus: status.output.length > 0,
      }));
    `,
  });

  expect(result.success).toBe(true);
  expect(JSON.parse(result.output.trim())).toEqual({ foundCategory: true, hasStatus: true });
});

it("emits grouped nested telemetry for real tool calls", async () => {
  const beforeNames: string[] = [];
  const afterNames: string[] = [];
  bus.on("tool:before", (event) => beforeNames.push(event.name));
  bus.on("tool:after", (event) => afterNames.push(event.name));

  const engine = new ToolScriptEngine({ registry, context, bus });
  await engine.execute({
    script: `
      await Promise.all([
        tools.find_files({ pattern: "package.json", max_results: 5 }),
        tools.git_status({}),
      ]);
      print("done");
    `,
  });

  expect(beforeNames.sort()).toEqual(["find_files", "git_status"]);
  expect(afterNames.sort()).toEqual(["find_files", "git_status"]);
});

describe("createToolScriptTool (live)", () => {
  it("executes the TypeScript script through the tool handler", async () => {
    const tool = createToolScriptTool({ registry, bus });

    const result = await tool.handler(
      {
        script: `
          const files = await tools.find_files({
            pattern: "packages/runtime/src/engine/tool-script*.ts",
          });
          const names: string[] = files.output.split("\\n").filter(Boolean);
          print(names.length + " script files");
        `,
      },
      context,
    );

    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("4 script files");
  });

  it("rejects workflow tools through the programmatic bridge", async () => {
    const tool = createToolScriptTool({ registry, bus });

    const result = await tool.handler(
      {
        script: `await tools.run_command({ command: "echo nope" });`,
      },
      context,
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain('Tool "run_command" is not available');
  });
});
