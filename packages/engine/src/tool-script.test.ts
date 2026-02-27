/**
 * Tests for ToolScriptEngine and createToolScriptTool.
 */

import { describe, it, expect, beforeEach } from "vitest";
import { ToolScriptEngine } from "./tool-script.js";
import { createToolScriptTool } from "./tool-script-tool.js";
import type { ToolScriptStep, ToolScript } from "./tool-script.js";
import type { ToolSpec, ToolContext, ToolResult } from "@devagent/core";
import { EventBus } from "@devagent/core";
import { ToolRegistry } from "@devagent/tools";

// ─── Helpers ────────────────────────────────────────────────

function makeTool(
  name: string,
  category: "readonly" | "mutating" | "workflow" | "external",
  handler: (params: Record<string, unknown>) => Promise<ToolResult>,
): ToolSpec {
  return {
    name,
    description: `Mock ${name}`,
    category,
    paramSchema: { type: "object" },
    resultSchema: { type: "object" },
    handler: async (params, _ctx) => handler(params),
  };
}

function makeReadonlyTool(
  name: string,
  handler: (params: Record<string, unknown>) => Promise<ToolResult>,
): ToolSpec {
  return makeTool(name, "readonly", handler);
}

function successResult(output: string): ToolResult {
  return { success: true, output, error: null, artifacts: [] };
}

function errorResult(error: string): ToolResult {
  return { success: false, output: "", error, artifacts: [] };
}

const defaultContext: ToolContext = {
  repoRoot: "/tmp",
  config: {} as never,
  sessionId: "test",
};

let registry: ToolRegistry;
let bus: EventBus;

beforeEach(() => {
  registry = new ToolRegistry();
  bus = new EventBus();
});

// ─── Engine Tests ───────────────────────────────────────────

describe("ToolScriptEngine", () => {
  it("executes a single-step script", async () => {
    registry.register(
      makeReadonlyTool("read_file", async () => successResult("file contents")),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [{ id: "r1", tool: "read_file", args: { path: "a.ts" } }],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(true);
    expect(result.steps[0]!.output).toBe("file contents");
    expect(result.steps[0]!.id).toBe("r1");
    expect(result.steps[0]!.tool).toBe("read_file");
    expect(result.totalDurationMs).toBeGreaterThanOrEqual(0);
    expect(result.truncated).toBe(false);
  });

  it("rejects namespaced tool names in script steps", async () => {
    registry.register(
      makeReadonlyTool("read_file", async () => successResult("ok")),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [{ id: "r1", tool: "functions.read_file", args: { path: "a.ts" } }],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain('Invalid tool name "functions.read_file"');
    expect(result.steps[0]!.error).toContain('"read_file"');
  });

  it("executes multi-step script without references", async () => {
    registry.register(
      makeReadonlyTool("find_files", async () =>
        successResult("a.ts\nb.ts\nc.ts"),
      ),
    );
    registry.register(
      makeReadonlyTool("git_status", async () =>
        successResult("On branch main"),
      ),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        { id: "find", tool: "find_files", args: { pattern: "**/*.ts" } },
        { id: "status", tool: "git_status", args: {} },
      ],
    });

    expect(result.steps).toHaveLength(2);
    expect(result.steps[0]!.success).toBe(true);
    expect(result.steps[0]!.output).toBe("a.ts\nb.ts\nc.ts");
    expect(result.steps[1]!.success).toBe(true);
    expect(result.steps[1]!.output).toBe("On branch main");
  });

  it("resolves $stepId reference (full output)", async () => {
    const receivedArgs: Record<string, unknown>[] = [];

    registry.register(
      makeReadonlyTool("find_files", async () =>
        successResult("src/index.ts"),
      ),
    );
    registry.register(
      makeReadonlyTool("read_file", async (params) => {
        receivedArgs.push({ ...params });
        return successResult("export default {};");
      }),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        { id: "find", tool: "find_files", args: { pattern: "*.ts" } },
        { id: "read", tool: "read_file", args: { path: "$find" } },
      ],
    });

    expect(result.steps).toHaveLength(2);
    expect(result.steps[1]!.success).toBe(true);
    // The $find reference should have been resolved to "src/index.ts"
    expect(receivedArgs[0]!["path"]).toBe("src/index.ts");
  });

  it("resolves $stepId.lines[N] reference", async () => {
    const receivedPaths: string[] = [];

    registry.register(
      makeReadonlyTool("find_files", async () =>
        successResult("first.ts\nsecond.ts\nthird.ts"),
      ),
    );
    registry.register(
      makeReadonlyTool("read_file", async (params) => {
        receivedPaths.push(params["path"] as string);
        return successResult(`contents of ${params["path"]}`);
      }),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        { id: "find", tool: "find_files", args: { pattern: "*.ts" } },
        { id: "read0", tool: "read_file", args: { path: "$find.lines[0]" } },
        { id: "read2", tool: "read_file", args: { path: "$find.lines[2]" } },
      ],
    });

    expect(result.steps).toHaveLength(3);
    expect(receivedPaths[0]).toBe("first.ts");
    expect(receivedPaths[1]).toBe("third.ts");
  });

  it("rejects mutating tools", async () => {
    registry.register(
      makeTool("write_file", "mutating", async () =>
        successResult("written"),
      ),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        { id: "w", tool: "write_file", args: { path: "a.ts", content: "x" } },
      ],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain("Only readonly tools");
    expect(result.steps[0]!.error).toContain("write_file");
  });

  it("rejects unknown tools", async () => {
    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [{ id: "x", tool: "nonexistent_tool", args: {} }],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain('Unknown tool: "nonexistent_tool"');
  });

  it("continues on step failure (fail-forward)", async () => {
    registry.register(
      makeReadonlyTool("search_files", async () =>
        errorResult("Pattern not found"),
      ),
    );
    registry.register(
      makeReadonlyTool("git_status", async () =>
        successResult("On branch main"),
      ),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        {
          id: "search",
          tool: "search_files",
          args: { query: "missing" },
        },
        { id: "status", tool: "git_status", args: {} },
      ],
    });

    expect(result.steps).toHaveLength(2);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[1]!.success).toBe(true);
    expect(result.steps[1]!.output).toBe("On branch main");
  });

  it("truncates output when exceeding maxOutputBytes", async () => {
    const bigOutput = "x".repeat(1024); // 1KB per call

    registry.register(
      makeReadonlyTool("read_file", async () => successResult(bigOutput)),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
      maxOutputBytes: 2500, // Less than 3 * 1024
    });

    const result = await engine.execute({
      steps: [
        { id: "r1", tool: "read_file", args: { path: "a.ts" } },
        { id: "r2", tool: "read_file", args: { path: "b.ts" } },
        { id: "r3", tool: "read_file", args: { path: "c.ts" } },
        { id: "r4", tool: "read_file", args: { path: "d.ts" } },
      ],
    });

    expect(result.truncated).toBe(true);
    // First 2 execute normally, 3rd exceeds limit, 4th is skipped
    const executed = result.steps.filter((s) => s.durationMs > 0 || s.success);
    const skipped = result.steps.filter(
      (s) => s.error?.includes("Skipped: output limit exceeded"),
    );
    expect(executed.length + skipped.length).toBe(4);
    expect(skipped.length).toBeGreaterThan(0);
  });

  it("rejects empty script", async () => {
    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({ steps: [] });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain("at least one step");
  });

  it("produces error marker for failed step reference", async () => {
    const receivedArgs: Record<string, unknown>[] = [];

    registry.register(
      makeReadonlyTool("search_files", async () =>
        errorResult("Pattern not found"),
      ),
    );
    registry.register(
      makeReadonlyTool("read_file", async (params) => {
        receivedArgs.push({ ...params });
        return successResult("ok");
      }),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        { id: "search", tool: "search_files", args: { query: "test" } },
        { id: "read", tool: "read_file", args: { path: "$search" } },
      ],
    });

    expect(result.steps).toHaveLength(2);
    expect(receivedArgs[0]!["path"]).toBe(
      '<ref error: step "search" failed>',
    );
  });

  it("rejects forward references", async () => {
    registry.register(
      makeReadonlyTool("read_file", async () => successResult("ok")),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        { id: "a", tool: "read_file", args: { path: "$b" } },
        { id: "b", tool: "read_file", args: { path: "file.ts" } },
      ],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain("forward reference");
    expect(result.steps[0]!.error).toContain('"b"');
  });

  it("rejects duplicate step IDs", async () => {
    registry.register(
      makeReadonlyTool("read_file", async () => successResult("ok")),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        { id: "r", tool: "read_file", args: { path: "a.ts" } },
        { id: "r", tool: "read_file", args: { path: "b.ts" } },
      ],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain('Duplicate step ID: "r"');
  });

  it("rejects self-reference", async () => {
    registry.register(
      makeReadonlyTool("read_file", async () => successResult("ok")),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [{ id: "self", tool: "read_file", args: { path: "$self" } }],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain("references itself");
  });

  it("prevents recursive scripts (execute_tool_script inside script)", async () => {
    // Register the tool script tool itself — the engine should block it by name
    registry.register(
      makeReadonlyTool("execute_tool_script", async () =>
        successResult("should never run"),
      ),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [
        {
          id: "nested",
          tool: "execute_tool_script",
          args: { steps: "[]" },
        },
      ],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain("recursion prevention");
  });

  it("emits tool:before and tool:after events per step", async () => {
    const beforeEvents: Array<{ name: string; callId: string }> = [];
    const afterEvents: Array<{
      name: string;
      callId: string;
      durationMs: number;
    }> = [];

    bus.on("tool:before", (e) =>
      beforeEvents.push({ name: e.name, callId: e.callId }),
    );
    bus.on("tool:after", (e) =>
      afterEvents.push({
        name: e.name,
        callId: e.callId,
        durationMs: e.durationMs,
      }),
    );

    registry.register(
      makeReadonlyTool("find_files", async () =>
        successResult("a.ts\nb.ts"),
      ),
    );
    registry.register(
      makeReadonlyTool("git_status", async () =>
        successResult("On branch main"),
      ),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    await engine.execute({
      steps: [
        { id: "find", tool: "find_files", args: {} },
        { id: "status", tool: "git_status", args: {} },
      ],
    });

    expect(beforeEvents).toHaveLength(2);
    expect(afterEvents).toHaveLength(2);
    expect(beforeEvents[0]!.name).toBe("find_files");
    expect(beforeEvents[0]!.callId).toBe("script_find");
    expect(beforeEvents[1]!.name).toBe("git_status");
    expect(afterEvents[0]!.name).toBe("find_files");
    expect(afterEvents[1]!.name).toBe("git_status");
  });

  it("handles out-of-bounds line index", async () => {
    const receivedArgs: Record<string, unknown>[] = [];

    registry.register(
      makeReadonlyTool("find_files", async () =>
        successResult("only-one-line.ts"),
      ),
    );
    registry.register(
      makeReadonlyTool("read_file", async (params) => {
        receivedArgs.push({ ...params });
        return successResult("ok");
      }),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    await engine.execute({
      steps: [
        { id: "find", tool: "find_files", args: {} },
        { id: "read", tool: "read_file", args: { path: "$find.lines[99]" } },
      ],
    });

    expect(receivedArgs[0]!["path"]).toContain("out of bounds");
    expect(receivedArgs[0]!["path"]).toContain("$find.lines[99]");
  });

  it("rejects scripts exceeding maxSteps", async () => {
    registry.register(
      makeReadonlyTool("read_file", async () => successResult("ok")),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
      maxSteps: 3,
    });

    const steps: ToolScriptStep[] = Array.from({ length: 5 }, (_, i) => ({
      id: `s${i}`,
      tool: "read_file",
      args: { path: `${i}.ts` },
    }));

    const result = await engine.execute({ steps });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toContain("exceeds maximum");
  });

  it("handles tool handler that throws", async () => {
    registry.register(
      makeReadonlyTool("read_file", async () => {
        throw new Error("Unexpected disk error");
      }),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const result = await engine.execute({
      steps: [{ id: "r", tool: "read_file", args: { path: "bad.ts" } }],
    });

    expect(result.steps).toHaveLength(1);
    expect(result.steps[0]!.success).toBe(false);
    expect(result.steps[0]!.error).toBe("Unexpected disk error");
  });

  // ─── Hybrid Parallel Execution Tests ──────────────────────

  it("runs independent steps in parallel (no references between them)", async () => {
    const executionLog: Array<{ id: string; event: "start" | "end"; time: number }> = [];
    const baseTime = Date.now();

    const makeSlowTool = (name: string): ToolSpec =>
      makeReadonlyTool(name, async (params) => {
        const id = params["id"] as string;
        executionLog.push({ id, event: "start", time: Date.now() - baseTime });
        await new Promise((r) => setTimeout(r, 80));
        executionLog.push({ id, event: "end", time: Date.now() - baseTime });
        return successResult(`${id} done`);
      });

    registry.register(makeSlowTool("find_files"));
    registry.register(makeSlowTool("git_status"));
    registry.register(makeSlowTool("search_files"));

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    const start = Date.now();
    const result = await engine.execute({
      steps: [
        { id: "a", tool: "find_files", args: { id: "a" } },
        { id: "b", tool: "git_status", args: { id: "b" } },
        { id: "c", tool: "search_files", args: { id: "c" } },
      ],
    });
    const elapsed = Date.now() - start;

    expect(result.steps).toHaveLength(3);
    expect(result.steps.every((s) => s.success)).toBe(true);

    // Parallel: 3 × 80ms should finish in ~80-130ms, not 240ms+
    expect(elapsed).toBeLessThan(300);

    // Verify all started roughly together (within 30ms)
    const starts = executionLog.filter((e) => e.event === "start");
    const maxDiff = Math.max(...starts.map((s) => s.time)) - Math.min(...starts.map((s) => s.time));
    expect(maxDiff).toBeLessThan(50);
  });

  it("runs dependent steps sequentially, independent steps in parallel (hybrid)", async () => {
    const executionOrder: string[] = [];

    registry.register(
      makeReadonlyTool("find_files", async () => {
        executionOrder.push("find");
        return successResult("a.ts\nb.ts");
      }),
    );
    registry.register(
      makeReadonlyTool("git_status", async () => {
        executionOrder.push("git_status");
        return successResult("On branch main");
      }),
    );
    registry.register(
      makeReadonlyTool("read_file", async (params) => {
        executionOrder.push(`read:${params["path"]}`);
        return successResult(`contents of ${params["path"]}`);
      }),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    // Script: find + git_status are independent (wave 1, parallel),
    // read depends on find (wave 2, sequential)
    const result = await engine.execute({
      steps: [
        { id: "find", tool: "find_files", args: { pattern: "*.ts" } },
        { id: "status", tool: "git_status", args: {} },
        { id: "read", tool: "read_file", args: { path: "$find.lines[0]" } },
      ],
    });

    expect(result.steps).toHaveLength(3);
    expect(result.steps.every((s) => s.success)).toBe(true);

    // find and git_status execute in wave 1 (parallel) — before read
    // read must come after find completes
    const readIdx = executionOrder.indexOf("read:a.ts");
    const findIdx = executionOrder.indexOf("find");
    expect(findIdx).toBeLessThan(readIdx);

    // read should have resolved $find.lines[0] to "a.ts"
    expect(result.steps[2]!.output).toBe("contents of a.ts");
  });

  it("cascading dependencies execute in correct wave order", async () => {
    const executionOrder: string[] = [];

    registry.register(
      makeReadonlyTool("find_files", async () => {
        executionOrder.push("find");
        return successResult("src/main.ts");
      }),
    );
    registry.register(
      makeReadonlyTool("read_file", async (params) => {
        const path = params["path"] as string;
        executionOrder.push(`read:${path}`);
        return successResult(`line1\nline2\nline3`);
      }),
    );
    registry.register(
      makeReadonlyTool("search_files", async (params) => {
        executionOrder.push(`search:${params["pattern"]}`);
        return successResult(`match in ${params["pattern"]}`);
      }),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    // Chain: find → read (depends on find) → search (depends on read)
    const result = await engine.execute({
      steps: [
        { id: "find", tool: "find_files", args: { pattern: "*.ts" } },
        { id: "read", tool: "read_file", args: { path: "$find" } },
        { id: "grep", tool: "search_files", args: { pattern: "$read.lines[0]" } },
      ],
    });

    expect(result.steps).toHaveLength(3);
    expect(result.steps.every((s) => s.success)).toBe(true);

    // Strict ordering: find → read → search (three waves, each with 1 step)
    expect(executionOrder[0]).toBe("find");
    expect(executionOrder[1]).toBe("read:src/main.ts");
    expect(executionOrder[2]).toBe("search:line1");
  });

  it("diamond dependency graph: a → (b, c) → d", async () => {
    const executionOrder: string[] = [];
    const waveTimestamps: Array<{ id: string; time: number }> = [];
    const baseTime = Date.now();

    registry.register(
      makeReadonlyTool("find_files", async () => {
        executionOrder.push("a");
        waveTimestamps.push({ id: "a", time: Date.now() - baseTime });
        return successResult("root.ts");
      }),
    );
    registry.register(
      makeReadonlyTool("read_file", async (params) => {
        const id = `read:${params["path"]}`;
        executionOrder.push(id);
        waveTimestamps.push({ id, time: Date.now() - baseTime });
        await new Promise((r) => setTimeout(r, 50));
        return successResult(`content-${params["path"]}`);
      }),
    );
    registry.register(
      makeReadonlyTool("search_files", async (params) => {
        const id = `search:${params["pattern"]}`;
        executionOrder.push(id);
        waveTimestamps.push({ id, time: Date.now() - baseTime });
        await new Promise((r) => setTimeout(r, 50));
        return successResult(`found-${params["pattern"]}`);
      }),
    );
    registry.register(
      makeReadonlyTool("git_status", async () => {
        executionOrder.push("d");
        waveTimestamps.push({ id: "d", time: Date.now() - baseTime });
        return successResult("final");
      }),
    );

    const engine = new ToolScriptEngine({
      registry,
      context: defaultContext,
      bus,
    });

    // Diamond: a has no deps → b,c both depend on a → d depends on b and c
    const result = await engine.execute({
      steps: [
        { id: "a", tool: "find_files", args: { pattern: "*.ts" } },
        { id: "b", tool: "read_file", args: { path: "$a" } },
        { id: "c", tool: "search_files", args: { pattern: "$a" } },
        { id: "d", tool: "git_status", args: { check: "$b $c" } },
      ],
    });

    expect(result.steps).toHaveLength(4);
    expect(result.steps.every((s) => s.success)).toBe(true);

    // Wave 1: a (no deps)
    // Wave 2: b and c (both depend only on a) — should run in parallel
    // Wave 3: d (depends on b and c)
    const aIdx = executionOrder.indexOf("a");
    const bIdx = executionOrder.indexOf("read:root.ts");
    const cIdx = executionOrder.indexOf("search:root.ts");
    const dIdx = executionOrder.indexOf("d");

    expect(aIdx).toBeLessThan(bIdx);
    expect(aIdx).toBeLessThan(cIdx);
    expect(bIdx).toBeLessThan(dIdx);
    expect(cIdx).toBeLessThan(dIdx);

    // b and c should have started in the same wave (close timestamps)
    const bTime = waveTimestamps.find((w) => w.id === "read:root.ts")!.time;
    const cTime = waveTimestamps.find((w) => w.id === "search:root.ts")!.time;
    expect(Math.abs(bTime - cTime)).toBeLessThan(30);
  });
});

// ─── Factory Tests ──────────────────────────────────────────

describe("createToolScriptTool", () => {
  it("returns correct metadata", () => {
    const tool = createToolScriptTool({ registry, bus });

    expect(tool.name).toBe("execute_tool_script");
    expect(tool.category).toBe("readonly");
    expect(tool.paramSchema.required).toContain("steps");
  });

  it("parses JSON and returns formatted output", async () => {
    registry.register(
      makeReadonlyTool("find_files", async () =>
        successResult("a.ts\nb.ts"),
      ),
    );
    registry.register(
      makeReadonlyTool("read_file", async (params) =>
        successResult(`contents of ${params["path"]}`),
      ),
    );

    const tool = createToolScriptTool({ registry, bus });
    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { id: "find", tool: "find_files", args: { pattern: "*.ts" } },
          {
            id: "read",
            tool: "read_file",
            args: { path: "$find.lines[0]" },
          },
        ]),
      },
      defaultContext,
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("=== Step find (find_files)");
    expect(result.output).toContain("a.ts\nb.ts");
    expect(result.output).toContain("=== Step read (read_file)");
    expect(result.output).toContain("contents of a.ts");
    expect(result.output).toContain("2/2 steps succeeded");
  });

  it("returns error on invalid JSON", async () => {
    const tool = createToolScriptTool({ registry, bus });
    const result = await tool.handler(
      { steps: "not valid json {{{" },
      defaultContext,
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Invalid JSON");
  });

  it("returns error on non-array JSON", async () => {
    const tool = createToolScriptTool({ registry, bus });
    const result = await tool.handler(
      { steps: '{"not": "array"}' },
      defaultContext,
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("JSON array");
  });
});
