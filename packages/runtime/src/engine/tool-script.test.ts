/**
 * Tests for programmatic readonly tool scripts.
 */

import { describe, it, expect, beforeEach } from "vitest";

import { createToolScriptTool } from "./tool-script-tool.js";
import { ToolScriptEngine } from "./tool-script.js";
import type { ToolSpec, ToolContext, ToolResult } from "../core/index.js";
import { AgentType, EventBus } from "../core/index.js";
import { ToolRegistry } from "../tools/index.js";

function makeTool(
  name: string,
  category: ToolSpec["category"],
  handler: (params: Record<string, unknown>) => Promise<ToolResult>,
): ToolSpec {
  return {
    name,
    description: `Mock ${name}`,
    category,
    paramSchema: { type: "object" },
    resultSchema: { type: "object" },
    handler: async (params) => handler(params),
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

describe("ToolScriptEngine", () => {
  it("executes TypeScript that calls multiple readonly tools and prints only a summary", async () => {
    registry.register(makeReadonlyTool("read_file", async ({ path }) => successResult(`contents:${path}`)));
    registry.register(makeReadonlyTool("git_status", async () => successResult("On branch main")));

    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus });
    const result = await engine.execute({
      script: `
        type Row = { path: string; size: number };
        const file = await tools.read_file({ path: "a.ts" });
        const status = await tools.git_status({});
        const rows: Row[] = [{ path: "a.ts", size: file.output.length }];
        print(JSON.stringify({ rows, status: status.output.includes("main") }));
      `,
    });

    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe('{"rows":[{"path":"a.ts","size":13}],"status":true}');
    expect(result.output).not.toContain("On branch main");
    expect(result.toolCallCount).toBe(2);
  });

  it("supports Promise.all for parallel readonly calls", async () => {
    registry.register(makeReadonlyTool("read_file", async ({ path }) => successResult(String(path))));

    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus });
    const result = await engine.execute({
      script: `
        const results = await Promise.all([
          tools.read_file({ path: "a.ts" }),
          tools.read_file({ path: "b.ts" }),
        ]);
        print(results.map((r) => r.output).join(","));
      `,
    });

    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("a.ts,b.ts");
    expect(result.toolCallCount).toBe(2);
  });

  it("returns inner tool failures to the script for local handling", async () => {
    registry.register(makeReadonlyTool("read_file", async () => errorResult("missing file")));

    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus });
    const result = await engine.execute({
      script: `
        const result = await tools.read_file({ path: "missing.ts" });
        print(result.success ? result.output : "handled:" + result.error);
      `,
    });

    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("handled:missing file");
  });

  it("emits nested tool telemetry scoped to the outer script call", async () => {
    const before: string[] = [];
    const after: Array<{ name: string; callId: string; agentId?: string; parentAgentId?: string | null }> = [];
    bus.on("tool:before", (event) => before.push(event.name));
    bus.on("tool:after", (event) =>
      after.push({
        name: event.name,
        callId: event.callId,
        agentId: event.agentId,
        parentAgentId: event.parentAgentId,
      })
    );
    registry.register(makeReadonlyTool("read_file", async () => successResult("very large raw content")));

    const engine = new ToolScriptEngine({
      registry,
      context: {
        ...defaultContext,
        callId: "call_script_outer",
        agentId: "root-sub-1",
        parentAgentId: "root",
        depth: 1,
        agentType: AgentType.EXPLORE,
      },
      bus,
    });
    const result = await engine.execute({
      script: `
        const result = await tools.read_file({ path: "a.ts" });
        print("len=" + result.output.length);
      `,
    });

    expect(result.output.trim()).toBe("len=22");
    expect(before).toEqual(["read_file"]);
    expect(after).toEqual([{
      name: "read_file",
      callId: "call_script_outer_script_read_file_1",
      agentId: "root-sub-1",
      parentAgentId: "root",
    }]);
    expect(result.output).not.toContain("very large raw content");
  });

  it("keeps fallback nested tool call IDs unique across script engines", async () => {
    const callIds: string[] = [];
    bus.on("tool:after", (event) => callIds.push(event.callId));
    registry.register(makeReadonlyTool("read_file", async ({ path }) => successResult(String(path))));

    const first = new ToolScriptEngine({ registry, context: defaultContext, bus });
    const second = new ToolScriptEngine({ registry, context: defaultContext, bus });

    await first.execute({ script: `await tools.read_file({ path: "a.ts" });` });
    await second.execute({ script: `await tools.read_file({ path: "b.ts" });` });

    expect(callIds).toHaveLength(2);
    expect(new Set(callIds).size).toBe(2);
    expect(callIds.every((callId) => callId.endsWith("_script_read_file_1"))).toBe(true);
  });

  it("rejects non-readonly tools", async () => {
    registry.register(makeTool("run_command", "workflow", async () => successResult("bad")));

    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus });
    const result = await engine.execute({
      script: `await tools.run_command({ command: "echo nope" });`,
    });

    expect(result.success).toBe(false);
    expect(result.error).toContain('Tool "run_command" is not available');
  });

  it("does not expose execute_tool_script recursively", async () => {
    registry.register(makeReadonlyTool("execute_tool_script", async () => successResult("nested")));

    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus });
    const result = await engine.execute({
      script: `await tools.execute_tool_script({ script: "print(1)" });`,
    });

    expect(result.success).toBe(false);
    expect(result.error).toContain('Tool "execute_tool_script" is not available');
  });

  it("blocks imports and require/process access", async () => {
    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus });

    const importResult = await engine.execute({
      script: `import { readFileSync } from "node:fs"; print(readFileSync("/etc/passwd", "utf8"));`,
    });
    const requireResult = await engine.execute({
      script: `print(typeof require + ":" + typeof process);`,
    });

    expect(importResult.success).toBe(false);
    expect(importResult.error).toContain("import");
    expect(requireResult.success).toBe(true);
    expect(requireResult.output.trim()).toBe("undefined:undefined");
  });

  it("blocks Function-constructor escape attempts", async () => {
    registry.register(makeReadonlyTool("read_file", async () => successResult("ok")));
    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus });

    const toolFunctionResult = await engine.execute({
      script: `print(tools.read_file.constructor("return typeof process")());`,
    });
    const printFunctionResult = await engine.execute({
      script: `print(print.constructor("return typeof process")());`,
    });

    expect(toolFunctionResult.success).toBe(false);
    expect(toolFunctionResult.error).toContain("Code generation from strings disallowed");
    expect(printFunctionResult.success).toBe(false);
    expect(printFunctionResult.error).toContain("Code generation from strings disallowed");
  });

  it("kills scripts that time out", async () => {
    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus });
    const result = await engine.execute({
      script: `while (true) {}`,
      timeoutMs: 50,
    });

    expect(result.success).toBe(false);
    expect(result.error).toContain("timed out");
  });

  it("enforces stdout and inner tool call limits", async () => {
    registry.register(makeReadonlyTool("read_file", async () => successResult("ok")));
    const engine = new ToolScriptEngine({ registry, context: defaultContext, bus, maxToolCalls: 2 });

    const cappedOutput = await engine.execute({
      script: `print("123456");`,
      maxOutputChars: 5,
    });
    const cappedCalls = await engine.execute({
      script: `
        await tools.read_file({});
        await tools.read_file({});
        await tools.read_file({});
      `,
    });

    expect(cappedOutput.success).toBe(false);
    expect(cappedOutput.error).toContain("stdout exceeded");
    expect(cappedCalls.success).toBe(false);
    expect(cappedCalls.error).toContain("maximum of 2 tool call");
  });
});

describe("createToolScriptTool", () => {
  it("describes natural adoption cases in the tool surface", () => {
    const tool = createToolScriptTool({ registry, bus });
    const scriptProperty = (tool.paramSchema.properties as Record<string, { description?: string }>)["script"];

    expect(tool.description).toContain("Default to this as the first inspection tool");
    expect(tool.description).toContain("known-path multi-file audits");
    expect(tool.description).toContain("grouped read_file checks");
    expect(tool.description).toContain("security-leakage verification");
    expect(tool.description).toContain("broad unknown-scope reconnaissance");
    expect(scriptProperty?.description).toContain("3+ readonly calls");
    expect(scriptProperty?.description).toContain("known-path multi-file read_file batches");
  });

  it("runs script input and returns final stdout", async () => {
    registry.register(makeReadonlyTool("find_files", async () => successResult("a.ts\nb.ts")));
    const tool = createToolScriptTool({ registry, bus });

    const result = await tool.handler(
      {
        script: `
          const files = await tools.find_files({ pattern: "*.ts" });
          print(files.output.split("\\n").length + " files");
        `,
      },
      defaultContext,
    );

    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("2 files");
  });

  it("rejects old step-array input with clear guidance", async () => {
    const tool = createToolScriptTool({ registry, bus });

    const result = await tool.handler(
      {
        steps: JSON.stringify([{ id: "find", tool: "find_files", args: {} }]),
      },
      defaultContext,
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("script");
    expect(result.error).toContain("steps");
  });
});
