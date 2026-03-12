import { describe, it, expect } from "vitest";
import { ToolRegistry } from "./registry.js";
import type { ToolSpec, ToolResult } from "../core/index.js";

function makeTool(name: string, category: "readonly" | "mutating" | "state" = "readonly"): ToolSpec {
  return {
    name,
    description: `Test tool: ${name}`,
    category,
    paramSchema: { type: "object" },
    resultSchema: { type: "object" },
    handler: async () => ({ success: true, output: "ok", error: null, artifacts: [] }),
  };
}

describe("ToolRegistry", () => {
  it("registers and retrieves a tool", () => {
    const registry = new ToolRegistry();
    const tool = makeTool("test_tool");
    registry.register(tool);

    const retrieved = registry.get("test_tool");
    expect(retrieved.name).toBe("test_tool");
  });

  it("throws on duplicate registration", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("dup"));
    expect(() => registry.register(makeTool("dup"))).toThrow(
      'Tool "dup" is already registered',
    );
  });

  it("throws on unknown tool", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("known"));
    expect(() => registry.get("unknown")).toThrow("unknown");
  });

  it("lists all tool names", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("a"));
    registry.register(makeTool("b"));
    registry.register(makeTool("c"));

    const names = registry.list();
    expect(names).toEqual(["a", "b", "c"]);
  });

  it("filters by category", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("read", "readonly"));
    registry.register(makeTool("write", "mutating"));
    registry.register(makeTool("read2", "readonly"));

    const readOnly = registry.getReadOnly();
    expect(readOnly.length).toBe(2);
    expect(readOnly.map((t) => t.name)).toEqual(["read", "read2"]);
  });

  it("has() checks registration", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("exists"));
    expect(registry.has("exists")).toBe(true);
    expect(registry.has("nope")).toBe(false);
  });

  it("getAll() returns all specs", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("a"));
    registry.register(makeTool("b"));

    const all = registry.getAll();
    expect(all.length).toBe(2);
  });

  it("getPlanModeTools() returns readonly and state tools", () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("read", "readonly"));
    registry.register(makeTool("plan", "state"));
    registry.register(makeTool("write", "mutating"));

    const planTools = registry.getPlanModeTools();
    expect(planTools.map((t) => t.name)).toEqual(["read", "plan"]);
  });
});
