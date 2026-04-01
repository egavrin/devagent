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

describe("ToolRegistry deferred tools", () => {
  it("registerDeferred stores tool, getDeferred returns stubs, getLoaded excludes deferred", () => {
    const reg = new ToolRegistry();
    const tool = makeTool("test");
    reg.registerDeferred(tool);

    const stubs = reg.getDeferred();
    expect(stubs).toHaveLength(1);
    expect(stubs[0]!.name).toBe("test");
    expect(stubs[0]!.description).toContain("test");

    const loaded = reg.getLoaded();
    expect(loaded.find((t) => t.name === "test")).toBeUndefined();
  });

  it("resolve moves deferred to loaded", () => {
    const reg = new ToolRegistry();
    reg.registerDeferred(makeTool("alpha"));

    expect(reg.getDeferred()).toHaveLength(1);
    expect(reg.getLoaded()).toHaveLength(0);

    const resolved = reg.resolve("alpha");
    expect(resolved).not.toBeNull();
    expect(resolved!.name).toBe("alpha");

    expect(reg.getDeferred()).toHaveLength(0);
    expect(reg.getLoaded()).toHaveLength(1);
    expect(reg.getLoaded()[0]!.name).toBe("alpha");
  });

  it("search matches by keyword and auto-resolves", () => {
    const reg = new ToolRegistry();
    reg.registerDeferred(makeTool("git_diff"));
    reg.registerDeferred(makeTool("git_log"));
    reg.registerDeferred(makeTool("diagnostics"));

    const results = reg.search("git");
    expect(results.length).toBeGreaterThanOrEqual(2);
    expect(results.every((s) => s.name.includes("git"))).toBe(true);

    // Auto-resolved: now in loaded
    const loaded = reg.getLoaded();
    expect(loaded.find((t) => t.name === "git_diff")).toBeDefined();
    expect(loaded.find((t) => t.name === "git_log")).toBeDefined();

    // diagnostics was not matched, stays deferred
    expect(reg.getDeferred().find((s) => s.name === "diagnostics")).toBeDefined();
  });

  it("get auto-resolves deferred", () => {
    const reg = new ToolRegistry();
    reg.registerDeferred(makeTool("symbols"));

    const tool = reg.get("symbols");
    expect(tool.name).toBe("symbols");

    // Now loaded
    expect(reg.getLoaded().find((t) => t.name === "symbols")).toBeDefined();
    expect(reg.getDeferred()).toHaveLength(0);
  });

  it("has returns true for both loaded and deferred", () => {
    const reg = new ToolRegistry();
    reg.register(makeTool("loaded_tool"));
    reg.registerDeferred(makeTool("deferred_tool"));

    expect(reg.has("loaded_tool")).toBe(true);
    expect(reg.has("deferred_tool")).toBe(true);
    expect(reg.has("nonexistent")).toBe(false);
  });

  it("getAll includes both loaded and deferred", () => {
    const reg = new ToolRegistry();
    reg.register(makeTool("a"));
    reg.registerDeferred(makeTool("b"));

    const all = reg.getAll();
    expect(all).toHaveLength(2);
    expect(all.map((t) => t.name).sort()).toEqual(["a", "b"]);
  });

  it("duplicate throws for deferred-deferred and loaded-deferred conflicts", () => {
    const reg = new ToolRegistry();
    reg.register(makeTool("dup"));
    expect(() => reg.registerDeferred(makeTool("dup"))).toThrow('Tool "dup" is already registered');

    const reg2 = new ToolRegistry();
    reg2.registerDeferred(makeTool("dup2"));
    expect(() => reg2.registerDeferred(makeTool("dup2"))).toThrow('Tool "dup2" is already registered');
    expect(() => reg2.register(makeTool("dup2"))).toThrow('Tool "dup2" is already registered');
  });
});
