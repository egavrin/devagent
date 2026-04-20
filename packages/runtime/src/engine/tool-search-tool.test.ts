import { describe, it, expect } from "vitest";

import { createToolSearchTool } from "./tool-search-tool.js";
import type { ToolSpec } from "../core/index.js";
import { ToolRegistry } from "../tools/index.js";

function makeTool(name: string, description: string): ToolSpec {
  return {
    name,
    description,
    category: "readonly" as const,
    paramSchema: { type: "object" },
    resultSchema: { type: "object" },
    handler: async () => ({ success: true, output: "", error: null, artifacts: [] }),
  };
}

describe("tool_search tool", () => {
  it("returns matches for keyword", async () => {
    const registry = new ToolRegistry();
    registry.registerDeferred(makeTool("git_diff", "Show file differences"));
    registry.registerDeferred(makeTool("git_log", "Show commit history"));

    const tool = createToolSearchTool(registry);
    const result = await tool.handler({ query: "git" });

    expect(result.success).toBe(true);
    expect(result.output).toContain("Resolved");
    expect(result.output).toContain("git_diff");
    expect(result.output).toContain("git_log");
  });

  it("returns 'already available' for loaded tools", async () => {
    const registry = new ToolRegistry();
    registry.register(makeTool("read_file", "Read a file from disk"));

    const tool = createToolSearchTool(registry);
    const result = await tool.handler({ query: "read_file" });

    expect(result.success).toBe(true);
    expect(result.output).toContain("already available");
  });

  it("empty query returns error", async () => {
    const registry = new ToolRegistry();
    const tool = createToolSearchTool(registry);

    const result = await tool.handler({ query: "" });
    expect(result.success).toBe(false);
    expect(result.error).toBe("Query is required");

    const result2 = await tool.handler({ query: "   " });
    expect(result2.success).toBe(false);
    expect(result2.error).toBe("Query is required");
  });

  it("max_results limits returned tools", async () => {
    const registry = new ToolRegistry();
    for (let i = 0; i < 10; i++) {
      registry.registerDeferred(makeTool(`tool_${i}`, "test tool"));
    }

    const tool = createToolSearchTool(registry);
    const result = await tool.handler({ query: "tool", max_results: 3 });

    expect(result.success).toBe(true);
    // Output should mention exactly 3 resolved tools
    expect(result.output).toContain("Resolved 3 tool(s)");
  });

  it("resolved tools appear in getLoaded after search", async () => {
    const registry = new ToolRegistry();
    registry.registerDeferred(makeTool("diagnostics", "LSP diagnostics"));

    expect(registry.getLoaded().find((t) => t.name === "diagnostics")).toBeUndefined();

    const tool = createToolSearchTool(registry);
    await tool.handler({ query: "diagnostics" });

    expect(registry.getLoaded().find((t) => t.name === "diagnostics")).toBeDefined();
  });
});
