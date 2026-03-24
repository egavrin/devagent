import { describe, it, expect } from "vitest";
import type { ToolSpec } from "../core/index.js";
import { createDefaultToolRegistry } from "./index.js";

function makeTool(name: string): ToolSpec {
  return {
    name,
    description: `override ${name}`,
    category: "readonly",
    paramSchema: { type: "object" },
    resultSchema: { type: "object" },
    handler: async () => ({ success: true, output: "override", error: null, artifacts: [] }),
  };
}

describe("createDefaultToolRegistry", () => {
  it("replaces built-in tools by name when overrides are provided", async () => {
    const registry = createDefaultToolRegistry({
      overrides: [makeTool("read_file")],
    });

    const result = await registry.get("read_file").handler({}, {
      repoRoot: "/tmp",
      config: {} as never,
      sessionId: "test-session",
    });

    expect(result.output).toBe("override");
  });
});
