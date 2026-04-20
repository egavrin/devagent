import { readdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect } from "vitest";

import { createDefaultToolRegistry } from "./index.js";
import type { ToolSpec } from "../core/types.js";

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

  it("avoids runtime imports from the async core barrel in tool sources", () => {
    const sourceFiles = collectTypeScriptFiles(import.meta.dirname).filter(
      (filePath) => !filePath.endsWith(".test.ts"),
    );

    for (const filePath of sourceFiles) {
      const content = readFileSync(filePath, "utf-8");
      const runtimeCoreImports = content.match(
        /^import\s+(?!type\b).*from\s+["'](?:\.\.\/core\/index\.js|\.\.\/\.\.\/core\/index\.js)["'];?$/gm,
      );
      expect(runtimeCoreImports, filePath).toBeNull();
    }
  });
});

function collectTypeScriptFiles(dirPath: string): string[] {
  const entries = readdirSync(dirPath, { withFileTypes: true });
  const files: string[] = [];

  for (const entry of entries) {
    const entryPath = join(dirPath, entry.name);
    if (entry.isDirectory()) {
      files.push(...collectTypeScriptFiles(entryPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith(".ts")) {
      files.push(entryPath);
    }
  }

  return files;
}
