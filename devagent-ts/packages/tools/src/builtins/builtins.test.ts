import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import type { ToolContext } from "@devagent/core";
import { readFileTool } from "./read-file.js";
import { writeFileTool } from "./write-file.js";
import { replaceInFileTool } from "./replace-in-file.js";
import { findFilesTool } from "./find-files.js";
import { searchFilesTool } from "./search-files.js";
import { builtinTools } from "./index.js";

let tmpDir: string;
let ctx: ToolContext;

beforeEach(() => {
  tmpDir = mkdtempSync(join(tmpdir(), "devagent-tools-test-"));
  ctx = {
    repoRoot: tmpDir,
    config: {} as ToolContext["config"],
    sessionId: "test-session",
  };

  // Create test files
  mkdirSync(join(tmpDir, "src"), { recursive: true });
  writeFileSync(join(tmpDir, "src", "index.ts"), "export const x = 1;\nexport const y = 2;\n");
  writeFileSync(join(tmpDir, "src", "utils.ts"), "export function add(a: number, b: number) {\n  return a + b;\n}\n");
  writeFileSync(join(tmpDir, "README.md"), "# Test Project\n");
});

afterEach(() => {
  rmSync(tmpDir, { recursive: true, force: true });
});

describe("builtinTools", () => {
  it("has 9 built-in tools", () => {
    expect(builtinTools.length).toBe(9);
  });

  it("all tools have unique names", () => {
    const names = builtinTools.map((t) => t.name);
    expect(new Set(names).size).toBe(names.length);
  });
});

describe("read_file", () => {
  it("reads entire file with line numbers", async () => {
    const result = await readFileTool.handler({ path: "src/index.ts" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("1\texport const x = 1;");
    expect(result.output).toContain("2\texport const y = 2;");
  });

  it("reads a line range", async () => {
    const result = await readFileTool.handler(
      { path: "src/utils.ts", start_line: 2, end_line: 2 },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("2\t  return a + b;");
    expect(result.output).not.toContain("export function");
  });

  it("throws on missing file", async () => {
    await expect(
      readFileTool.handler({ path: "nonexistent.ts" }, ctx),
    ).rejects.toThrow("File not found");
  });
});

describe("write_file", () => {
  it("writes a new file", async () => {
    const result = await writeFileTool.handler(
      { path: "new-file.ts", content: "const z = 3;\n" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.artifacts.length).toBe(1);

    // Verify by reading back
    const read = await readFileTool.handler({ path: "new-file.ts" }, ctx);
    expect(read.output).toContain("const z = 3;");
  });

  it("creates parent directories", async () => {
    const result = await writeFileTool.handler(
      { path: "deep/nested/file.ts", content: "// deep\n" },
      ctx,
    );
    expect(result.success).toBe(true);

    const read = await readFileTool.handler({ path: "deep/nested/file.ts" }, ctx);
    expect(read.output).toContain("// deep");
  });
});

describe("replace_in_file", () => {
  it("replaces text in file", async () => {
    const result = await replaceInFileTool.handler(
      { path: "src/index.ts", search: "const x = 1", replace: "const x = 42" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("1 occurrence");

    const read = await readFileTool.handler({ path: "src/index.ts" }, ctx);
    expect(read.output).toContain("const x = 42");
  });

  it("throws when search string not found", async () => {
    await expect(
      replaceInFileTool.handler(
        { path: "src/index.ts", search: "nonexistent", replace: "x" },
        ctx,
      ),
    ).rejects.toThrow("Search string not found");
  });
});

describe("find_files", () => {
  it("finds files by pattern", async () => {
    const result = await findFilesTool.handler({ pattern: "**/*.ts" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("src/index.ts");
    expect(result.output).toContain("src/utils.ts");
  });

  it("finds files with specific pattern", async () => {
    const result = await findFilesTool.handler({ pattern: "**/*.md" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("README.md");
    expect(result.output).not.toContain(".ts");
  });

  it("returns message when no files match", async () => {
    const result = await findFilesTool.handler({ pattern: "**/*.xyz" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("No files matched");
  });
});

describe("search_files", () => {
  it("searches for text in files", async () => {
    const result = await searchFilesTool.handler(
      { pattern: "export const" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("src/index.ts");
    expect(result.output).toContain("export const x");
  });

  it("searches with file pattern filter", async () => {
    const result = await searchFilesTool.handler(
      { pattern: "Test", file_pattern: "**/*.md" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("README.md");
    expect(result.output).not.toContain(".ts");
  });

  it("returns message when no matches", async () => {
    const result = await searchFilesTool.handler(
      { pattern: "zzz_nonexistent_zzz" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("No matches found");
  });
});
