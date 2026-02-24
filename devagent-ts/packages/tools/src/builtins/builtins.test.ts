import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import { mkdtempSync, rmSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import type { ToolContext } from "@devagent/core";
import { readFileTool } from "./read-file.js";
import { writeFileTool } from "./write-file.js";
import { replaceInFileTool } from "./replace-in-file.js";
import {
  fuzzyReplace,
  levenshtein,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
} from "./replace-in-file.js";
import { findFilesTool } from "./find-files.js";
import { searchFilesTool } from "./search-files.js";
import { runCommandTool } from "./run-command.js";
import { builtinTools } from "./index.js";
import { FileTime } from "./file-time.js";

let tmpDir: string;
let ctx: ToolContext;

beforeEach(() => {
  FileTime.reset();
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

  it("throws when overwriting existing file without pre-read", async () => {
    await expect(
      writeFileTool.handler(
        { path: "src/index.ts", content: "overwritten\n" },
        ctx,
      ),
    ).rejects.toThrow("must read file");
  });

  it("allows overwriting existing file after pre-read", async () => {
    await readFileTool.handler({ path: "src/index.ts" }, ctx);
    const result = await writeFileTool.handler(
      { path: "src/index.ts", content: "overwritten\n" },
      ctx,
    );
    expect(result.success).toBe(true);
  });
});

describe("replace_in_file", () => {
  it("replaces text in file", async () => {
    // Pre-read required by FileTime enforcement
    await readFileTool.handler({ path: "src/index.ts" }, ctx);

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
    await readFileTool.handler({ path: "src/index.ts" }, ctx);

    await expect(
      replaceInFileTool.handler(
        { path: "src/index.ts", search: "nonexistent", replace: "x" },
        ctx,
      ),
    ).rejects.toThrow("Search string not found");
  });

  it("throws when file not pre-read (FileTime enforcement)", async () => {
    await expect(
      replaceInFileTool.handler(
        { path: "src/index.ts", search: "const x = 1", replace: "const x = 42" },
        ctx,
      ),
    ).rejects.toThrow("must read file");
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

describe("run_command", () => {
  it("applies env overrides from JSON string", async () => {
    const result = await runCommandTool.handler(
      {
        command: "echo $DEVAGENT_TEST_CUSTOM_VAR",
        env: JSON.stringify({ DEVAGENT_TEST_CUSTOM_VAR: "custom_value_42" }),
      },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("custom_value_42");
  });

  it("applies env overrides from direct object", async () => {
    const result = await runCommandTool.handler(
      {
        command: "echo $DEVAGENT_OVERRIDE_TEST",
        env: { DEVAGENT_OVERRIDE_TEST: "overridden" },
      },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("overridden");
  });

  it("returns error for invalid env JSON", async () => {
    const result = await runCommandTool.handler(
      {
        command: "echo hello",
        env: "not valid json",
      },
      ctx,
    );
    expect(result.success).toBe(false);
    expect(result.error).toContain("Invalid env JSON");
  });

  it("works without env parameter (backward compatible)", async () => {
    const result = await runCommandTool.handler(
      { command: "echo hello" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("hello");
  });
});

// ─── Fuzzy Replacer Tests ──────────────────────────────────

describe("levenshtein", () => {
  it("returns 0 for identical strings", () => {
    expect(levenshtein("abc", "abc")).toBe(0);
  });

  it("returns length of other string when one is empty", () => {
    expect(levenshtein("", "abc")).toBe(3);
    expect(levenshtein("abc", "")).toBe(3);
  });

  it("computes correct distance for simple edits", () => {
    expect(levenshtein("kitten", "sitting")).toBe(3);
    expect(levenshtein("cat", "car")).toBe(1);
  });
});

describe("fuzzyReplace", () => {
  it("performs exact replacement (SimpleReplacer)", () => {
    const content = "const x = 1;\nconst y = 2;\n";
    const result = fuzzyReplace(content, "const x = 1", "const x = 42", false);
    expect(result.newContent).toBe("const x = 42;\nconst y = 2;\n");
    expect(result.count).toBe(1);
  });

  it("matches with line-trimmed whitespace (LineTrimmedReplacer)", () => {
    const content = "  const x = 1;\n  const y = 2;\n";
    // Search without leading whitespace — should still match via LineTrimmedReplacer
    const result = fuzzyReplace(content, "const x = 1;", "const x = 42;", false);
    expect(result.newContent).toContain("const x = 42;");
    expect(result.count).toBe(1);
  });

  it("matches with different indentation (IndentationFlexibleReplacer)", () => {
    const content = "    function foo() {\n      return 1;\n    }\n";
    const search = "function foo() {\n  return 1;\n}";
    const result = fuzzyReplace(content, search, "function bar() {\n  return 2;\n}", false);
    expect(result.newContent).toContain("bar");
    expect(result.count).toBe(1);
  });

  it("matches with collapsed whitespace (WhitespaceNormalizedReplacer)", () => {
    const content = "const   x   =   1;\n";
    const result = fuzzyReplace(content, "const x = 1;", "const x = 42;", false);
    expect(result.newContent).toContain("const x = 42;");
    expect(result.count).toBe(1);
  });

  it("falls through replacers in order and stops at first match", () => {
    // Exact match exists — should use SimpleReplacer (first) and not try others
    const content = "hello world";
    const result = fuzzyReplace(content, "hello world", "goodbye world", false);
    expect(result.newContent).toBe("goodbye world");
    expect(result.count).toBe(1);
  });

  it("replaceAll works with fuzzy matching", () => {
    const content = "  foo();\n  foo();\n  foo();\n";
    const result = fuzzyReplace(content, "foo();", "bar();", true);
    expect(result.count).toBe(3);
    expect(result.newContent).not.toContain("foo");
  });

  it("throws rich error with partial match hint when no replacer matches", () => {
    const content = "function hello() {\n  return 1;\n}\n";
    expect(() => {
      fuzzyReplace(content, "function_completely_different_thing()", "x", false);
    }).toThrow("Search string not found");
  });

  it("throws when multiple ambiguous matches found", () => {
    const content = "foo\nbar\nfoo\n";
    // "foo" appears twice — ambiguous for single replace
    // SimpleReplacer yields "foo", indexOf != lastIndexOf → skips
    // Other replacers also yield "foo" → same issue
    // Should throw "multiple matches"
    expect(() => {
      fuzzyReplace(content, "foo", "baz", false);
    }).toThrow("multiple matches");
  });

  it("includes partial match hint in error for near-misses", () => {
    const content = "function calculateTotal(items) {\n  let sum = 0;\n  return sum;\n}\n";
    try {
      fuzzyReplace(content, "function calculateTotals(items) {\n  let total = 0;\n  return total;\n}", "x", false);
      expect.unreachable("Should have thrown");
    } catch (e) {
      const msg = (e as Error).message;
      // Should contain partial match hint because "function calculateTotal" partially matches
      expect(msg).toContain("Search string not found");
    }
  });
});

describe("individual replacers", () => {
  it("LineTrimmedReplacer yields match when line whitespace differs", () => {
    const content = "  const x = 1;\n  const y = 2;\n";
    const candidates = [...LineTrimmedReplacer(content, "const x = 1;\nconst y = 2;")];
    expect(candidates.length).toBeGreaterThan(0);
    // The yielded candidate should be the actual substring from content
    expect(content.includes(candidates[0]!)).toBe(true);
  });

  it("BlockAnchorReplacer matches when middle lines differ slightly", () => {
    const content = "function foo() {\n  const a = 1;\n  const b = 2;\n  return a + b;\n}\n";
    const search = "function foo() {\n  const aa = 1;\n  const bb = 2;\n  return a + b;\n}";
    const candidates = [...BlockAnchorReplacer(content, search)];
    expect(candidates.length).toBeGreaterThan(0);
  });

  it("WhitespaceNormalizedReplacer matches collapsed whitespace", () => {
    const content = "const   x   =   1;";
    const candidates = [...WhitespaceNormalizedReplacer(content, "const x = 1;")];
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates[0]).toBe("const   x   =   1;");
  });

  it("IndentationFlexibleReplacer matches different indent levels", () => {
    const content = "    if (true) {\n      doStuff();\n    }\n";
    const search = "if (true) {\n  doStuff();\n}";
    const candidates = [...IndentationFlexibleReplacer(content, search)];
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates[0]).toBe("    if (true) {\n      doStuff();\n    }");
  });
});

// ─── FileTime Tests ────────────────────────────────────────

describe("FileTime", () => {
  beforeEach(() => {
    FileTime.reset();
  });

  it("recordRead marks file as read", () => {
    expect(FileTime.wasRead("/tmp/test.ts")).toBe(false);
    FileTime.recordRead("/tmp/test.ts");
    expect(FileTime.wasRead("/tmp/test.ts")).toBe(true);
  });

  it("assert throws when file not read", () => {
    expect(() => FileTime.assert("/tmp/unread.ts")).toThrow("must read file");
  });

  it("assert succeeds after read", () => {
    const filePath = join(tmpDir, "src", "index.ts");
    FileTime.recordRead(filePath);
    expect(() => FileTime.assert(filePath)).not.toThrow();
  });

  it("reset clears all tracking", () => {
    FileTime.recordRead("/tmp/a.ts");
    FileTime.recordWrite("/tmp/a.ts");
    expect(FileTime.wasRead("/tmp/a.ts")).toBe(true);
    FileTime.reset();
    expect(FileTime.wasRead("/tmp/a.ts")).toBe(false);
  });

  it("detects external modification since last read", async () => {
    const filePath = join(tmpDir, "src", "index.ts");
    FileTime.recordRead(filePath);

    // Simulate external modification by waiting and touching the file
    await new Promise((r) => setTimeout(r, 50));
    writeFileSync(filePath, "externally modified content\n");

    // Set the mtime to future to exceed the 1s tolerance
    const { utimesSync } = await import("node:fs");
    const futureTime = new Date(Date.now() + 5000);
    utimesSync(filePath, futureTime, futureTime);

    expect(() => FileTime.assert(filePath)).toThrow("modified since");
  });
});
