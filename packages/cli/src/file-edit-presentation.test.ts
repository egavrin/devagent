import { afterEach, describe, expect, it } from "vitest";

const originalNoColor = process.env["NO_COLOR"];

afterEach(() => {
  if (originalNoColor === undefined) {
    delete process.env["NO_COLOR"];
  } else {
    process.env["NO_COLOR"] = originalNoColor;
  }
});

describe("buildHighlightedFileEdit", () => {
  it("detects language from file path and adds ANSI-highlighted code", async () => {
    delete process.env["NO_COLOR"];
    const { buildHighlightedFileEdit } = await import("./file-edit-presentation.js");
    const fileEdit = {
      path: "src/example.ts",
      kind: "create" as const,
      additions: 1,
      deletions: 0,
      unifiedDiff: "--- /dev/null\n+++ b/src/example.ts\n@@ -0,0 +1,1 @@\n+const value = 1;",
      truncated: false,
      after: "const value = 1;\n",
      structuredDiff: {
        hunks: [{
          oldStart: 0,
          oldLines: 0,
          newStart: 1,
          newLines: 1,
          lines: [{ type: "add" as const, text: "const value = 1;", oldLine: null, newLine: 1 }],
        }],
      },
    };

    const highlighted = buildHighlightedFileEdit(fileEdit);
    const line = highlighted.hunks[0]?.lines[0];
    expect(line?.syntaxHighlighted).toBe(true);
    expect(line?.renderedText).toContain("\x1b[");
    expect(line?.renderedText).toContain("const");
  });

  it("falls back to plain text when no language can be detected", async () => {
    delete process.env["NO_COLOR"];
    const { buildHighlightedFileEdit } = await import("./file-edit-presentation.js");
    const fileEdit = {
      path: "notes.txt",
      kind: "update" as const,
      additions: 1,
      deletions: 0,
      unifiedDiff: "@@ -1,1 +1,2 @@\n line\n+tail",
      truncated: false,
      structuredDiff: {
        hunks: [{
          oldStart: 1,
          oldLines: 1,
          newStart: 1,
          newLines: 2,
          lines: [{ type: "add" as const, text: "tail", oldLine: null, newLine: 2 }],
        }],
      },
    };

    const highlighted = buildHighlightedFileEdit(fileEdit);
    const line = highlighted.hunks[0]?.lines[0];
    expect(line?.syntaxHighlighted).toBe(false);
    expect(line?.renderedText).toBe("tail");
  });

  it("caches highlighted output by file object and width", async () => {
    delete process.env["NO_COLOR"];
    const { buildHighlightedFileEdit } = await import("./file-edit-presentation.js");
    const fileEdit = {
      path: "src/cache.ts",
      kind: "create" as const,
      additions: 1,
      deletions: 0,
      unifiedDiff: "--- /dev/null\n+++ b/src/cache.ts\n@@ -0,0 +1,1 @@\n+const cache = true;",
      truncated: false,
      after: "const cache = true;\n",
      structuredDiff: {
        hunks: [{
          oldStart: 0,
          oldLines: 0,
          newStart: 1,
          newLines: 1,
          lines: [{ type: "add" as const, text: "const cache = true;", oldLine: null, newLine: 1 }],
        }],
      },
    };

    const first = buildHighlightedFileEdit(fileEdit, { bodyWidth: 30 });
    const second = buildHighlightedFileEdit(fileEdit, { bodyWidth: 30 });
    const third = buildHighlightedFileEdit(fileEdit, { bodyWidth: 10 });

    expect(second).toBe(first);
    expect(third).not.toBe(first);
  });
});
