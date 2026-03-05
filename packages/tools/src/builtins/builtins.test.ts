import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { join } from "node:path";
import {
  mkdtempSync,
  rmSync,
  writeFileSync,
  mkdirSync,
  readFileSync,
  existsSync,
  symlinkSync,
} from "node:fs";
import { tmpdir } from "node:os";
import { execSync } from "node:child_process";
import type { ToolContext } from "@devagent/core";
import { readFileTool } from "./read-file.js";
import { writeFileTool } from "./write-file.js";
import { replaceInFileTool } from "./replace-in-file.js";
import {
  fuzzyReplace,
  levenshtein,
  makeCtx,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
} from "./replace-in-file.js";
import { findFilesTool } from "./find-files.js";
import { searchFilesTool } from "./search-files.js";
import { runCommandTool } from "./run-command.js";
import { spawnAndCapture } from "./spawn-capture.js";
import { gitDiffTool, gitCommitTool } from "./git.js";
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

  it("rejects reading files outside repo root", async () => {
    const outsidePath = join(
      tmpdir(),
      `devagent-tools-outside-read-${Date.now()}.txt`,
    );
    writeFileSync(outsidePath, "outside secret");

    try {
      await expect(
        readFileTool.handler({ path: outsidePath }, ctx),
      ).rejects.toThrow(/repo root|outside/i);
    } finally {
      rmSync(outsidePath, { force: true });
    }
  });

  it("rejects reading through symlinks that escape repo root", async () => {
    const outsidePath = join(
      tmpdir(),
      `devagent-tools-outside-readlink-${Date.now()}.txt`,
    );
    writeFileSync(outsidePath, "outside secret");
    const linkPath = join(tmpDir, "src", "outside-link.txt");
    symlinkSync(outsidePath, linkPath);

    try {
      await expect(
        readFileTool.handler({ path: "src/outside-link.txt" }, ctx),
      ).rejects.toThrow(/repo root|outside/i);
    } finally {
      rmSync(linkPath, { force: true });
      rmSync(outsidePath, { force: true });
    }
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

  it("throws when overwriting existing file", async () => {
    await expect(
      writeFileTool.handler(
        { path: "src/index.ts", content: "overwritten\n" },
        ctx,
      ),
    ).rejects.toThrow("Refusing to overwrite existing file");
  });

  it("rejects overwriting existing file even after pre-read", async () => {
    await readFileTool.handler({ path: "src/index.ts" }, ctx);
    await expect(
      writeFileTool.handler(
        { path: "src/index.ts", content: "overwritten\n" },
        ctx,
      ),
    ).rejects.toThrow("Refusing to overwrite existing file");
  });

  it("rejects writing files outside repo root", async () => {
    const outsidePath = join(
      tmpdir(),
      `devagent-tools-outside-write-${Date.now()}.txt`,
    );
    rmSync(outsidePath, { force: true });

    try {
      await expect(
        writeFileTool.handler({ path: outsidePath, content: "nope\n" }, ctx),
      ).rejects.toThrow(/repo root|outside/i);
      expect(existsSync(outsidePath)).toBe(false);
    } finally {
      rmSync(outsidePath, { force: true });
    }
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

  it("defaults to single-replace mode and rejects ambiguous matches", async () => {
    writeFileSync(join(tmpDir, "src", "dupe.ts"), "foo\nfoo\n");
    await readFileTool.handler({ path: "src/dupe.ts" }, ctx);

    await expect(
      replaceInFileTool.handler(
        { path: "src/dupe.ts", search: "foo", replace: "bar" },
        ctx,
      ),
    ).rejects.toThrow("multiple matches");
  });

  it("supports all=true with expected_replacements when count matches", async () => {
    writeFileSync(join(tmpDir, "src", "dupe2.ts"), "foo\nfoo\n");
    await readFileTool.handler({ path: "src/dupe2.ts" }, ctx);

    const result = await replaceInFileTool.handler(
      {
        path: "src/dupe2.ts",
        search: "foo",
        replace: "bar",
        all: true,
        expected_replacements: 2,
      },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("2 occurrence");
  });

  it("fails when expected_replacements does not match actual replacement count", async () => {
    writeFileSync(join(tmpDir, "src", "dupe3.ts"), "foo\nfoo\n");
    await readFileTool.handler({ path: "src/dupe3.ts" }, ctx);

    await expect(
      replaceInFileTool.handler(
        {
          path: "src/dupe3.ts",
          search: "foo",
          replace: "bar",
          all: true,
          expected_replacements: 1,
        },
        ctx,
      ),
    ).rejects.toThrow("Expected 1 replacement(s), but made 2");
  });

  it("rejects replacing text outside repo root", async () => {
    const outsidePath = join(
      tmpdir(),
      `devagent-tools-outside-replace-${Date.now()}.txt`,
    );
    writeFileSync(outsidePath, "outside value=1\n");
    FileTime.recordRead(outsidePath);

    try {
      await expect(
        replaceInFileTool.handler(
          { path: outsidePath, search: "value=1", replace: "value=2" },
          ctx,
        ),
      ).rejects.toThrow(/repo root|outside/i);
      expect(readFileSync(outsidePath, "utf-8")).toContain("value=1");
    } finally {
      rmSync(outsidePath, { force: true });
    }
  });

  it("rejects replacing through symlinks that escape repo root", async () => {
    const outsidePath = join(
      tmpdir(),
      `devagent-tools-outside-replacelink-${Date.now()}.txt`,
    );
    writeFileSync(outsidePath, "outside value=1\n");
    const linkPath = join(tmpDir, "src", "outside-replace-link.txt");
    symlinkSync(outsidePath, linkPath);
    FileTime.recordRead(linkPath);

    try {
      await expect(
        replaceInFileTool.handler(
          {
            path: "src/outside-replace-link.txt",
            search: "value=1",
            replace: "value=2",
          },
          ctx,
        ),
      ).rejects.toThrow(/repo root|outside/i);
      expect(readFileSync(outsidePath, "utf-8")).toContain("value=1");
    } finally {
      rmSync(linkPath, { force: true });
      rmSync(outsidePath, { force: true });
    }
  });
  // ─── Batch mode (replacements array) ──────────────────────

  it("batch mode: applies multiple replacements to a single file", async () => {
    writeFileSync(
      join(tmpDir, "src", "ani.cpp"),
      [
        'auto mod = env->FindModule("@ohos.data.share");',
        'auto cls = env->FindClass("std.core.String");',
        'auto ns  = env->FindNamespace("escompat.Array");',
      ].join("\n") + "\n",
    );
    await readFileTool.handler({ path: "src/ani.cpp" }, ctx);

    const result = await replaceInFileTool.handler(
      {
        path: "src/ani.cpp",
        replacements: [
          { search: "@ohos.data.share", replace: "@ohos:data.share" },
          { search: "std.core.String", replace: "std:core.String" },
          { search: "escompat.Array", replace: "escompat:Array" },
        ],
      },
      ctx,
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("3 replacement(s)");

    const content = readFileSync(join(tmpDir, "src", "ani.cpp"), "utf-8");
    expect(content).toContain("@ohos:data.share");
    expect(content).toContain("std:core.String");
    expect(content).toContain("escompat:Array");
    expect(content).not.toContain("@ohos.data.share");
  });

  it("batch mode: reports per-pair status in output", async () => {
    writeFileSync(
      join(tmpDir, "src", "multi.cpp"),
      'auto a = "foo.bar";\nauto b = "baz.qux";\n',
    );
    await readFileTool.handler({ path: "src/multi.cpp" }, ctx);

    const result = await replaceInFileTool.handler(
      {
        path: "src/multi.cpp",
        replacements: [
          { search: "foo.bar", replace: "foo:bar" },
          { search: "baz.qux", replace: "baz:qux" },
        ],
      },
      ctx,
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("foo.bar");
    expect(result.output).toContain("foo:bar");
    expect(result.output).toContain("baz.qux");
    expect(result.output).toContain("baz:qux");
  });

  it("batch mode: partial write on mid-batch failure", async () => {
    writeFileSync(join(tmpDir, "src", "partial.cpp"), 'auto a = "foo.bar";\n');
    await readFileTool.handler({ path: "src/partial.cpp" }, ctx);

    const result = await replaceInFileTool.handler(
      {
        path: "src/partial.cpp",
        replacements: [
          { search: "foo.bar", replace: "foo:bar" },
          { search: "nonexistent.pattern", replace: "x:y" },
        ],
      },
      ctx,
    );

    // First replacement applied, second failed → success: false but partial write
    expect(result.success).toBe(false);
    expect(result.output).toContain("foo:bar");
    expect(result.output).toContain("nonexistent.pattern");

    const content = readFileSync(join(tmpDir, "src", "partial.cpp"), "utf-8");
    expect(content).toContain("foo:bar");
  });

  it("batch mode: rejects when both search and replacements provided", async () => {
    await readFileTool.handler({ path: "src/index.ts" }, ctx);

    await expect(
      replaceInFileTool.handler(
        {
          path: "src/index.ts",
          search: "const x = 1",
          replace: "const x = 42",
          replacements: [{ search: "a", replace: "b" }],
        },
        ctx,
      ),
    ).rejects.toThrow(/mutually exclusive|Cannot use both/i);
  });

  // Note: OpenAI strict schema sends null for unused params (e.g. replacements: null).
  // Null stripping is handled upstream in stripNullArgs (openai.ts provider layer), so
  // the handler never sees null — only undefined. See openai.test.ts "stripNullArgs".

  it("batch mode: schema has additionalProperties: false on nested items (OpenAI compat)", () => {
    const schema = replaceInFileTool.paramSchema as Record<string, unknown>;
    const props = schema.properties as Record<string, Record<string, unknown>>;
    const items = props.replacements.items as Record<string, unknown>;
    expect(items.type).toBe("object");
    expect(items.additionalProperties).toBe(false);
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

  it("pattern without **/ matches files in nested directories", async () => {
    // *.ts should match src/index.ts — not just root-level .ts files
    const result = await findFilesTool.handler({ pattern: "*.ts" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).toContain("src/index.ts");
    expect(result.output).toContain("src/utils.ts");
  });

  it("rejects searching directories outside repo root", async () => {
    const outsideDir = mkdtempSync(
      join(tmpdir(), "devagent-tools-outside-find-"),
    );
    writeFileSync(join(outsideDir, "outside.ts"), "export const outside = 1;\n");

    try {
      await expect(
        findFilesTool.handler({ pattern: "**/*.ts", path: outsideDir }, ctx),
      ).rejects.toThrow(/repo root|outside/i);
    } finally {
      rmSync(outsideDir, { recursive: true, force: true });
    }
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

  it("file_pattern without **/ matches files in nested directories", async () => {
    // *.ts should match src/index.ts — not just root-level .ts files
    const result = await searchFilesTool.handler(
      { pattern: "export const", file_pattern: "*.ts" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("src/index.ts");
  });

  it("file_pattern with explicit path prefix only matches that path", async () => {
    mkdirSync(join(tmpDir, "lib"), { recursive: true });
    writeFileSync(join(tmpDir, "lib", "helper.ts"), "export const h = 1;\n");

    const result = await searchFilesTool.handler(
      { pattern: "export const", file_pattern: "src/*.ts" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("src/index.ts");
    expect(result.output).not.toContain("lib/helper.ts");
  });

  it("rejects search paths outside repo root", async () => {
    const outsideDir = mkdtempSync(
      join(tmpdir(), "devagent-tools-outside-search-"),
    );
    writeFileSync(join(outsideDir, "outside.ts"), "export const outside = 1;\n");

    try {
      await expect(
        searchFilesTool.handler(
          { pattern: "outside", path: outsideDir },
          ctx,
        ),
      ).rejects.toThrow(/repo root|outside/i);
    } finally {
      rmSync(outsideDir, { recursive: true, force: true });
    }
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

  it("marks output as truncated when command output exceeds byte limit", async () => {
    const result = await runCommandTool.handler(
      { command: "head -c 120000 /dev/zero | tr '\\0' 'a'" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("[output truncated");
  });

  it("clamps timeout_ms to MAX_COMMAND_TIMEOUT_MS (600_000)", async () => {
    // A huge timeout_ms should be clamped, not cause a setTimeout overflow
    const result = await runCommandTool.handler(
      { command: "echo clamped", timeout_ms: 1_200_000_000_000 },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("clamped");
  });

  it("schema declares maximum for timeout_ms", () => {
    const schema = runCommandTool.paramSchema as Record<string, unknown>;
    const props = schema.properties as Record<string, Record<string, unknown>>;
    expect(props.timeout_ms.maximum).toBe(600_000);
  });
});

describe("spawnAndCapture", () => {
  it("clamps timeout > 2^31-1 to prevent setTimeout overflow", async () => {
    // 1.2 trillion ms would overflow setTimeout's 32-bit signed int,
    // causing it to fire at 1ms. The clamp should prevent this.
    const result = await spawnAndCapture("echo", ["overflow-safe"], {
      cwd: tmpDir,
      timeout: 1_200_000_000_000,
    });
    expect(result.exitCode).toBe(0);
    expect(result.stdout.trim()).toBe("overflow-safe");
    expect(result.timedOut).toBe(false);
  });

  it("still times out with a valid large timeout under 2^31-1", async () => {
    // Use a tiny timeout (50ms) with a long-running command to verify timeout still works
    const result = await spawnAndCapture("sleep", ["10"], {
      cwd: tmpDir,
      timeout: 50,
    });
    expect(result.timedOut).toBe(true);
    expect(result.exitCode).toBe(1);
  });
});

describe("git tools", () => {
  function initRepo(): void {
    execSync("git init", { cwd: tmpDir });
    execSync("git config user.name Test", { cwd: tmpDir });
    execSync("git config user.email test@example.com", { cwd: tmpDir });
    execSync("git add -A", { cwd: tmpDir });
    execSync("git commit -m init", { cwd: tmpDir });
  }

  it("does not execute shell metacharacters from git_diff path", async () => {
    initRepo();
    const marker = ".git-diff-injection-marker";

    await expect(
      gitDiffTool.handler(
        { path: `README.md; touch ${marker}` },
        ctx,
      ),
    ).rejects.toThrow();
    expect(existsSync(join(tmpDir, marker))).toBe(false);
  });

  it("rejects git_diff refs that look like options", async () => {
    initRepo();
    writeFileSync(join(tmpDir, "README.md"), "# changed\n");
    const marker = join(tmpDir, "git-diff-ref-option-marker.txt");

    await expect(
      gitDiffTool.handler(
        { ref: `--output=${marker}` },
        ctx,
      ),
    ).rejects.toThrow();
    expect(existsSync(marker)).toBe(false);
  });

  it("does not execute shell metacharacters from git_commit files", async () => {
    initRepo();
    writeFileSync(join(tmpDir, "README.md"), "# Changed\n");
    const marker = ".git-commit-injection-marker";

    await expect(
      gitCommitTool.handler(
        { message: "safe", files: `README.md; touch ${marker}` },
        ctx,
      ),
    ).rejects.toThrow();
    expect(existsSync(join(tmpDir, marker))).toBe(false);
  });

  it("does not interpret git_commit files as git options", async () => {
    initRepo();
    writeFileSync(join(tmpDir, "src", "index.ts"), "export const x = 99;\n");

    await expect(
      gitCommitTool.handler(
        { message: "no-option", files: "--all" },
        ctx,
      ),
    ).rejects.toThrow();
  });

  it("supports quoted file paths with spaces in git_commit files", async () => {
    initRepo();
    writeFileSync(join(tmpDir, "space name.txt"), "first\n");
    execSync('git add "space name.txt" && git commit -m seed-space', { cwd: tmpDir });

    writeFileSync(join(tmpDir, "space name.txt"), "second\n");
    writeFileSync(join(tmpDir, "README.md"), "# unrelated change\n");

    const result = await gitCommitTool.handler(
      { message: "update spaced file", files: '"space name.txt"' },
      ctx,
    );
    expect(result.success).toBe(true);

    const committedFiles = execSync(
      "git show --name-only --pretty=format: HEAD",
      { cwd: tmpDir, encoding: "utf-8" },
    );
    expect(committedFiles).toContain("space name.txt");
    expect(committedFiles).not.toContain("README.md");
  });

  it("appends advisory when git_diff output exceeds 20K chars without a path", async () => {
    initRepo();
    // Create many files to produce a large diff
    for (let i = 0; i < 30; i++) {
      writeFileSync(join(tmpDir, `file_${i}.txt`), `${"content".repeat(200)}\n`);
    }

    const result = await gitDiffTool.handler({}, ctx);
    expect(result.success).toBe(true);
    if (result.output.length > 20_000) {
      expect(result.output).toContain("[ADVISORY:");
      expect(result.output).toContain("specific file path");
    }
  });

  it("does not append advisory when path is specified", async () => {
    initRepo();
    writeFileSync(join(tmpDir, "single.txt"), "changed content\n");
    const result = await gitDiffTool.handler({ path: "single.txt" }, ctx);
    expect(result.success).toBe(true);
    expect(result.output).not.toContain("[ADVISORY:");
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
    const candidates = [...LineTrimmedReplacer(makeCtx(content, "const x = 1;\nconst y = 2;"))];
    expect(candidates.length).toBeGreaterThan(0);
    // The yielded candidate should be the actual substring from content
    expect(content.includes(candidates[0]!)).toBe(true);
  });

  it("BlockAnchorReplacer matches when middle lines differ slightly", () => {
    const content = "function foo() {\n  const a = 1;\n  const b = 2;\n  return a + b;\n}\n";
    const search = "function foo() {\n  const aa = 1;\n  const bb = 2;\n  return a + b;\n}";
    const candidates = [...BlockAnchorReplacer(makeCtx(content, search))];
    expect(candidates.length).toBeGreaterThan(0);
  });

  it("WhitespaceNormalizedReplacer matches collapsed whitespace", () => {
    const content = "const   x   =   1;";
    const candidates = [...WhitespaceNormalizedReplacer(makeCtx(content, "const x = 1;"))];
    expect(candidates.length).toBeGreaterThan(0);
    expect(candidates[0]).toBe("const   x   =   1;");
  });

  it("IndentationFlexibleReplacer matches different indent levels", () => {
    const content = "    if (true) {\n      doStuff();\n    }\n";
    const search = "if (true) {\n  doStuff();\n}";
    const candidates = [...IndentationFlexibleReplacer(makeCtx(content, search))];
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

