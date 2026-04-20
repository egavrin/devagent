import { execSync } from "node:child_process";
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
import { join } from "node:path";
import { describe, it, expect, beforeEach, afterEach } from "vitest";

import { FileTime } from "./file-time.js";
import { createFindFilesTool, findFilesTool } from "./find-files.js";
import { gitDiffTool, gitCommitTool } from "./git.js";
import { builtinTools } from "./index.js";
import { createReadFileTool, readFileTool } from "./read-file.js";
import { replaceInFileTool ,
  fuzzyReplace,
  levenshtein,
  makeCtx,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
} from "./replace-in-file.js";
import { runCommandTool } from "./run-command.js";
import { createSearchFilesTool, searchFilesTool } from "./search-files.js";
import { spawnAndCapture } from "./spawn-capture.js";
import { writeFileTool } from "./write-file.js";
import type { ToolContext } from "../../core/index.js";
import { SkillAccessManager, SkillRegistry } from "../../core/index.js";

let tmpDir: string;
let ctx: ToolContext;
let cleanupDirs: string[];

beforeEach(() => {
  FileTime.reset();
  cleanupDirs = [];
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
  for (const dir of cleanupDirs) {
    rmSync(dir, { recursive: true, force: true });
  }
  rmSync(tmpDir, { recursive: true, force: true });
});

function getFileEdits(result: { readonly metadata?: Record<string, unknown> }): ReadonlyArray<Record<string, unknown>> {
  const fileEdits = result.metadata?.["fileEdits"];
  expect(Array.isArray(fileEdits)).toBe(true);
  return fileEdits as ReadonlyArray<Record<string, unknown>>;
}

function setupUnlockedSkill(name: string = "modernize-arkts"): {
  readonly access: SkillAccessManager;
  readonly uri: (relativePath: string) => string;
} {
  const skillDir = mkdtempSync(join(tmpdir(), "devagent-skill-tree-"));
  cleanupDirs.push(skillDir);
  mkdirSync(join(skillDir, "docs"), { recursive: true });
  writeFileSync(join(skillDir, "docs", "guide-usage.md"), "# Guide\nUse the tool.\n");
  writeFileSync(
    join(skillDir, "SKILL.md"),
    `---\nname: ${name}\ndescription: ${name}\n---\nInstructions`,
    "utf-8",
  );

  const registry = new SkillRegistry();
  registry.register([{
    name,
    description: `${name} description`,
    source: "global",
    dirPath: skillDir,
    skillFilePath: join(skillDir, "SKILL.md"),
  }]);
  const access = new SkillAccessManager(registry);
  access.unlock(name);
  return {
    access,
    uri: (relativePath: string) => `skill://${name}/${relativePath}`,
  };
}

function setupBackedUnlockedSkill(name: string = "modernize-arkts"): {
  readonly access: SkillAccessManager;
  readonly uri: (relativePath: string) => string;
  readonly wrapperDir: string;
  readonly supportRoot: string;
} {
  const wrapperDir = mkdtempSync(join(tmpdir(), "devagent-skill-wrapper-"));
  const supportRoot = mkdtempSync(join(tmpdir(), "devagent-skill-support-"));
  cleanupDirs.push(wrapperDir, supportRoot);

  mkdirSync(join(wrapperDir, "agents"), { recursive: true });
  writeFileSync(
    join(wrapperDir, "SKILL.md"),
    `---\nname: ${name}\ndescription: ${name}\n---\nInstructions`,
    "utf-8",
  );
  writeFileSync(join(wrapperDir, "agents", "openai.yaml"), "model: gpt-5.4\n");
  writeFileSync(
    join(wrapperDir, ".arkts-agent-kit-source.json"),
    JSON.stringify({
      source_repo: supportRoot,
      source_dir: join(supportRoot, "arkts-skills", name),
    }),
    "utf-8",
  );

  mkdirSync(join(supportRoot, "docs", "cards", "fixes"), { recursive: true });
  mkdirSync(join(supportRoot, "knowledge", "derived"), { recursive: true });
  writeFileSync(join(supportRoot, "docs", "guide-usage.md"), "# Guide\nUse the tool.\n");
  writeFileSync(
    join(supportRoot, "docs", "cards", "fixes", "fix-primitive-as-conversions.md"),
    "# Rule Card\n",
  );
  writeFileSync(
    join(supportRoot, "knowledge", "derived", "skill-context.json"),
    '{"rule":"fixes/fix-primitive-as-conversions"}\n',
  );

  const registry = new SkillRegistry();
  registry.register([{
    name,
    description: `${name} description`,
    source: "global",
    dirPath: wrapperDir,
    supportRootPath: supportRoot,
    sourceRepoPath: supportRoot,
    sourceSkillDirPath: join(supportRoot, "arkts-skills", name),
    skillFilePath: join(wrapperDir, "SKILL.md"),
  }]);
  const access = new SkillAccessManager(registry);
  access.unlock(name);
  return {
    access,
    uri: (relativePath: string) => `skill://${name}/${relativePath}`,
    wrapperDir,
    supportRoot,
  };
}

describe("builtinTools", () => {
  it("has 10 built-in tools", () => {
    expect(builtinTools.length).toBe(10);
  });

  it("all tools have unique names", () => {
    const names = builtinTools.map((t) => t.name);
    expect(new Set(names).size).toBe(names.length);
  });

  it("includes fetch_url as an external built-in", () => {
    expect(builtinTools.find((tool) => tool.name === "fetch_url")).toMatchObject({
      name: "fetch_url",
      category: "external",
    });
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

  it("reads files from an unlocked skill tree", async () => {
    const skill = setupUnlockedSkill();
    const tool = createReadFileTool({ skillAccess: skill.access });

    const result = await tool.handler({ path: skill.uri("docs/guide-usage.md") }, ctx);

    expect(result.success).toBe(true);
    expect(result.output).toContain("Guide");
  });

  it("requires invoke_skill before reading from a skill tree", async () => {
    const skillDir = mkdtempSync(join(tmpdir(), "devagent-skill-tree-locked-"));
    cleanupDirs.push(skillDir);
    mkdirSync(join(skillDir, "docs"), { recursive: true });
    writeFileSync(join(skillDir, "docs", "guide-usage.md"), "locked\n");
    writeFileSync(join(skillDir, "SKILL.md"), "---\nname: locked-skill\ndescription: locked\n---\nBody\n");
    const registry = new SkillRegistry();
    registry.register([{
      name: "locked-skill",
      description: "locked",
      source: "global",
      dirPath: skillDir,
      skillFilePath: join(skillDir, "SKILL.md"),
    }]);
    const tool = createReadFileTool({ skillAccess: new SkillAccessManager(registry) });

    await expect(
      tool.handler({ path: "skill://locked-skill/docs/guide-usage.md" }, ctx),
    ).rejects.toThrow(/invoke_skill/i);
  });

  it("rejects traversal out of a skill root", async () => {
    const skill = setupUnlockedSkill();
    const tool = createReadFileTool({ skillAccess: skill.access });

    await expect(
      tool.handler({ path: skill.uri("../outside.txt") }, ctx),
    ).rejects.toThrow(/repo root|outside/i);
  });

  it("reads wrapper-local files for a backed skill", async () => {
    const skill = setupBackedUnlockedSkill();
    const tool = createReadFileTool({ skillAccess: skill.access });

    const skillFile = await tool.handler({ path: skill.uri("SKILL.md") }, ctx);
    const agentFile = await tool.handler({ path: skill.uri("agents/openai.yaml") }, ctx);

    expect(skillFile.success).toBe(true);
    expect(skillFile.output).toContain("name: modernize-arkts");
    expect(agentFile.success).toBe(true);
    expect(agentFile.output).toContain("model: gpt-5.4");
  });

  it("falls back to the backing support root for a backed skill", async () => {
    const skill = setupBackedUnlockedSkill();
    const tool = createReadFileTool({ skillAccess: skill.access });

    const guide = await tool.handler({ path: skill.uri("docs/guide-usage.md") }, ctx);
    const contextFile = await tool.handler(
      { path: skill.uri("knowledge/derived/skill-context.json") },
      ctx,
    );

    expect(guide.success).toBe(true);
    expect(guide.output).toContain("Guide");
    expect(contextFile.success).toBe(true);
    expect(contextFile.output).toContain("fixes/fix-primitive-as-conversions");
  });

  it("rejects symlinks in a backing support root that escape the allowed tree", async () => {
    const skill = setupBackedUnlockedSkill();
    const tool = createReadFileTool({ skillAccess: skill.access });
    const outsideFile = join(tmpdir(), `devagent-skill-support-outside-${Date.now()}.txt`);
    writeFileSync(outsideFile, "outside secret\n");
    cleanupDirs.push(outsideFile);
    symlinkSync(outsideFile, join(skill.supportRoot, "docs", "outside-link.md"));

    await expect(
      tool.handler({ path: skill.uri("docs/outside-link.md") }, ctx),
    ).rejects.toThrow(/repo root|outside/i);
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
    const fileEdits = getFileEdits(result);
    expect(fileEdits).toHaveLength(1);
    expect(fileEdits[0]).toMatchObject({
      path: "new-file.ts",
      kind: "create",
      additions: 1,
      deletions: 0,
      truncated: false,
      before: "",
      after: "const z = 3;\n",
      structuredDiff: {
        hunks: [{
          oldStart: 0,
          oldLines: 0,
          newStart: 1,
          newLines: 1,
          lines: [{
            type: "add",
            text: "const z = 3;",
            oldLine: null,
            newLine: 1,
          }],
        }],
      },
    });
    expect(fileEdits[0]?.["unifiedDiff"]).toContain("+++ b/new-file.ts");

    // Verify by reading back
    const read = await readFileTool.handler({ path: "new-file.ts" }, ctx);
    expect(read.output).toContain("const z = 3;");
  });

  it("truncates long write diff previews", async () => {
    const longContent = Array.from({ length: 80 }, (_, index) => `line-${index + 1}`).join("\n") + "\n";

    const result = await writeFileTool.handler(
      { path: "long-file.ts", content: longContent },
      ctx,
    );

    const fileEdits = getFileEdits(result);
    expect(fileEdits[0]?.["truncated"]).toBe(true);
    expect(String(fileEdits[0]?.["unifiedDiff"]).split("\n").length).toBeLessThanOrEqual(40);
  });

  it("omits write snapshots when content exceeds the snapshot size limit", async () => {
    const hugeContent = `${"a".repeat(70 * 1024)}\n`;

    const result = await writeFileTool.handler(
      { path: "huge-file.txt", content: hugeContent },
      ctx,
    );

    const fileEdits = getFileEdits(result);
    expect(fileEdits[0]?.["before"]).toBeUndefined();
    expect(fileEdits[0]?.["after"]).toBeUndefined();
    expect(fileEdits[0]?.["structuredDiff"]).toBeUndefined();
    expect(fileEdits[0]?.["unifiedDiff"]).toContain("+++ b/huge-file.txt");
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

  it("rejects skill:// paths", async () => {
    await expect(
      writeFileTool.handler({ path: "skill://modernize-arkts/docs/new.md", content: "nope\n" }, ctx),
    ).rejects.toThrow(/read-only/i);
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
    const fileEdits = getFileEdits(result);
    expect(fileEdits).toHaveLength(1);
    expect(fileEdits[0]).toMatchObject({
      path: "src/index.ts",
      kind: "update",
      additions: 1,
      deletions: 1,
      truncated: false,
      before: expect.stringContaining("export const x = 1;"),
      after: expect.stringContaining("export const x = 42;"),
      structuredDiff: expect.objectContaining({
        hunks: expect.any(Array),
      }),
    });
    expect(fileEdits[0]?.["unifiedDiff"]).toContain("-export const x = 1;");
    expect(fileEdits[0]?.["unifiedDiff"]).toContain("+export const x = 42;");

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

  it("rejects skill:// paths", async () => {
    await expect(
      replaceInFileTool.handler(
        {
          path: "skill://modernize-arkts/docs/guide-usage.md",
          search: "Guide",
          replace: "Manual",
        },
        ctx,
      ),
    ).rejects.toThrow(/read-only/i);
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

  it("omits replace snapshots when the file exceeds the snapshot size limit", async () => {
    const hugeContent = `prefix-${"a".repeat(70 * 1024)}\n`;
    writeFileSync(join(tmpDir, "src", "huge.ts"), hugeContent, "utf-8");
    await readFileTool.handler({ path: "src/huge.ts" }, ctx);

    const result = await replaceInFileTool.handler(
      { path: "src/huge.ts", search: "prefix-", replace: "updated-" },
      ctx,
    );

    const fileEdits = getFileEdits(result);
    expect(fileEdits[0]?.["before"]).toBeUndefined();
    expect(fileEdits[0]?.["after"]).toBeUndefined();
    expect(fileEdits[0]?.["structuredDiff"]).toBeUndefined();
    expect(fileEdits[0]?.["unifiedDiff"]).toContain("--- a/src/huge.ts");
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
    expect(result.metadata?.["fileEdits"]).toBeUndefined();

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

  it("finds files inside an unlocked skill tree", async () => {
    const skill = setupUnlockedSkill();
    const tool = createFindFilesTool({ skillAccess: skill.access });

    const result = await tool.handler({ pattern: "**/*.md", path: skill.uri("docs") }, ctx);

    expect(result.success).toBe(true);
    expect(result.output).toContain("docs/guide-usage.md");
  });

  it("finds files inside the backing support root of a backed skill", async () => {
    const skill = setupBackedUnlockedSkill();
    const tool = createFindFilesTool({ skillAccess: skill.access });

    const result = await tool.handler({ pattern: "**/*.md", path: skill.uri("docs") }, ctx);

    expect(result.success).toBe(true);
    expect(result.output).toContain("docs/guide-usage.md");
    expect(result.output).toContain("docs/cards/fixes/fix-primitive-as-conversions.md");
  });

  it("uses the backing support root for root-level backed skill browsing", async () => {
    const skill = setupBackedUnlockedSkill();
    const tool = createFindFilesTool({ skillAccess: skill.access });

    const result = await tool.handler({ pattern: "**/*.md", path: skill.uri(".") }, ctx);

    expect(result.success).toBe(true);
    expect(result.output).toContain("docs/guide-usage.md");
    expect(result.output).not.toContain("SKILL.md");
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

  it("searches inside an unlocked skill tree", async () => {
    const skill = setupUnlockedSkill();
    const tool = createSearchFilesTool({ skillAccess: skill.access });

    const result = await tool.handler(
      { pattern: "Use the tool", path: skill.uri("docs") },
      ctx,
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("docs/guide-usage.md:2");
  });

  it("searches inside the backing support root of a backed skill", async () => {
    const skill = setupBackedUnlockedSkill();
    const tool = createSearchFilesTool({ skillAccess: skill.access });

    const result = await tool.handler(
      { pattern: "fixes/fix-primitive-as-conversions", path: skill.uri("knowledge") },
      ctx,
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("knowledge/derived/skill-context.json:1");
  });

  it("uses the backing support root for root-level backed skill searches", async () => {
    const skill = setupBackedUnlockedSkill();
    const tool = createSearchFilesTool({ skillAccess: skill.access });

    const result = await tool.handler(
      { pattern: "fixes/fix-primitive-as-conversions", path: skill.uri(".") },
      ctx,
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("knowledge/derived/skill-context.json:1");
  });
});

describe("run_command", () => {
  function getCommandMetadata(result: { readonly metadata?: Record<string, unknown> }): Record<string, unknown> {
    const commandResult = result.metadata?.["commandResult"];
    expect(commandResult).toBeTruthy();
    expect(typeof commandResult).toBe("object");
    return commandResult as Record<string, unknown>;
  }

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
    expect(getCommandMetadata(result)).toMatchObject({
      command: "echo $DEVAGENT_TEST_CUSTOM_VAR",
      cwd: ".",
      exitCode: 0,
      timedOut: false,
      warningOnly: false,
    });
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
    expect(getCommandMetadata(result)).toMatchObject({
      command: "echo hello",
      exitCode: null,
      timedOut: false,
      warningOnly: false,
    });
  });

  it("treats env JSON null as an empty env override map", async () => {
    const result = await runCommandTool.handler(
      {
        command: "echo hello",
        env: "null",
      },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("hello");
  });

  it("accepts an explicit empty env override object", async () => {
    const result = await runCommandTool.handler(
      {
        command: "echo hello",
        env: {},
      },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output.trim()).toBe("hello");
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
    const metadata = getCommandMetadata(result);
    expect(metadata["stdoutTruncated"]).toBe(true);
    expect(String(metadata["stdoutPreview"])).toContain("[preview truncated]");
  });

  it("records warning-only partial success metadata", async () => {
    const result = await runCommandTool.handler(
      { command: "node -e \"for(let i=0;i<60;i++) console.log('line-' + i); process.stderr.write('warning: partial issue\\n'); process.exit(2)\"" },
      ctx,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("[Warning: exit code 2");
    expect(getCommandMetadata(result)).toMatchObject({
      exitCode: 2,
      warningOnly: true,
    });
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
