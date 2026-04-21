import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { it, expect, beforeEach, afterEach } from "vitest";

import { SkillLoader } from "./loader.js";

const TEST_DIR = "/tmp/devagent-skill-loader-test";
const ORIGINAL_HOME = process.env["HOME"];

function createSkillDir(
  base: string,
  name: string,
  frontmatter: string,
  body: string,
): void {
  const dir = join(base, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "SKILL.md"),
    `---\n${frontmatter}\n---\n${body}`,
    "utf-8",
  );
}
let loader: SkillLoader;

beforeEach(() => {
  loader = new SkillLoader();
  mkdirSync(join(TEST_DIR, ".agents", "skills"), { recursive: true });
  process.env["HOME"] = join(TEST_DIR, "home");
});

afterEach(() => {
  if (ORIGINAL_HOME === undefined) {
    delete process.env["HOME"];
  } else {
    process.env["HOME"] = ORIGINAL_HOME;
  }
  rmSync(TEST_DIR, { recursive: true, force: true });
});

it("discovers skills from SKILL.md in subdirectories", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(skillsDir, "review", "name: review\ndescription: Code review", "Review the code");
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(1);
  expect(result[0]!.name).toBe("review");
  expect(result[0]!.description).toBe("Code review");
  expect(result[0]!.source).toBe("project");
});

it("scans multiple discovery paths in priority order", () => {
  const globalDir = join(TEST_DIR, "global-skills");
  mkdirSync(globalDir, { recursive: true });
  createSkillDir(globalDir, "shared", "name: shared\ndescription: Global version", "Global instructions");

  const projectDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(projectDir, "shared", "name: shared\ndescription: Project version", "Project instructions");

  const result = loader.discover({
    repoRoot: TEST_DIR,
    globalPaths: [globalDir],
  });
  expect(result).toHaveLength(1);
  expect(result[0]!.description).toBe("Project version");
  expect(result[0]!.source).toBe("project");
});

it("validates skill name matches directory name", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(skillsDir, "my-tool", "name: wrong-name\ndescription: Mismatch", "Body");
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(0);
});

it("validates skill name format", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(skillsDir, "Invalid_Name", "name: Invalid_Name\ndescription: Bad format", "Body");
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(0);
});

it("skips directories without SKILL.md", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  mkdirSync(join(skillsDir, "empty-dir"), { recursive: true });
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(0);
});

it("handles missing skills directory gracefully", () => {
  rmSync(join(TEST_DIR, ".agents"), { recursive: true, force: true });
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(0);
});

it("skips skills with missing required frontmatter fields", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(skillsDir, "no-desc", "name: no-desc", "Body");
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(0);
});

it("parses optional frontmatter fields", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(
    skillsDir,
    "full-meta",
    "name: full-meta\ndescription: Full metadata\nlicense: MIT\ncompatibility: devagent, claude-code",
    "Body",
  );
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(1);
  expect(result[0]!.license).toBe("MIT");
});

it("uses source_repo as the support root for backed skills", () => {
  const globalDir = join(TEST_DIR, "global-skills");
  const supportRoot = join(TEST_DIR, "skill-source");
  mkdirSync(supportRoot, { recursive: true });
  createSkillDir(
    globalDir,
    "modernize-code",
    "name: modernize-code\ndescription: Modernize code",
    "Instructions",
  );
  writeFileSync(
    join(globalDir, "modernize-code", ".devagent-skill-source.json"),
    JSON.stringify({
      source_repo: supportRoot,
      source_dir: join(supportRoot, "skills", "modernize-code"),
    }),
    "utf-8",
  );

  const result = loader.discover({
    repoRoot: TEST_DIR,
    globalPaths: [globalDir],
  });

  expect(result).toHaveLength(1);
  expect(result[0]!.dirPath).toBe(join(globalDir, "modernize-code"));
  expect(result[0]!.supportRootPath).toBe(supportRoot);
  expect(result[0]!.sourceRepoPath).toBe(supportRoot);
  expect(result[0]!.sourceSkillDirPath).toBe(
    join(supportRoot, "skills", "modernize-code"),
  );
});

it("defaults supportRootPath to dirPath when no backing metadata exists", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(
    skillsDir,
    "normal-skill",
    "name: normal-skill\ndescription: Normal skill",
    "Instructions",
  );

  const result = loader.discover({ repoRoot: TEST_DIR });

  expect(result).toHaveLength(1);
  expect(result[0]!.supportRootPath).toBe(join(skillsDir, "normal-skill"));
  expect(result[0]!.sourceRepoPath).toBeUndefined();
  expect(result[0]!.sourceSkillDirPath).toBeUndefined();
});

it("scans .agents/skills/ for cross-tool compatibility", () => {
  mkdirSync(join(TEST_DIR, ".agents", "skills"), { recursive: true });
  createSkillDir(
    join(TEST_DIR, ".agents", "skills"),
    "cross-tool",
    "name: cross-tool\ndescription: Cross-tool skill",
    "Instructions",
  );
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(1);
  expect(result[0]!.name).toBe("cross-tool");
  expect(result[0]!.source).toBe("project");
});

it("ignores unsupported project-local skill directories", () => {
  createSkillDir(
    join(TEST_DIR, ".claude", "skills"),
    "claude-skill",
    "name: claude-skill\ndescription: Claude-compatible skill",
    "Instructions",
  );
  createSkillDir(
    join(TEST_DIR, ".github", "skills"),
    "github-skill",
    "name: github-skill\ndescription: GitHub-local skill",
    "Instructions",
  );
  createSkillDir(
    join(TEST_DIR, ".devagent", "skills"),
    "devagent-skill",
    "name: devagent-skill\ndescription: Legacy DevAgent-local skill",
    "Instructions",
  );
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(0);
});

it("discovers global skills from ~/.agents/skills by default", () => {
  const globalSkillsDir = join(TEST_DIR, "home", ".agents", "skills");
  createSkillDir(
    globalSkillsDir,
    "agents-global",
    "name: agents-global\ndescription: Global skill",
    "Instructions",
  );

  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result.some((skill) => skill.name === "agents-global")).toBe(true);
});

it("ignores unsupported global skill directories", () => {
  createSkillDir(
    join(TEST_DIR, "home", ".codex", "skills"),
    "codex-global",
    "name: codex-global\ndescription: Codex global skill",
    "Instructions",
  );
  createSkillDir(
    join(TEST_DIR, "home", ".claude", "skills"),
    "claude-global",
    "name: claude-global\ndescription: Claude global skill",
    "Instructions",
  );
  createSkillDir(
    join(TEST_DIR, "home", ".config", "devagent", "skills"),
    "config-global",
    "name: config-global\ndescription: Config global skill",
    "Instructions",
  );

  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(0);
});

it("strips quotes from frontmatter values", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(skillsDir, "quoted", 'name: "quoted"\ndescription: \'hello world\'', "Body");
  const result = loader.discover({ repoRoot: TEST_DIR });
  expect(result).toHaveLength(1);
  expect(result[0]!.name).toBe("quoted");
  expect(result[0]!.description).toBe("hello world");
});

it("loads full skill content from SKILL.md", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(skillsDir, "loadable", "name: loadable\ndescription: Loadable skill", "These are the instructions.");
  const metadata = loader.discover({ repoRoot: TEST_DIR });
  expect(metadata).toHaveLength(1);
  const skill = loader.loadSkillContent(metadata[0]!);
  expect(skill.instructions).toBe("These are the instructions.");
  expect(skill.hasScripts).toBe(false);
  expect(skill.hasReferences).toBe(false);
  expect(skill.hasAssets).toBe(false);
});

it("detects supporting directories when loading", () => {
  const skillsDir = join(TEST_DIR, ".agents", "skills");
  createSkillDir(skillsDir, "scripted", "name: scripted\ndescription: Has scripts", "Body");
  mkdirSync(join(skillsDir, "scripted", "scripts"), { recursive: true });
  const metadata = loader.discover({ repoRoot: TEST_DIR });
  const skill = loader.loadSkillContent(metadata[0]!);
  expect(skill.hasScripts).toBe(true);
  expect(skill.hasReferences).toBe(false);
});

it("detects support directories in a distinct backing support root", () => {
  const globalDir = join(TEST_DIR, "global-skills");
  const supportRoot = join(TEST_DIR, "skill-source");
  createSkillDir(
    globalDir,
    "backed-skill",
    "name: backed-skill\ndescription: Backed skill",
    "Instructions",
  );
  mkdirSync(join(supportRoot, "references"), { recursive: true });
  writeFileSync(
    join(globalDir, "backed-skill", ".devagent-skill-source.json"),
    JSON.stringify({ source_repo: supportRoot }),
    "utf-8",
  );

  const metadata = loader.discover({ repoRoot: TEST_DIR, globalPaths: [globalDir] });
  const skill = loader.loadSkillContent(metadata[0]!);
  expect(skill.hasScripts).toBe(false);
  expect(skill.hasReferences).toBe(true);
  expect(skill.hasAssets).toBe(false);
});

it("falls back to dirPath when backing metadata is malformed", () => {
  const globalDir = join(TEST_DIR, "global-skills");
  createSkillDir(
    globalDir,
    "broken-backed-skill",
    "name: broken-backed-skill\ndescription: Broken backing metadata",
    "Instructions",
  );
  writeFileSync(
    join(globalDir, "broken-backed-skill", ".devagent-skill-source.json"),
    "{not-json",
    "utf-8",
  );

  const result = loader.discover({
    repoRoot: TEST_DIR,
    globalPaths: [globalDir],
  });

  expect(result).toHaveLength(1);
  expect(result[0]!.supportRootPath).toBe(join(globalDir, "broken-backed-skill"));
  expect(result[0]!.sourceRepoPath).toBeUndefined();
});
