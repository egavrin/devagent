import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { SkillLoader } from "./loader.js";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";

const TEST_DIR = "/tmp/devagent-skill-loader-test";

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

describe("SkillLoader", () => {
  let loader: SkillLoader;

  beforeEach(() => {
    loader = new SkillLoader();
    mkdirSync(join(TEST_DIR, ".devagent", "skills"), { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("discovers skills from SKILL.md in subdirectories", () => {
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
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

    const projectDir = join(TEST_DIR, ".devagent", "skills");
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
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
    createSkillDir(skillsDir, "my-tool", "name: wrong-name\ndescription: Mismatch", "Body");
    const result = loader.discover({ repoRoot: TEST_DIR });
    expect(result).toHaveLength(0);
  });

  it("validates skill name format", () => {
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
    createSkillDir(skillsDir, "Invalid_Name", "name: Invalid_Name\ndescription: Bad format", "Body");
    const result = loader.discover({ repoRoot: TEST_DIR });
    expect(result).toHaveLength(0);
  });

  it("skips directories without SKILL.md", () => {
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
    mkdirSync(join(skillsDir, "empty-dir"), { recursive: true });
    const result = loader.discover({ repoRoot: TEST_DIR });
    expect(result).toHaveLength(0);
  });

  it("handles missing skills directory gracefully", () => {
    rmSync(join(TEST_DIR, ".devagent"), { recursive: true, force: true });
    const result = loader.discover({ repoRoot: TEST_DIR });
    expect(result).toHaveLength(0);
  });

  it("skips skills with missing required frontmatter fields", () => {
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
    createSkillDir(skillsDir, "no-desc", "name: no-desc", "Body");
    const result = loader.discover({ repoRoot: TEST_DIR });
    expect(result).toHaveLength(0);
  });

  it("parses optional frontmatter fields", () => {
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
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

  it("scans .claude/skills/ for claude compatibility", () => {
    mkdirSync(join(TEST_DIR, ".claude", "skills"), { recursive: true });
    createSkillDir(
      join(TEST_DIR, ".claude", "skills"),
      "claude-skill",
      "name: claude-skill\ndescription: Claude-compatible skill",
      "Instructions",
    );
    const result = loader.discover({ repoRoot: TEST_DIR });
    expect(result).toHaveLength(1);
    expect(result[0]!.name).toBe("claude-skill");
    expect(result[0]!.source).toBe("claude-compat");
  });

  it("strips quotes from frontmatter values", () => {
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
    createSkillDir(skillsDir, "quoted", 'name: "quoted"\ndescription: \'hello world\'', "Body");
    const result = loader.discover({ repoRoot: TEST_DIR });
    expect(result).toHaveLength(1);
    expect(result[0]!.name).toBe("quoted");
    expect(result[0]!.description).toBe("hello world");
  });

  it("loads full skill content from SKILL.md", () => {
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
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
    const skillsDir = join(TEST_DIR, ".devagent", "skills");
    createSkillDir(skillsDir, "scripted", "name: scripted\ndescription: Has scripts", "Body");
    mkdirSync(join(skillsDir, "scripted", "scripts"), { recursive: true });
    const metadata = loader.discover({ repoRoot: TEST_DIR });
    const skill = loader.loadSkillContent(metadata[0]!);
    expect(skill.hasScripts).toBe(true);
    expect(skill.hasReferences).toBe(false);
  });
});
