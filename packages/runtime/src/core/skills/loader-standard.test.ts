import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { SkillLoader } from "./loader.js";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";

const TEST_DIR = "/tmp/devagent-skill-loader-standard-test";

function writeSkillMd(
  skillsDir: string,
  name: string,
  description: string,
  body = `Instructions for ${name}`,
): void {
  const dir = join(skillsDir, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "SKILL.md"),
    `---\nname: ${name}\ndescription: ${description}\n---\n${body}`,
    "utf-8",
  );
}

describe("SkillLoader — .agents/skills/ discovery", () => {
  let loader: SkillLoader;

  beforeEach(() => {
    loader = new SkillLoader();
    mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("discovers skills from .agents/skills/", () => {
    const agentsSkillsDir = join(TEST_DIR, ".agents", "skills");
    writeSkillMd(agentsSkillsDir, "my-skill", "A test skill");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(1);
    expect(skills[0].name).toBe("my-skill");
    expect(skills[0].description).toBe("A test skill");
    expect(skills[0].source).toBe("project");
  });

  it("parses frontmatter name and description correctly", () => {
    const agentsSkillsDir = join(TEST_DIR, ".agents", "skills");
    writeSkillMd(agentsSkillsDir, "code-review", "Automated code review helper");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(1);
    expect(skills[0].name).toBe("code-review");
    expect(skills[0].description).toBe("Automated code review helper");
    expect(skills[0].skillFilePath).toBe(
      join(agentsSkillsDir, "code-review", "SKILL.md"),
    );
  });

  it(".agents/skills/ overrides global paths for the same skill name", () => {
    const globalSkillsDir = join(TEST_DIR, "global-skills");
    const agentsSkillsDir = join(TEST_DIR, ".agents", "skills");
    writeSkillMd(globalSkillsDir, "shared", "From global");
    writeSkillMd(agentsSkillsDir, "shared", "From agents");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [globalSkillsDir] });

    expect(skills).toHaveLength(1);
    expect(skills[0].description).toBe("From agents");
  });

  it("ignores unsupported .github/skills/ directories", () => {
    const ghSkillsDir = join(TEST_DIR, ".github", "skills");
    writeSkillMd(ghSkillsDir, "gh-skill", "From github dir");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(0);
  });
});
