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

describe("SkillLoader — .github/skills/ discovery", () => {
  let loader: SkillLoader;

  beforeEach(() => {
    loader = new SkillLoader();
    mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("discovers skills from .github/skills/", () => {
    const ghSkillsDir = join(TEST_DIR, ".github", "skills");
    writeSkillMd(ghSkillsDir, "my-skill", "A test skill");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(1);
    expect(skills[0].name).toBe("my-skill");
    expect(skills[0].description).toBe("A test skill");
    expect(skills[0].source).toBe("project");
  });

  it("parses frontmatter name and description correctly", () => {
    const ghSkillsDir = join(TEST_DIR, ".github", "skills");
    writeSkillMd(ghSkillsDir, "code-review", "Automated code review helper");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(1);
    expect(skills[0].name).toBe("code-review");
    expect(skills[0].description).toBe("Automated code review helper");
    expect(skills[0].skillFilePath).toBe(
      join(ghSkillsDir, "code-review", "SKILL.md"),
    );
  });

  it(".devagent/skills/ overrides .github/skills/ for same skill name", () => {
    const ghSkillsDir = join(TEST_DIR, ".github", "skills");
    const devagentSkillsDir = join(TEST_DIR, ".devagent", "skills");
    writeSkillMd(ghSkillsDir, "shared", "From github");
    writeSkillMd(devagentSkillsDir, "shared", "From devagent");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(1);
    expect(skills[0].description).toBe("From devagent");
  });

  it("discovers skills from multiple project paths simultaneously", () => {
    const ghSkillsDir = join(TEST_DIR, ".github", "skills");
    const agentsSkillsDir = join(TEST_DIR, ".agents", "skills");
    writeSkillMd(ghSkillsDir, "gh-skill", "From github dir");
    writeSkillMd(agentsSkillsDir, "agents-skill", "From agents dir");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(2);
    const names = skills.map((s) => s.name).sort();
    expect(names).toEqual(["agents-skill", "gh-skill"]);
  });
});
