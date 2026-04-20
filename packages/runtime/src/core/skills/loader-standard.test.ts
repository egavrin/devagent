import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect, beforeEach, afterEach } from "vitest";

import { SkillLoader } from "./loader.js";

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

  it("parses optional trigger metadata from comma-separated and list frontmatter", () => {
    const agentsSkillsDir = join(TEST_DIR, ".agents", "skills");
    const dir = join(agentsSkillsDir, "surface-change-e2e");
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, "SKILL.md"),
      [
        "---",
        "name: surface-change-e2e",
        "description: Coordinate CLI and runtime surface changes.",
        "triggers: update CLI behavior, help text drift, command help",
        "paths:",
        "  - packages/cli",
        "  - packages/runtime",
        "examples:",
        "  - update CLI behavior",
        "  - fix docs drift after a command change",
        "---",
        "Instructions",
      ].join("\n"),
      "utf-8",
    );

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(1);
    expect(skills[0].triggers).toEqual([
      "update CLI behavior",
      "help text drift",
      "command help",
    ]);
    expect(skills[0].paths).toEqual([
      "packages/cli",
      "packages/runtime",
    ]);
    expect(skills[0].examples).toEqual([
      "update CLI behavior",
      "fix docs drift after a command change",
    ]);
  });

  it("keeps old skills discoverable when optional trigger metadata is absent", () => {
    const agentsSkillsDir = join(TEST_DIR, ".agents", "skills");
    writeSkillMd(agentsSkillsDir, "testing", "Focused verification guidance");

    const skills = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });

    expect(skills).toHaveLength(1);
    expect(skills[0].triggers).toBeUndefined();
    expect(skills[0].paths).toBeUndefined();
    expect(skills[0].examples).toBeUndefined();
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
