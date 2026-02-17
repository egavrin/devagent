import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { SkillRegistry } from "./skills.js";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";

const TEST_DIR = "/tmp/devagent-skills-test";
const SKILLS_DIR = join(TEST_DIR, ".devagent", "skills");

function writeSkill(name: string, frontmatter: string, body: string): void {
  writeFileSync(
    join(SKILLS_DIR, `${name}.md`),
    `---\n${frontmatter}\n---\n${body}`,
    "utf-8",
  );
}

describe("SkillRegistry", () => {
  let registry: SkillRegistry;

  beforeEach(() => {
    registry = new SkillRegistry();
    mkdirSync(SKILLS_DIR, { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("discovers skills from project directory", () => {
    writeSkill("review", 'name: review\ndescription: "Code review"', "Review the code...");
    registry.discover(TEST_DIR);
    expect(registry.size).toBe(1);
    expect(registry.has("review")).toBe(true);
  });

  it("uses filename as fallback name", () => {
    writeFileSync(
      join(SKILLS_DIR, "my-skill.md"),
      "No frontmatter here, just instructions.",
      "utf-8",
    );
    registry.discover(TEST_DIR);
    expect(registry.has("my-skill")).toBe(true);
  });

  it("lists skill metadata", () => {
    writeSkill("a", "name: a\ndescription: First", "Do thing A");
    writeSkill("b", "name: b\ndescription: Second", "Do thing B");
    registry.discover(TEST_DIR);
    const list = registry.list();
    expect(list).toHaveLength(2);
    expect(list.map((s) => s.name).sort()).toEqual(["a", "b"]);
  });

  it("loads skill instructions on demand", () => {
    writeSkill("arkts", 'name: arkts\ndescription: "ArkTS check"', "Check ArkTS constraints...");
    registry.discover(TEST_DIR);

    const skill = registry.load("arkts");
    expect(skill.name).toBe("arkts");
    expect(skill.instructions).toContain("Check ArkTS constraints");
    expect(skill.source).toBe("project");
  });

  it("caches loaded skills", () => {
    writeSkill("cached", "name: cached\ndescription: test", "Instructions");
    registry.discover(TEST_DIR);
    const first = registry.load("cached");
    const second = registry.load("cached");
    expect(first).toBe(second); // Same reference
  });

  it("throws on unknown skill name", () => {
    registry.discover(TEST_DIR);
    expect(() => registry.load("nonexistent")).toThrow(
      'Skill "nonexistent" not found',
    );
  });

  it("handles missing skills directory gracefully", () => {
    rmSync(SKILLS_DIR, { recursive: true, force: true });
    registry.discover(TEST_DIR);
    expect(registry.size).toBe(0);
  });

  it("ignores non-markdown files", () => {
    writeFileSync(join(SKILLS_DIR, "notes.txt"), "not a skill", "utf-8");
    writeSkill("real", "name: real\ndescription: yes", "Instructions");
    registry.discover(TEST_DIR);
    expect(registry.size).toBe(1);
    expect(registry.has("real")).toBe(true);
  });

  it("strips quotes from frontmatter values", () => {
    writeSkill("quoted", 'name: "my-skill"\ndescription: \'hello world\'', "Body");
    registry.discover(TEST_DIR);
    const meta = registry.getMetadata("my-skill");
    expect(meta).toBeDefined();
    expect(meta!.description).toBe("hello world");
  });

  it("clears all skills", () => {
    writeSkill("temp", "name: temp\ndescription: temp", "Body");
    registry.discover(TEST_DIR);
    expect(registry.size).toBe(1);
    registry.clear();
    expect(registry.size).toBe(0);
  });
});
