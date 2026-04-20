import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect, beforeEach, afterEach } from "vitest";

import { SkillRegistry } from "./registry.js";
import type { SkillMetadata } from "./types.js";

const TEST_DIR = "/tmp/devagent-skill-registry-test";

function makeMetadata(name: string, description: string): SkillMetadata {
  const dirPath = join(TEST_DIR, name);
  mkdirSync(dirPath, { recursive: true });
  writeFileSync(
    join(dirPath, "SKILL.md"),
    `---\nname: ${name}\ndescription: ${description}\n---\nInstructions for ${name}`,
    "utf-8",
  );
  return {
    name,
    description,
    source: "project",
    dirPath,
    skillFilePath: join(dirPath, "SKILL.md"),
  };
}

describe("SkillRegistry", () => {
  let registry: SkillRegistry;

  beforeEach(() => {
    registry = new SkillRegistry();
    mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("registers and lists skill metadata", () => {
    registry.register([makeMetadata("alpha", "First"), makeMetadata("beta", "Second")]);
    const list = registry.list();
    expect(list).toHaveLength(2);
    expect(list.map((s) => s.name).sort()).toEqual(["alpha", "beta"]);
  });

  it("reports correct size", () => {
    registry.register([makeMetadata("one", "desc")]);
    expect(registry.size).toBe(1);
  });

  it("checks skill existence", () => {
    registry.register([makeMetadata("exists", "desc")]);
    expect(registry.has("exists")).toBe(true);
    expect(registry.has("nope")).toBe(false);
  });

  it("returns metadata without loading body", () => {
    registry.register([makeMetadata("meta", "desc")]);
    const meta = registry.getMetadata("meta");
    expect(meta).toBeDefined();
    expect(meta!.name).toBe("meta");
  });

  it("loads full skill content asynchronously", async () => {
    registry.register([makeMetadata("loadable", "desc")]);
    const skill = await registry.load("loadable");
    expect(skill.instructions).toContain("Instructions for loadable");
    expect(skill.hasScripts).toBe(false);
  });

  it("caches loaded skills", async () => {
    registry.register([makeMetadata("cached", "desc")]);
    const first = await registry.load("cached");
    const second = await registry.load("cached");
    expect(first).toBe(second);
  });

  it("throws on unknown skill name", async () => {
    await expect(registry.load("nonexistent")).rejects.toThrow(
      'Skill "nonexistent" not found',
    );
  });

  it("includes available skills in error message", async () => {
    registry.register([makeMetadata("one", "d"), makeMetadata("two", "d")]);
    await expect(registry.load("three")).rejects.toThrow("Available:");
  });

  it("clears all skills and cache", async () => {
    registry.register([makeMetadata("temp", "desc")]);
    await registry.load("temp");
    registry.clear();
    expect(registry.size).toBe(0);
    await expect(registry.load("temp")).rejects.toThrow();
  });

  it("returns undefined for unknown metadata", () => {
    expect(registry.getMetadata("unknown")).toBeUndefined();
  });
});
