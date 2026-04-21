import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

import { SkillAccessManager } from "./access.js";
import { SkillRegistry } from "./registry.js";
import type { SkillMetadata } from "./types.js";

const TEST_DIR = "/tmp/devagent-skill-access-test";

function makeMetadata(name: string): SkillMetadata {
  const dirPath = join(TEST_DIR, name);
  mkdirSync(dirPath, { recursive: true });
  writeFileSync(
    join(dirPath, "SKILL.md"),
    `---\nname: ${name}\ndescription: ${name}\n---\nInstructions`,
    "utf-8",
  );
  return {
    name,
    description: `${name} description`,
    source: "global",
    dirPath,
    skillFilePath: join(dirPath, "SKILL.md"),
  };
}

describe("SkillAccessManager", () => {
  let registry: SkillRegistry;

  beforeEach(() => {
    registry = new SkillRegistry();
    mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("unlocks known skills and persists them", () => {
    registry.register([makeMetadata("modernize-code")]);
    const persist = vi.fn();
    const manager = new SkillAccessManager(registry, {
      persistUnlockedSkill: persist,
    });

    manager.unlock("modernize-code");

    expect(manager.isUnlocked("modernize-code")).toBe(true);
    expect(persist).toHaveBeenCalledWith("modernize-code");
  });

  it("restores unlocked skills from persistence", () => {
    registry.register([makeMetadata("modernize-code")]);
    const manager = new SkillAccessManager(registry, {
      loadUnlockedSkillNames: () => ["modernize-code"],
    });

    expect(manager.isUnlocked("modernize-code")).toBe(true);
  });

  it("hydrates persisted state only once", () => {
    registry.register([makeMetadata("modernize-code")]);
    const loadUnlockedSkillNames = vi.fn(() => ["modernize-code"]);
    const manager = new SkillAccessManager(registry, {
      loadUnlockedSkillNames,
    });

    expect(manager.isUnlocked("modernize-code")).toBe(true);
    expect(manager.requireUnlocked("modernize-code").name).toBe("modernize-code");
    expect(manager.isUnlocked("modernize-code")).toBe(true);
    expect(loadUnlockedSkillNames).toHaveBeenCalledTimes(1);
  });

  it("ignores restored skills that are no longer registered", () => {
    registry.register([makeMetadata("testing")]);
    const manager = new SkillAccessManager(registry, {
      loadUnlockedSkillNames: () => ["missing-skill"],
    });

    expect(manager.isUnlocked("missing-skill")).toBe(false);
    expect(() => manager.requireUnlocked("missing-skill")).toThrow("not found");
  });
});
