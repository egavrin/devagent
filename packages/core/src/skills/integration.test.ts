import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { SkillLoader, SkillRegistry, SkillResolver } from "./index.js";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";

const TEST_DIR = "/tmp/devagent-skills-integration-test";

function createFullSkill(
  base: string,
  name: string,
  description: string,
  instructions: string,
  opts?: { scripts?: boolean; references?: boolean; assets?: boolean },
): void {
  const dir = join(base, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "SKILL.md"),
    `---\nname: ${name}\ndescription: ${description}\n---\n${instructions}`,
    "utf-8",
  );
  if (opts?.scripts) mkdirSync(join(dir, "scripts"), { recursive: true });
  if (opts?.references) mkdirSync(join(dir, "references"), { recursive: true });
  if (opts?.assets) mkdirSync(join(dir, "assets"), { recursive: true });
}

describe("Skills Integration", () => {
  beforeEach(() => {
    mkdirSync(join(TEST_DIR, ".devagent", "skills"), { recursive: true });
    mkdirSync(join(TEST_DIR, ".agents", "skills"), { recursive: true });
    mkdirSync(join(TEST_DIR, ".claude", "skills"), { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("full pipeline: discover → register → load → resolve", async () => {
    createFullSkill(
      join(TEST_DIR, ".devagent", "skills"),
      "review",
      "Code review guidance",
      "Review using: $ARGUMENTS\nDir: ${SKILL_DIR}\nSession: ${SESSION_ID}",
      { scripts: true },
    );

    createFullSkill(
      join(TEST_DIR, ".agents", "skills"),
      "test",
      "Testing guidance",
      "Run tests for $0",
    );

    // Discover
    const loader = new SkillLoader();
    const metadata = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });
    expect(metadata).toHaveLength(2);

    // Register
    const registry = new SkillRegistry();
    registry.register(metadata);
    expect(registry.size).toBe(2);
    expect(registry.has("review")).toBe(true);
    expect(registry.has("test")).toBe(true);

    // Load
    const reviewSkill = await registry.load("review");
    expect(reviewSkill.hasScripts).toBe(true);
    expect(reviewSkill.hasReferences).toBe(false);

    // Resolve
    const resolver = new SkillResolver();
    const resolved = await resolver.resolve(reviewSkill, "main.ts", {
      sessionId: "sess-42",
      allowShellPreprocess: true,
    });

    expect(resolved.resolvedInstructions).toContain("Review using: main.ts");
    expect(resolved.resolvedInstructions).toContain(`Dir: ${reviewSkill.dirPath}`);
    expect(resolved.resolvedInstructions).toContain("Session: sess-42");
  });

  it("priority: .devagent overrides .agents for same-name skill", async () => {
    createFullSkill(
      join(TEST_DIR, ".agents", "skills"),
      "deploy",
      "Deploy (agents version)",
      "Lower priority",
    );
    createFullSkill(
      join(TEST_DIR, ".devagent", "skills"),
      "deploy",
      "Deploy (devagent version)",
      "Higher priority",
    );

    const loader = new SkillLoader();
    const metadata = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });
    expect(metadata).toHaveLength(1);

    const registry = new SkillRegistry();
    registry.register(metadata);
    const skill = await registry.load("deploy");
    expect(skill.instructions).toBe("Higher priority");
    expect(skill.description).toBe("Deploy (devagent version)");
  });

  it("claude-compat: discovers from .claude/skills/", async () => {
    createFullSkill(
      join(TEST_DIR, ".claude", "skills"),
      "claude-native",
      "Works with Claude Code too",
      "Cross-tool instructions",
    );

    const loader = new SkillLoader();
    const metadata = loader.discover({ repoRoot: TEST_DIR, globalPaths: [] });
    expect(metadata).toHaveLength(1);
    expect(metadata[0]!.source).toBe("claude-compat");
  });
});
