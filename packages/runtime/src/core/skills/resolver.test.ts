import { mkdirSync, rmSync } from "node:fs";
import { join } from "node:path";
import { describe, it, expect, beforeEach, afterEach } from "vitest";

import { SkillResolver } from "./resolver.js";
import type { Skill } from "./types.js";

const TEST_DIR = "/tmp/devagent-skill-resolver-test";

function makeSkill(instructions: string, dirPath?: string): Skill {
  const dir = dirPath ?? join(TEST_DIR, "test-skill");
  return {
    name: "test-skill",
    description: "Test",
    source: "project",
    dirPath: dir,
    skillFilePath: join(dir, "SKILL.md"),
    instructions,
    hasScripts: false,
    hasReferences: false,
    hasAssets: false,
  };
}

const DEFAULT_CONTEXT = { sessionId: "test-session-123", allowShellPreprocess: true };

describe("SkillResolver", () => {
  let resolver: SkillResolver;

  beforeEach(() => {
    resolver = new SkillResolver();
    mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("passes through instructions with no substitution markers", async () => {
    const skill = makeSkill("Just plain instructions");
    const result = await resolver.resolve(skill, "", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toBe("Just plain instructions");
  });

  it("substitutes $ARGUMENTS with full argument string", async () => {
    const skill = makeSkill("Run: $ARGUMENTS");
    const result = await resolver.resolve(skill, "hello world", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toBe("Run: hello world");
  });

  it("substitutes positional $0, $1, $2", async () => {
    const skill = makeSkill("First: $0, Second: $1, Third: $2");
    const result = await resolver.resolve(skill, "alpha beta gamma", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toBe("First: alpha, Second: beta, Third: gamma");
  });

  it("substitutes $ARGUMENTS[N] syntax", async () => {
    const skill = makeSkill("Item: $ARGUMENTS[1]");
    const result = await resolver.resolve(skill, "foo bar baz", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toBe("Item: bar");
  });

  it("leaves missing positional args as empty string", async () => {
    const skill = makeSkill("Got: $0, Missing: $5");
    const result = await resolver.resolve(skill, "only-one", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toBe("Got: only-one, Missing: ");
  });

  it("substitutes ${SKILL_DIR}", async () => {
    const dir = join(TEST_DIR, "my-skill");
    mkdirSync(dir, { recursive: true });
    const skill = makeSkill("Dir: ${SKILL_DIR}", dir);
    const result = await resolver.resolve(skill, "", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toBe(`Dir: ${dir}`);
  });

  it("substitutes ${SESSION_ID}", async () => {
    const skill = makeSkill("Session: ${SESSION_ID}");
    const result = await resolver.resolve(skill, "", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toBe("Session: test-session-123");
  });

  it("executes !`command` shell preprocessing", async () => {
    const dir = join(TEST_DIR, "shell-skill");
    mkdirSync(dir, { recursive: true });
    const skill = makeSkill("Result: !`echo hello`", dir);
    const result = await resolver.resolve(skill, "", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toBe("Result: hello");
  });

  it("skips shell preprocessing when allowShellPreprocess is false", async () => {
    const skill = makeSkill("Result: !`echo hello`");
    const result = await resolver.resolve(skill, "", {
      ...DEFAULT_CONTEXT,
      allowShellPreprocess: false,
    });
    expect(result.resolvedInstructions).toBe("Result: !`echo hello`");
  });

  it("does not recursively substitute shell output", async () => {
    const dir = join(TEST_DIR, "no-recurse");
    mkdirSync(dir, { recursive: true });
    // Use printf to avoid shell expansion of $ARGUMENTS
    const skill = makeSkill("Result: !`printf '$ARGUMENTS'`", dir);
    const result = await resolver.resolve(skill, "secret", DEFAULT_CONTEXT);
    // Shell output should NOT be re-processed for $ARGUMENTS
    expect(result.resolvedInstructions).toBe("Result: $ARGUMENTS");
  });

  it("handles shell command timeout", async () => {
    const dir = join(TEST_DIR, "timeout-skill");
    mkdirSync(dir, { recursive: true });
    const skill = makeSkill("Result: !`sleep 10`", dir);
    const shortResolver = new SkillResolver({ shellTimeoutMs: 100 });
    const result = await shortResolver.resolve(skill, "", DEFAULT_CONTEXT);
    expect(result.resolvedInstructions).toContain("[shell error");
  });

  it("preserves original skill fields in resolved output", async () => {
    const skill = makeSkill("Instructions: $ARGUMENTS");
    const result = await resolver.resolve(skill, "test", DEFAULT_CONTEXT);
    expect(result.name).toBe("test-skill");
    expect(result.description).toBe("Test");
    expect(result.source).toBe("project");
    expect(result.instructions).toBe("Instructions: $ARGUMENTS");
    expect(result.resolvedInstructions).toBe("Instructions: test");
  });
});
