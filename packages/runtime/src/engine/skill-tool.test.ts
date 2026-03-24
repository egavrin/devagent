import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { createSkillTool } from "./skill-tool.js";
import { SkillAccessManager, SkillRegistry, SkillResolver } from "../core/index.js";
import type { SkillMetadata, ToolContext, DevAgentConfig } from "../core/index.js";
import { mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join } from "node:path";

const TEST_DIR = "/tmp/devagent-skill-tool-test";

function setupSkill(name: string, description: string, instructions: string): SkillMetadata {
  const dir = join(TEST_DIR, name);
  mkdirSync(dir, { recursive: true });
  writeFileSync(
    join(dir, "SKILL.md"),
    `---\nname: ${name}\ndescription: ${description}\n---\n${instructions}`,
    "utf-8",
  );
  return {
    name,
    description,
    source: "project",
    dirPath: dir,
    skillFilePath: join(dir, "SKILL.md"),
  };
}

const toolContext: ToolContext = {
  repoRoot: TEST_DIR,
  config: {} as DevAgentConfig,
  sessionId: "test-session",
};

describe("createSkillTool", () => {
  let registry: SkillRegistry;
  let resolver: SkillResolver;

  beforeEach(() => {
    registry = new SkillRegistry();
    resolver = new SkillResolver();
    mkdirSync(TEST_DIR, { recursive: true });
  });

  afterEach(() => {
    rmSync(TEST_DIR, { recursive: true, force: true });
  });

  it("creates a tool with correct metadata", () => {
    const tool = createSkillTool(registry, resolver);
    expect(tool.name).toBe("invoke_skill");
    expect(tool.category).toBe("readonly");
  });

  it("loads and returns skill instructions", async () => {
    registry.register([setupSkill("my-skill", "A test skill", "Do the thing")]);
    const tool = createSkillTool(registry, resolver);
    const result = await tool.handler({ name: "my-skill" }, toolContext);
    expect(result.success).toBe(true);
    expect(result.output).toContain("Do the thing");
  });

  it("substitutes arguments in skill instructions", async () => {
    registry.register([setupSkill("arg-skill", "Has args", "Run: $ARGUMENTS")]);
    const tool = createSkillTool(registry, resolver);
    const result = await tool.handler(
      { name: "arg-skill", arguments: "hello world" },
      toolContext,
    );
    expect(result.success).toBe(true);
    expect(result.output).toContain("Run: hello world");
  });

  it("returns error with available skills on unknown name", async () => {
    registry.register([setupSkill("known", "Known skill", "body")]);
    const tool = createSkillTool(registry, resolver);
    const result = await tool.handler({ name: "unknown" }, toolContext);
    expect(result.success).toBe(false);
    expect(result.error).toContain("not found");
    expect(result.error).toContain("known");
  });

  it("returns error when name parameter is missing", async () => {
    const tool = createSkillTool(registry, resolver);
    const result = await tool.handler({}, toolContext);
    expect(result.success).toBe(false);
    expect(result.error).toContain("name");
  });

  it("includes skill metadata in output", async () => {
    registry.register([setupSkill("meta-skill", "Has metadata", "Instructions here")]);
    const tool = createSkillTool(registry, resolver);
    const result = await tool.handler({ name: "meta-skill" }, toolContext);
    expect(result.success).toBe(true);
    expect(result.output).toContain("meta-skill");
  });

  it("unlocks support trees and advertises skill:// access", async () => {
    const metadata = setupSkill("support-skill", "Has support files", "Instructions here");
    mkdirSync(join(metadata.dirPath, "scripts"), { recursive: true });
    registry.register([metadata]);
    const skillAccess = new SkillAccessManager(registry);
    const tool = createSkillTool(registry, resolver, { skillAccess });

    const result = await tool.handler({ name: "support-skill" }, toolContext);

    expect(result.success).toBe(true);
    expect(skillAccess.isUnlocked("support-skill")).toBe(true);
    expect(result.output).toContain("skill://support-skill/");
  });

  it("advertises skill:// access for a backed skill with support-root content only", async () => {
    const metadata = setupSkill("backed-skill", "Backed skill", "Instructions here");
    const supportRoot = join(TEST_DIR, "backed-skill-support");
    mkdirSync(join(supportRoot, "docs"), { recursive: true });
    writeFileSync(join(supportRoot, "docs", "guide.md"), "# Guide\n", "utf-8");
    registry.register([{
      ...metadata,
      source: "global",
      supportRootPath: supportRoot,
      sourceRepoPath: supportRoot,
    }]);
    const skillAccess = new SkillAccessManager(registry);
    const tool = createSkillTool(registry, resolver, { skillAccess });

    const result = await tool.handler({ name: "backed-skill" }, toolContext);

    expect(result.success).toBe(true);
    expect(result.output).toContain("skill://backed-skill/");
  });
});
