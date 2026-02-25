import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { loadProjectContext } from "./project-context.js";

function createTempRepo(): string {
  return mkdtempSync(join(tmpdir(), "devagent-prompt-context-"));
}

describe("loadProjectContext", () => {
  it("returns null when no instruction files are present", () => {
    const repo = createTempRepo();
    expect(loadProjectContext(repo)).toBeNull();
  });

  it("loads supported instruction files in priority order", () => {
    const repo = createTempRepo();
    mkdirSync(join(repo, ".devagent"), { recursive: true });

    writeFileSync(
      join(repo, ".devagent", "ai_agent_instructions.md"),
      "# AI rules\nDo X.\n",
      "utf-8",
    );
    writeFileSync(
      join(repo, ".devagent", "instructions.md"),
      "# Legacy rules\nDo Y.\n",
      "utf-8",
    );
    writeFileSync(join(repo, "AGENTS.md"), "# Agents\nDo Z.\n", "utf-8");
    writeFileSync(join(repo, "CLAUDE.md"), "# Claude\nDo W.\n", "utf-8");

    const context = loadProjectContext(repo);
    expect(context).not.toBeNull();

    const text = context!;
    const aiIdx = text.indexOf("`.devagent/ai_agent_instructions.md`");
    const legacyIdx = text.indexOf("`.devagent/instructions.md`");
    const agentsIdx = text.indexOf("`AGENTS.md`");
    const claudeIdx = text.indexOf("`CLAUDE.md`");

    expect(aiIdx).toBeGreaterThanOrEqual(0);
    expect(legacyIdx).toBeGreaterThan(aiIdx);
    expect(agentsIdx).toBeGreaterThan(legacyIdx);
    expect(claudeIdx).toBeGreaterThan(agentsIdx);
  });

  it("marks content as truncated when it exceeds budget", () => {
    const repo = createTempRepo();
    writeFileSync(join(repo, "AGENTS.md"), "A".repeat(50_000), "utf-8");

    const context = loadProjectContext(repo);
    expect(context).not.toBeNull();
    expect(context!).toContain("[...truncated]");
    expect(context!).toMatch(/\[Source exceeds \d+ chars\./);
  });
});
