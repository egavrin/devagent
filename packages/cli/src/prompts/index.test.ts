import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { SkillRegistry } from "@devagent/core";
import { assembleSystemPrompt } from "./index.js";

function createTempRepo(): string {
  return mkdtempSync(join(tmpdir(), "devagent-prompt-assemble-"));
}

function mockSkills(
  list: ReadonlyArray<{ name: string; description: string }>,
): SkillRegistry {
  return {
    list: () => list,
  } as unknown as SkillRegistry;
}

describe("assembleSystemPrompt", () => {
  it("includes mode and environment metadata", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      approvalMode: "full-auto",
      provider: "openai",
      model: "gpt-5",
    });

    expect(prompt).toContain("## Mode: ACT");
    expect(prompt).toContain(`Working directory: ${repoRoot}`);
    expect(prompt).toContain("Task mode: act");
    expect(prompt).toContain("Approval mode: full-auto");
    expect(prompt).toContain("Provider: openai / gpt-5");
    expect(prompt).toMatch(/Date: \d{4}-\d{2}-\d{2}/);
  });

  it("includes review guidance only when requested", () => {
    const repoRoot = createTempRepo();

    const withoutReview = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      includeReview: false,
    });
    expect(withoutReview).not.toContain("## Code Review Guidelines");

    const withReview = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      includeReview: true,
    });
    expect(withReview).toContain("## Code Review Guidelines");
  });

  it("injects project instructions and available skills", () => {
    const repoRoot = createTempRepo();
    writeFileSync(join(repoRoot, "AGENTS.md"), "# Rules\nStay consistent.\n", "utf-8");

    const prompt = assembleSystemPrompt({
      mode: "plan",
      repoRoot,
      skills: mockSkills([{ name: "reviewer", description: "Review code quality." }]),
    });

    expect(prompt).toContain("Source: `AGENTS.md`");
    expect(prompt).toContain("## Available Skills");
    expect(prompt).toContain("- reviewer: Review code quality.");
  });
});
