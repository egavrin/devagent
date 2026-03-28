import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import type { SkillRegistry } from "@devagent/runtime";
import { assembleSystemPrompt } from "./index.js";

const FULL_TOOLSET = [
  { name: "read_file", category: "readonly" },
  { name: "find_files", category: "readonly" },
  { name: "search_files", category: "readonly" },
  { name: "replace_in_file", category: "mutating" },
  { name: "write_file", category: "mutating" },
  { name: "run_command", category: "external" },
  { name: "execute_tool_script", category: "readonly" },
  { name: "delegate", category: "workflow" },
] as const;

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
      availableTools: FULL_TOOLSET,
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
      availableTools: FULL_TOOLSET,
      includeReview: false,
    });
    expect(withoutReview).not.toContain("## Code Review Guidelines");

    const withReview = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET,
      includeReview: true,
    });
    expect(withReview).toContain("## Code Review Guidelines");
  });

  it("injects project instructions and available skills", () => {
    const repoRoot = createTempRepo();
    writeFileSync(join(repoRoot, "AGENTS.md"), "# Rules\nStay consistent.\n", "utf-8");

    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([{ name: "reviewer", description: "Review code quality." }]),
      availableTools: FULL_TOOLSET,
    });

    expect(prompt).toContain("Source: `AGENTS.md`");
    expect(prompt).toContain("## Available Skills");
    expect(prompt).toContain("- reviewer: Review code quality.");
  });

  it("makes scenario-mandated delegation and investigation playbooks explicit in the parent prompt", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET,
      agentModelOverrides: {
        explore: "gpt-5.4-mini",
        reviewer: "gpt-5.4",
        architect: "gpt-5.4",
      },
      agentReasoningOverrides: {
        explore: "low",
        reviewer: "high",
        architect: "high",
      },
    });

    expect(prompt).toContain("## Delegation and Decomposition");
    expect(prompt).toContain("`explore` for codebase search, repo discovery");
    expect(prompt).toContain("Multiple codebase searches or repo discovery -> `explore`.");
    expect(prompt).toContain("delegate before proceeding");
    expect(prompt).toContain("emit multiple `explore` delegates in the same turn");
    expect(prompt).toContain("Do not send one umbrella `explore` delegate");
    expect(prompt).toContain("Broad cross-repo contradiction analysis");
    expect(prompt).toContain("Narrow lookup like `where is X lowered?`");
    expect(prompt).toContain("## Investigation Playbook");
    expect(prompt).toContain("Do not send one umbrella `explore` delegate");
    expect(prompt).toContain("first delegate returns `partial`");
    expect(prompt).toContain("## Subagent Model Policy");
    expect(prompt).toContain("explore | model: gpt-5.4-mini | reasoning: low");
  });

  it("documents the cross-repo fallback after repo-root path-guard failures", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET,
    });

    expect(prompt).toContain("## Cross-Repo Search");
    expect(prompt).toContain("bounded to the current repo root");
    expect(prompt).toContain("One path-guard failure on a `../...` target is enough to pivot");
    expect(prompt).toContain("readonly shell search with targeted `run_command`");
    expect(prompt).toContain("../arkcompiler_ets_frontend");
  });

  it("teaches evidence-lane planning for broad research tasks", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET,
    });

    expect(prompt).toContain("## Investigation Playbook");
    expect(prompt).toContain("Create an evidence-lane plan before searching.");
    expect(prompt).toContain("docs/spec claims");
    expect(prompt).toContain("frontend/compile-time behavior");
    expect(prompt).toContain("runtime/tests behavior");
    expect(prompt).toContain("Avoid serial phase plans like `locate -> inspect -> compare -> summarize`");
  });

  it("omits delegation guidance when delegate is unavailable", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET.filter((tool) => tool.name !== "delegate"),
    });

    expect(prompt).not.toContain("## Delegation and Decomposition");
    expect(prompt).not.toContain("delegate before proceeding");
    expect(prompt).toContain("If delegation is unavailable, keep the same lane plan");
  });

  it("omits cross-repo shell fallback when run_command is unavailable", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET.filter((tool) => tool.name !== "run_command"),
    });

    expect(prompt).not.toContain("## Shell Operations");
    expect(prompt).not.toContain("## Cross-Repo Search");
    expect(prompt).not.toContain("readonly shell search with targeted `run_command`");
  });

  it("omits editing guidance when mutating tools are unavailable", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET.filter((tool) => tool.category !== "mutating"),
    });

    expect(prompt).not.toContain("## Editing");
    expect(prompt).not.toContain("After `write_file`, immediately verify");
  });

  it("treats readonly as no mutation, not no inspection", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET,
    });

    expect(prompt).toContain("\"Read-only\" means no repo mutation.");
    expect(prompt).toContain("readonly delegates");
    expect(prompt).toContain("readonly shell search");
  });

  it("prefers delegation over execute_tool_script for broad research", () => {
    const repoRoot = createTempRepo();
    const prompt = assembleSystemPrompt({
      mode: "act",
      repoRoot,
      skills: mockSkills([]),
      availableTools: FULL_TOOLSET,
    });

    expect(prompt).toContain("Use it after lane selection. Do not use broad reconnaissance batches as a substitute for evidence-lane decomposition.");
    expect(prompt).toContain("lane decomposition via `explore` delegates takes priority over local batching.");
  });
});
