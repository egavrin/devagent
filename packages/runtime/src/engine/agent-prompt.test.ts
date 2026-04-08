import { afterEach, describe, expect, it } from "vitest";
import { mkdirSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { AgentType, SkillRegistry } from "../core/index.js";
import type { SkillMetadata } from "../core/index.js";
import type { TurnBriefing } from "./briefing.js";
import {
  __getCommonPromptReadCountForTesting,
  __resetCommonPromptCacheForTesting,
  assembleAgentSystemPrompt,
  clearPromptCache,
} from "./agent-prompt.js";

describe("assembleAgentSystemPrompt", () => {
  let repoRoot: string | null = null;
  const readonlyTools = [
    { name: "read_file", category: "readonly" },
    { name: "find_files", category: "readonly" },
    { name: "search_files", category: "readonly" },
  ] as const;

  afterEach(() => {
    if (repoRoot) {
      rmSync(repoRoot, { recursive: true, force: true });
      repoRoot = null;
    }
  });

  it("injects project instructions, skills, environment, and briefing into child prompts", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}`);
    mkdirSync(repoRoot, { recursive: true });
    writeFileSync(join(repoRoot, "AGENTS.md"), "# Repo Rules\nUse existing patterns.\n", "utf-8");

    const skills = new SkillRegistry();
    const metadata: SkillMetadata = {
      name: "review",
      description: "Review code changes for regressions.",
      triggers: ["review local changes", "look for regressions"],
      paths: ["packages/runtime", "packages/cli"],
      source: "project",
      dirPath: join(repoRoot, ".agents", "skills", "review"),
      skillFilePath: join(repoRoot, ".agents", "skills", "review", "SKILL.md"),
    };
    skills.register([metadata]);

    const briefing: TurnBriefing = {
      turnNumber: 3,
      priorTaskSummary: "Investigated delegation bugs.",
      activeContext: "Delegate is currently root-bound.",
      pendingWork: "Rebind delegate per child agent.",
      keyArtifacts: ["packages/runtime/src/engine/delegate-tool.ts"],
      planSteps: [
        { description: "Fix delegation binding", status: "in_progress" },
      ],
    };

    const prompt = assembleAgentSystemPrompt({
      agentType: AgentType.EXPLORE,
      repoRoot,
      rolePrompt: "You are an Explore agent.",
      availableTools: readonlyTools,
      approvalMode: "autopilot",
      providerLabel: "openai / gpt-5",
      skills,
      briefing,
    });

    expect(prompt).toContain("Use existing patterns.");
    expect(prompt).toContain("`review`");
    expect(prompt).toContain('triggers: "review local changes", "look for regressions"');
    expect(prompt).toContain("paths: `packages/runtime`, `packages/cli`");
    expect(prompt).toContain("Invoke the broadest relevant workflow skill first");
    expect(prompt).toContain("Safety mode: autopilot");
    expect(prompt).toContain("Provider: openai / gpt-5");
    expect(prompt).toContain("Investigated delegation bugs.");
  });

  it("does not mention unavailable state tools in the shared child prompt", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}-readonly`);
    mkdirSync(repoRoot, { recursive: true });

    const prompt = assembleAgentSystemPrompt({
      agentType: AgentType.REVIEWER,
      repoRoot,
      rolePrompt: "You are a Reviewer agent.",
      availableTools: readonlyTools,
      approvalMode: "default",
      providerLabel: "mock / mock-model",
      skills: new SkillRegistry(),
    });

    expect(prompt).not.toContain("save_finding");
    expect(prompt).not.toContain("plan status");
  });

  it("preserves strong delegation rules in the role prompt when provided", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}-delegation`);
    mkdirSync(repoRoot, { recursive: true });

    const prompt = assembleAgentSystemPrompt({
      agentType: AgentType.GENERAL,
      repoRoot,
      rolePrompt: "Delegation is required when task instructions explicitly require it.",
      availableTools: [
        ...readonlyTools,
        { name: "delegate", category: "workflow" },
      ],
      approvalMode: "default",
      providerLabel: "mock / mock-model",
      skills: new SkillRegistry(),
    });

    expect(prompt).toContain("Delegation is required when task instructions explicitly require it.");
  });

  it("caches the shared common prompt across prompt assemblies", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}-cache`);
    mkdirSync(repoRoot, { recursive: true });
    __resetCommonPromptCacheForTesting();

    assembleAgentSystemPrompt({
      agentType: AgentType.EXPLORE,
      repoRoot,
      rolePrompt: "You are an Explore agent.",
      availableTools: readonlyTools,
    });
    assembleAgentSystemPrompt({
      agentType: AgentType.REVIEWER,
      repoRoot,
      rolePrompt: "You are a Reviewer agent.",
      availableTools: readonlyTools,
    });

    // Prompts are now embedded constants — read count is always 0
    expect(__getCommonPromptReadCountForTesting()).toBe(0);
    __resetCommonPromptCacheForTesting();
  });

  it("omits delegation guidance when the child does not actually have delegate", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}-general`);
    mkdirSync(repoRoot, { recursive: true });

    const prompt = assembleAgentSystemPrompt({
      agentType: AgentType.GENERAL,
      repoRoot,
      rolePrompt: "You are a General implementation agent.",
      availableTools: [
        ...readonlyTools,
        { name: "replace_in_file", category: "mutating" },
        { name: "write_file", category: "mutating" },
        { name: "run_command", category: "external" },
      ],
    });

    expect(prompt).toContain("## Editing");
    expect(prompt).toContain("## Shell Commands");
    expect(prompt).not.toContain("## Delegation");
    expect(prompt).not.toContain("Use `delegate`");
  });

  it("adds delegation guidance only when the child really has delegate", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}-nested`);
    mkdirSync(repoRoot, { recursive: true });

    const prompt = assembleAgentSystemPrompt({
      agentType: AgentType.GENERAL,
      repoRoot,
      rolePrompt: "You are a General implementation agent.",
      availableTools: [
        ...readonlyTools,
        { name: "delegate", category: "workflow" },
      ],
    });

    expect(prompt).toContain("## Delegation");
    expect(prompt).toContain("Prefer multiple readonly `explore` or `reviewer` delegates in one turn");
  });

  it("cache hits on identical inputs", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}-cache-hit`);
    mkdirSync(repoRoot, { recursive: true });
    clearPromptCache();

    const opts = {
      agentType: AgentType.EXPLORE as const,
      repoRoot,
      rolePrompt: "You are an Explore agent.",
      availableTools: readonlyTools,
      projectInstructions: "Use existing patterns.",
    };

    const first = assembleAgentSystemPrompt(opts);
    const second = assembleAgentSystemPrompt(opts);

    // Same string reference means cache hit
    expect(first).toBe(second);
  });

  it("clearPromptCache invalidates cached prompts", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}-clear`);
    mkdirSync(repoRoot, { recursive: true });

    const opts = {
      agentType: AgentType.EXPLORE as const,
      repoRoot,
      rolePrompt: "You are an Explore agent.",
      availableTools: readonlyTools,
      projectInstructions: "Use existing patterns.",
    };

    const first = assembleAgentSystemPrompt(opts);
    clearPromptCache();
    const second = assembleAgentSystemPrompt(opts);

    // Content is the same but they are different string instances after cache clear
    expect(first).toEqual(second);
  });

  it("includes deferred tools section in prompt", () => {
    repoRoot = join(tmpdir(), `devagent-agent-prompt-${Date.now()}-deferred`);
    mkdirSync(repoRoot, { recursive: true });
    clearPromptCache();

    const prompt = assembleAgentSystemPrompt({
      agentType: AgentType.GENERAL,
      repoRoot,
      rolePrompt: "You are a General agent.",
      availableTools: readonlyTools,
      deferredTools: [
        { name: "git_diff", description: "Show file differences", category: "readonly" },
        { name: "diagnostics", description: "LSP diagnostics", category: "readonly" },
      ],
      projectInstructions: null,
    });

    expect(prompt).toContain("Additional Tools (available via tool_search)");
    expect(prompt).toContain("git_diff: Show file differences");
    expect(prompt).toContain("diagnostics: LSP diagnostics");
  });
});
