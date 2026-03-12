/**
 * System prompt assembly — composes modular prompt sections
 * from markdown files into a single system prompt.
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { SkillRegistry } from "@devagent/runtime";
import type { TaskMode, TurnBriefing } from "@devagent/runtime";
import { formatBriefing } from "@devagent/runtime";
import { loadProjectContext } from "./project-context.js";

const PROMPTS_DIR = dirname(fileURLToPath(import.meta.url));

function loadPromptFile(filename: string): string {
  return readFileSync(join(PROMPTS_DIR, filename), "utf-8");
}

// Cache loaded prompt files (they never change during a process lifetime)
let cachedBase: string | null = null;
let cachedTools: string | null = null;
let cachedModeAct: string | null = null;
let cachedReview: string | null = null;

function getBase(): string {
  cachedBase ??= loadPromptFile("base.md");
  return cachedBase;
}

function getTools(): string {
  cachedTools ??= loadPromptFile("tools.md");
  return cachedTools;
}

function getModeAct(): string {
  cachedModeAct ??= loadPromptFile("mode-act.md");
  return cachedModeAct;
}

function getReview(): string {
  cachedReview ??= loadPromptFile("review.md");
  return cachedReview;
}

export interface AssemblePromptOptions {
  readonly mode: TaskMode;
  readonly repoRoot: string;
  readonly skills: SkillRegistry;
  readonly approvalMode?: string;
  readonly provider?: string;
  readonly model?: string;
  readonly includeReview?: boolean;
  /** Structured briefing from a prior turn (enables turn isolation). */
  readonly briefing?: TurnBriefing;
}

/**
 * Assemble the full system prompt from modular markdown sections.
 * Order: base + tools + mode + environment + project context + skills.
 */
export function assembleSystemPrompt(opts: AssemblePromptOptions): string {
  const sections: string[] = [];

  // Core identity and behavior
  sections.push(getBase());

  // Tool usage strategy
  sections.push(getTools());

  // Mode-specific constraints
  sections.push(getModeAct());

  // Review guidelines (loaded conditionally)
  if (opts.includeReview) {
    sections.push(getReview());
  }

  // Environment context
  const envLines = [
    `Working directory: ${opts.repoRoot}`,
    `Task mode: ${opts.mode}`,
  ];
  if (opts.approvalMode) {
    envLines.push(`Approval mode: ${opts.approvalMode}`);
  }
  if (opts.provider && opts.model) {
    envLines.push(`Provider: ${opts.provider} / ${opts.model}`);
  } else if (opts.provider) {
    envLines.push(`Provider: ${opts.provider}`);
  } else if (opts.model) {
    envLines.push(`Model: ${opts.model}`);
  }
  envLines.push(`Date: ${new Date().toISOString().split("T")[0]}`);
  sections.push(`## Environment\n\n${envLines.join("\n")}`);

  // Project-level instructions (AGENTS.md, CLAUDE.md, etc.)
  const projectContext = loadProjectContext(opts.repoRoot);
  if (projectContext) {
    sections.push(projectContext);
  }

  // Skills (Agent Skills standard)
  const skillList = opts.skills.list();
  if (skillList.length > 0) {
    const skillLines = skillList
      .map((s) => `- ${s.name}: ${s.description} (${s.source})`)
      .join("\n");
    sections.push(
      `## Available Skills\n\n${skillLines}\n\n` +
      `Use the \`invoke_skill\` tool to load a skill's full instructions when the ` +
      `user's task matches a skill description. Always invoke a relevant skill before starting work.`,
    );
  }

  // Session context briefing (turn isolation — replaces raw history)
  if (opts.briefing) {
    sections.push(
      `## Session Context\n\nYou are continuing a conversation. Here is a summary of prior work:\n\n${formatBriefing(opts.briefing)}`,
    );
  }

  return sections.join("\n\n");
}
