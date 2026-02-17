/**
 * System prompt assembly — composes modular prompt sections
 * from markdown files into a single system prompt.
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { SkillRegistry } from "@devagent/core";
import type { TaskMode } from "@devagent/engine";
import { loadProjectContext } from "./project-context.js";

const PROMPTS_DIR = dirname(fileURLToPath(import.meta.url));

function loadPromptFile(filename: string): string {
  return readFileSync(join(PROMPTS_DIR, filename), "utf-8");
}

// Cache loaded prompt files (they never change during a process lifetime)
let cachedBase: string | null = null;
let cachedTools: string | null = null;
let cachedModeAct: string | null = null;
let cachedModePlan: string | null = null;

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

function getModePlan(): string {
  cachedModePlan ??= loadPromptFile("mode-plan.md");
  return cachedModePlan;
}

export interface AssemblePromptOptions {
  readonly mode: TaskMode;
  readonly repoRoot: string;
  readonly skills: SkillRegistry;
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
  sections.push(opts.mode === "plan" ? getModePlan() : getModeAct());

  // Environment context
  sections.push(`## Environment\n\nWorking directory: ${opts.repoRoot}`);

  // Project-level instructions (AGENTS.md, CLAUDE.md, etc.)
  const projectContext = loadProjectContext(opts.repoRoot);
  if (projectContext) {
    sections.push(projectContext);
  }

  // Skills
  const skillList = opts.skills.list();
  if (skillList.length > 0) {
    const skillLines = skillList
      .map((s) => `- ${s.name}: ${s.description}`)
      .join("\n");
    sections.push(
      `## Available Skills\n\n${skillLines}\nYou can reference these skills when the user asks about related topics.`,
    );
  }

  return sections.join("\n\n");
}
