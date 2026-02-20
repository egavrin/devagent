/**
 * System prompt assembly — composes modular prompt sections
 * from markdown files into a single system prompt.
 */

import { readFileSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";
import type { SkillRegistry, Memory } from "@devagent/core";
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

function getModePlan(): string {
  cachedModePlan ??= loadPromptFile("mode-plan.md");
  return cachedModePlan;
}

function getReview(): string {
  cachedReview ??= loadPromptFile("review.md");
  return cachedReview;
}

export interface AssemblePromptOptions {
  readonly mode: TaskMode;
  readonly repoRoot: string;
  readonly skills: SkillRegistry;
  readonly memories?: ReadonlyArray<Memory>;
  readonly approvalMode?: string;
  readonly provider?: string;
  readonly model?: string;
  readonly includeReview?: boolean;
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

  // Review guidelines (loaded conditionally)
  if (opts.includeReview) {
    sections.push(getReview());
  }

  // Environment context
  const envLines = [`Working directory: ${opts.repoRoot}`];
  if (opts.approvalMode) {
    envLines.push(`Approval mode: ${opts.approvalMode}`);
  }
  if (opts.provider && opts.model) {
    envLines.push(`Provider: ${opts.provider} / ${opts.model}`);
  }
  envLines.push(`Date: ${new Date().toISOString().split("T")[0]}`);
  sections.push(`## Environment\n\n${envLines.join("\n")}`);

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

  // Cross-session memories (cap at 10 entries, ~2000 chars)
  if (opts.memories && opts.memories.length > 0) {
    const MAX_MEMORIES = 10;
    const MAX_CHARS = 2000;
    const capped = opts.memories.slice(0, MAX_MEMORIES);
    let totalChars = 0;
    const memoryLines: string[] = [];
    for (const m of capped) {
      const line = `- [${m.category}] ${m.key}: ${m.content}`;
      if (totalChars + line.length > MAX_CHARS) break;
      memoryLines.push(line);
      totalChars += line.length;
    }
    if (memoryLines.length > 0) {
      sections.push(
        `## Learned Patterns\n\nThe following are lessons from previous sessions. Apply them when relevant:\n\n${memoryLines.join("\n")}`,
      );
    }
  }

  return sections.join("\n\n");
}
