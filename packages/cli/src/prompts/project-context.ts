/**
 * Project context loader — discovers and loads project-level instructions.
 * Loads ALL found instruction files (they may coexist with different purposes):
 *   1. .devagent/ai_agent_instructions.md — DevAgent-specific project rules
 *   2. .devagent/instructions.md — legacy DevAgent project rules
 *   3. AGENTS.md (Codex-compatible) — general agent instructions
 *   4. CLAUDE.md (Claude-compatible) — Claude-specific instructions
 *
 * Total content budget is distributed by file priority rather than equally.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { extractErrorMessage } from "@devagent/core";

interface InstructionFileSpec {
  readonly filename: string;
  readonly scope: string;
  readonly priority: number;
}

interface FoundInstruction extends InstructionFileSpec {
  readonly content: string;
}

const INSTRUCTION_FILES: readonly InstructionFileSpec[] = [
  {
    filename: ".devagent/ai_agent_instructions.md",
    scope: "DevAgent project rules",
    priority: 1.0,
  },
  {
    filename: ".devagent/instructions.md",
    scope: "DevAgent project rules (legacy)",
    priority: 0.9,
  },
  {
    filename: "AGENTS.md",
    scope: "Agent instructions (Codex-compatible)",
    priority: 0.8,
  },
  {
    filename: "CLAUDE.md",
    scope: "Agent instructions (Claude Code-compatible)",
    priority: 0.6,
  },
];

// Benchmark target: Codex defaults project instructions to 32 KiB.
const TOTAL_MAX_CHARS = 32 * 1024;

function allocateBudgets(
  files: ReadonlyArray<FoundInstruction>,
  totalBudget: number,
): number[] {
  if (files.length === 0 || totalBudget <= 0) return [];

  const minPerFile = Math.max(250, Math.floor(totalBudget / (files.length * 3)));
  const budgets = new Array<number>(files.length).fill(0);

  let remainingBudget = totalBudget;
  let remainingPriority = files.reduce((sum, file) => sum + file.priority, 0);

  for (let i = 0; i < files.length; i++) {
    const isLast = i === files.length - 1;
    if (isLast) {
      budgets[i] = Math.max(0, remainingBudget);
      break;
    }

    const filesLeft = files.length - i;
    const minForRest = minPerFile * (filesLeft - 1);
    const maxForCurrent = Math.max(minPerFile, remainingBudget - minForRest);

    const weightedShare = remainingPriority > 0
      ? Math.floor((remainingBudget * files[i]!.priority) / remainingPriority)
      : maxForCurrent;

    const budget = Math.max(minPerFile, Math.min(maxForCurrent, weightedShare));

    budgets[i] = budget;
    remainingBudget -= budget;
    remainingPriority -= files[i]!.priority;
  }

  return budgets;
}

function truncateContent(content: string, maxChars: number): {
  readonly text: string;
  readonly truncated: boolean;
} {
  if (content.length <= maxChars) {
    return { text: content, truncated: false };
  }

  const trimmed = content.substring(0, maxChars).trimEnd();
  return {
    text: `${trimmed}\n\n[...truncated]`,
    truncated: true,
  };
}

export function loadProjectContext(repoRoot: string): string | null {
  // Discover all instruction files that exist and have content
  const found: FoundInstruction[] = [];

  for (const spec of INSTRUCTION_FILES) {
    const { filename, scope, priority } = spec;
    const filePath = join(repoRoot, filename);
    if (!existsSync(filePath)) continue;

    let content: string;
    try {
      content = readFileSync(filePath, "utf-8");
    } catch (error) {
      const message = extractErrorMessage(error);
      throw new Error(`Failed to read project instruction file "${filename}": ${message}`);
    }

    if (content.trim().length === 0) continue;
    found.push({ filename, scope, priority, content });
  }

  if (found.length === 0) return null;

  // Higher-priority instruction files get proportionally larger budgets.
  const ordered = [...found].sort((a, b) => b.priority - a.priority);
  const budgets = allocateBudgets(ordered, TOTAL_MAX_CHARS);

  const sections: string[] = [];
  for (let i = 0; i < ordered.length; i++) {
    const { filename, scope, content } = ordered[i]!;
    const budget = budgets[i] ?? 0;
    const { text, truncated } = truncateContent(content, budget);
    let section = `## Project Instructions (${scope})\n\nSource: \`${filename}\`\n\n${text}`;
    if (truncated) {
      section += `\n\n[Source exceeds ${budget} chars. Read \`${filename}\` for full context.]`;
    }
    sections.push(section);
  }

  return sections.join("\n\n");
}
