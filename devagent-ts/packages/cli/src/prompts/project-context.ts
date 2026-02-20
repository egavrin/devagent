/**
 * Project context loader — discovers and loads project-level instructions.
 * Loads ALL found instruction files (they may coexist with different purposes):
 *   1. .devagent/instructions.md — DevAgent-specific project rules
 *   2. AGENTS.md (Codex CLI compatible) — general agent instructions
 *   3. CLAUDE.md (Claude Code compatible) — Claude-specific instructions
 *
 * Total content budget: ~8000 chars (~2000 tokens), distributed proportionally
 * across found files.
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const INSTRUCTION_FILES: readonly { filename: string; scope: string }[] = [
  { filename: ".devagent/instructions.md", scope: "DevAgent project rules" },
  { filename: "AGENTS.md", scope: "Agent instructions (Codex-compatible)" },
  { filename: "CLAUDE.md", scope: "Agent instructions (Claude Code-compatible)" },
];

const TOTAL_MAX_CHARS = 8000;

export function loadProjectContext(repoRoot: string): string | null {
  // Discover all instruction files that exist and have content
  const found: { filename: string; scope: string; content: string }[] = [];

  for (const { filename, scope } of INSTRUCTION_FILES) {
    const filePath = join(repoRoot, filename);
    if (!existsSync(filePath)) continue;

    let content: string;
    try {
      content = readFileSync(filePath, "utf-8");
    } catch {
      continue;
    }

    if (content.trim().length === 0) continue;
    found.push({ filename, scope, content });
  }

  if (found.length === 0) return null;

  // Distribute char budget proportionally across found files
  const perFileMax = Math.floor(TOTAL_MAX_CHARS / found.length);

  const sections: string[] = [];
  for (const { filename, scope, content } of found) {
    let truncated = content;
    if (truncated.length > perFileMax) {
      truncated = truncated.substring(0, perFileMax) + "\n\n[...truncated]";
    }
    sections.push(`## Project Instructions (${scope})\n\nSource: \`${filename}\`\n\n${truncated}`);
  }

  return sections.join("\n\n");
}
