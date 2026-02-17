/**
 * Project context loader — discovers and loads project-level instructions.
 * Discovery order (first found wins):
 *   1. .devagent/instructions.md
 *   2. AGENTS.md (Codex CLI compatible)
 *   3. CLAUDE.md (Claude Code compatible)
 *
 * Content is truncated at ~6000 chars (~1500 tokens).
 */

import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";

const INSTRUCTION_FILES = [
  ".devagent/instructions.md",
  "AGENTS.md",
  "CLAUDE.md",
];

const MAX_CHARS = 6000;

export function loadProjectContext(repoRoot: string): string | null {
  for (const filename of INSTRUCTION_FILES) {
    const filePath = join(repoRoot, filename);
    if (existsSync(filePath)) {
      let content: string;
      try {
        content = readFileSync(filePath, "utf-8");
      } catch {
        continue;
      }

      if (content.trim().length === 0) {
        continue;
      }

      if (content.length > MAX_CHARS) {
        content = content.substring(0, MAX_CHARS) + "\n\n[...truncated]";
      }

      return `## Project Instructions (from ${filename})\n\n${content}`;
    }
  }

  return null;
}
