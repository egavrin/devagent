/**
 * Repository context loader — resolves repo-level instruction files
 * for workflow phases. Follows the priority order:
 *
 * 1. WORKFLOW.md (repo process contract)
 * 2. Nearest AGENTS.md (agent behavior rules)
 * 3. .github/copilot-instructions.md (repo-wide instructions)
 * 4. .github/instructions/**\/*.instructions.md (path-specific)
 * 5. Stage/path-selected skills
 */

import { readFileSync, existsSync, readdirSync } from "node:fs";
import { join, relative } from "node:path";

interface RepoContext {
  workflowMd: string | null;
  agentsMd: string | null;
  copilotInstructions: string | null;
  pathInstructions: PathInstruction[];
}

interface PathInstruction {
  glob: string;
  content: string;
  filePath: string;
}

/**
 * Load all repo-level context files from a repository root.
 */
export function loadRepoContext(repoRoot: string): RepoContext {
  return {
    workflowMd: readIfExists(join(repoRoot, "WORKFLOW.md")),
    agentsMd: readIfExists(join(repoRoot, "AGENTS.md")),
    copilotInstructions: readIfExists(
      join(repoRoot, ".github", "copilot-instructions.md"),
    ),
    pathInstructions: loadPathInstructions(repoRoot),
  };
}

/**
 * Build a context string for a workflow phase, filtering path instructions
 * to those that match the given changed files.
 */
export function buildContextPrompt(
  ctx: RepoContext,
  changedFiles?: string[],
): string {
  const sections: string[] = [];

  if (ctx.workflowMd) {
    sections.push("## Repository Workflow\n\n" + ctx.workflowMd);
  }

  if (ctx.agentsMd) {
    sections.push("## Agent Instructions\n\n" + ctx.agentsMd);
  }

  if (ctx.copilotInstructions) {
    sections.push("## Repository Instructions\n\n" + ctx.copilotInstructions);
  }

  // Filter path instructions to those matching changed files
  if (ctx.pathInstructions.length > 0 && changedFiles && changedFiles.length > 0) {
    const matching = ctx.pathInstructions.filter((pi) => {
      return changedFiles.some((f) => matchesGlob(normalizePath(f), normalizePath(pi.glob)));
    });

    if (matching.length > 0) {
      const parts = matching.map(
        (pi) => `### Instructions for \`${pi.glob}\`\n\n${pi.content}`,
      );
      sections.push("## Path-Specific Instructions\n\n" + parts.join("\n\n"));
    }
  }

  return sections.join("\n\n---\n\n");
}

// ─── Internal ────────────────────────────────────────────────

function readIfExists(path: string): string | null {
  if (!existsSync(path)) return null;
  try {
    return readFileSync(path, "utf-8");
  } catch {
    return null;
  }
}

/**
 * Load .github/instructions/**\/*.instructions.md files.
 * Each file's frontmatter `applyTo` or filename determines the glob pattern.
 */
function loadPathInstructions(repoRoot: string): PathInstruction[] {
  const instructionsDir = join(repoRoot, ".github", "instructions");
  if (!existsSync(instructionsDir)) return [];

  const results: PathInstruction[] = [];
  for (const match of collectInstructionFiles(instructionsDir)) {
    const filePath = join(instructionsDir, match);
    const content = readIfExists(filePath);
    if (!content) continue;

    // Extract glob from applyTo frontmatter or derive from filename
    const applyTo = extractApplyTo(content);
    const pattern = applyTo ?? deriveGlobFromFilename(match);

    results.push({
      glob: pattern,
      content: stripFrontmatter(content),
      filePath: relative(repoRoot, filePath),
    });
  }

  return results;
}

function collectInstructionFiles(rootDir: string, currentDir = rootDir): string[] {
  const matches: string[] = [];

  for (const entry of readdirSync(currentDir, { withFileTypes: true })) {
    const fullPath = join(currentDir, entry.name);
    if (entry.isDirectory()) {
      matches.push(...collectInstructionFiles(rootDir, fullPath));
      continue;
    }
    if (entry.isFile() && entry.name.endsWith(".instructions.md")) {
      matches.push(normalizePath(relative(rootDir, fullPath)));
    }
  }

  return matches;
}

function normalizePath(path: string): string {
  return path.replace(/\\/g, "/");
}

function matchesGlob(path: string, pattern: string): boolean {
  const normalizedPattern = normalizePath(pattern);
  const regex = new RegExp(`^${globToRegexSource(normalizedPattern)}$`);
  return regex.test(path);
}

function globToRegexSource(pattern: string): string {
  let source = "";

  for (let index = 0; index < pattern.length; index += 1) {
    const char = pattern[index]!;
    const next = pattern[index + 1];

    if (char === "*") {
      if (next === "*") {
        const afterNext = pattern[index + 2];
        if (afterNext === "/") {
          source += "(?:.*/)?";
          index += 2;
        } else {
          source += ".*";
          index += 1;
        }
      } else {
        source += "[^/]*";
      }
      continue;
    }

    if (char === "?") {
      source += "[^/]";
      continue;
    }

    source += escapeRegex(char);
  }

  return source;
}

function escapeRegex(char: string): string {
  return /[|\\{}()[\]^$+?.]/.test(char) ? `\\${char}` : char;
}

function extractApplyTo(content: string): string | null {
  const match = content.match(/^---\r?\n[\s\S]*?applyTo:\s*["']?([^\n"']+)["']?[\s\S]*?\r?\n---/);
  return match ? match[1]!.trim() : null;
}

function stripFrontmatter(content: string): string {
  return content.replace(/^---\r?\n[\s\S]*?\r?\n---\r?\n/, "").trim();
}

function deriveGlobFromFilename(filename: string): string {
  // foo.instructions.md → **/*foo*
  const base = filename.replace(/\.instructions\.md$/, "").replace(/\//g, ".");
  return `**/*${base}*`;
}
