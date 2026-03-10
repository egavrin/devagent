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

import { readFileSync, existsSync } from "node:fs";
import { join, relative } from "node:path";
import { Glob } from "bun";

export interface RepoContext {
  workflowMd: string | null;
  agentsMd: string | null;
  copilotInstructions: string | null;
  pathInstructions: PathInstruction[];
}

export interface PathInstruction {
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
      const glob = new Glob(pi.glob);
      return changedFiles.some((f) => glob.match(f));
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
  const glob = new Glob("**/*.instructions.md");

  for (const match of glob.scanSync({ cwd: instructionsDir })) {
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
