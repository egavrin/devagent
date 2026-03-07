import { readFileSync, existsSync, readdirSync } from "fs";
import { join, resolve, relative, dirname } from "path";

export interface RepoInstruction {
  source: string;
  path: string;
  content: string;
  scope: "workflow" | "agent" | "repo" | "path-specific" | "skill";
  priority: number; // lower = higher priority
}

/**
 * Load repository instructions with explicit precedence:
 * 1. WORKFLOW.md — orchestration contract (priority 0)
 * 2. nearest AGENTS.md — agent behavior (priority 1)
 * 3. .github/copilot-instructions.md — repo-wide guidance (priority 2)
 * 4. .github/instructions/**\/*.instructions.md — path-specific (priority 3)
 * 5. attached skills (priority 4, handled separately)
 */
export class RepositoryInstructionLoader {
  private repoRoot: string;

  constructor(repoRoot: string) {
    this.repoRoot = resolve(repoRoot);
  }

  load(): RepoInstruction[] {
    const instructions: RepoInstruction[] = [];

    // Priority 0: WORKFLOW.md
    const workflowPath = join(this.repoRoot, "WORKFLOW.md");
    if (existsSync(workflowPath)) {
      instructions.push({
        source: "WORKFLOW.md",
        path: workflowPath,
        content: readFileSync(workflowPath, "utf-8"),
        scope: "workflow",
        priority: 0,
      });
    }

    // Priority 1: AGENTS.md (root level)
    const agentsPath = join(this.repoRoot, "AGENTS.md");
    if (existsSync(agentsPath)) {
      instructions.push({
        source: "AGENTS.md",
        path: agentsPath,
        content: readFileSync(agentsPath, "utf-8"),
        scope: "agent",
        priority: 1,
      });
    }

    // Priority 2: .github/copilot-instructions.md
    const copilotPath = join(
      this.repoRoot,
      ".github",
      "copilot-instructions.md",
    );
    if (existsSync(copilotPath)) {
      instructions.push({
        source: "copilot-instructions.md",
        path: copilotPath,
        content: readFileSync(copilotPath, "utf-8"),
        scope: "repo",
        priority: 2,
      });
    }

    // Priority 3: .github/instructions/**/*.instructions.md
    const instructionsDir = join(this.repoRoot, ".github", "instructions");
    if (existsSync(instructionsDir)) {
      const files = this.walkInstructionFiles(instructionsDir);
      for (const filePath of files) {
        instructions.push({
          source: relative(this.repoRoot, filePath),
          path: filePath,
          content: readFileSync(filePath, "utf-8"),
          scope: "path-specific",
          priority: 3,
        });
      }
    }

    return instructions.sort((a, b) => a.priority - b.priority);
  }

  /**
   * Load instructions relevant to a specific file path.
   * Returns all non-path-specific instructions plus any path-specific
   * instructions whose glob pattern (derived from filename) matches
   * the given file path.
   */
  loadForPath(filePath: string): RepoInstruction[] {
    const all = this.load();
    const absPath = resolve(filePath);
    const relPath = relative(this.repoRoot, absPath);

    return all.filter((inst) => {
      if (inst.scope !== "path-specific") return true;
      // Path-specific instructions apply based on their location
      // within .github/instructions/. The directory structure mirrors
      // the repo, or the filename encodes a pattern.
      return this.pathSpecificMatches(inst, relPath);
    });
  }

  /**
   * Find the nearest AGENTS.md by walking up from a given directory.
   */
  findNearestAgentsMd(fromDir: string): RepoInstruction | null {
    let current = resolve(fromDir);
    const root = this.repoRoot;

    while (current.startsWith(root)) {
      const candidate = join(current, "AGENTS.md");
      if (existsSync(candidate)) {
        return {
          source: relative(root, candidate) || "AGENTS.md",
          path: candidate,
          content: readFileSync(candidate, "utf-8"),
          scope: "agent",
          priority: 1,
        };
      }
      const parent = dirname(current);
      if (parent === current) break;
      current = parent;
    }

    return null;
  }

  private walkInstructionFiles(dir: string): string[] {
    const results: string[] = [];
    try {
      const entries = readdirSync(dir, {
        withFileTypes: true,
        recursive: true,
      });
      for (const entry of entries) {
        if (entry.isFile() && entry.name.endsWith(".instructions.md")) {
          // In Node 20+ with recursive, parentPath/path contains the directory
          const parentDir =
            (entry as any).parentPath ?? (entry as any).path ?? dir;
          results.push(join(parentDir, entry.name));
        }
      }
    } catch {
      // Directory unreadable — return empty
    }
    return results.sort();
  }

  /**
   * Determine if a path-specific instruction applies to the given relative path.
   * Convention: the instruction file is located at
   *   .github/instructions/<pattern>.instructions.md
   * where <pattern> uses directory structure to scope. For example:
   *   .github/instructions/src/components.instructions.md
   * applies to files under src/components/.
   *
   * We extract the relative path within .github/instructions/ and check
   * if the target file's relative path starts with the instruction's
   * directory prefix (minus the .instructions.md filename).
   */
  private pathSpecificMatches(
    inst: RepoInstruction,
    relFilePath: string,
  ): boolean {
    // inst.source is e.g. ".github/instructions/src/components.instructions.md"
    const prefix = ".github/instructions/";
    const source = inst.source.replace(/\\/g, "/");
    if (!source.startsWith(prefix)) return false;

    const instructionRelPath = source.slice(prefix.length);
    const normalizedFilePath = relFilePath.replace(/\\/g, "/");
    const scopePath = instructionRelPath.replace(/\.instructions\.md$/, "");

    if (scopePath === "" || scopePath === ".") {
      // Top-level instruction file — applies to everything
      return true;
    }

    if (normalizedFilePath === scopePath) return true;
    return normalizedFilePath.startsWith(scopePath + "/");
  }
}
