/**
 * SkillLoader — Discovers and parses skills following the Agent Skills standard.
 * Scans multiple directory paths for <name>/SKILL.md skill directories.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, basename } from "node:path";
import { homedir } from "node:os";
import type { SkillMetadata, SkillSource, Skill } from "./types.js";
import { isValidSkillName } from "./types.js";

// ─── Discovery Options ──────────────────────────────────────

export interface DiscoverOptions {
  readonly repoRoot: string;
  /** Override global skill paths (for testing). */
  readonly globalPaths?: ReadonlyArray<string>;
}

// ─── Frontmatter Parser ─────────────────────────────────────

interface ParsedFrontmatter {
  readonly fields: Record<string, string>;
  readonly body: string;
}

function parseFrontmatter(content: string): ParsedFrontmatter {
  const trimmed = content.trimStart();
  if (!trimmed.startsWith("---")) {
    return { fields: {}, body: content };
  }

  const endIndex = trimmed.indexOf("---", 3);
  if (endIndex === -1) {
    return { fields: {}, body: content };
  }

  const frontmatter = trimmed.substring(3, endIndex).trim();
  const body = trimmed.substring(endIndex + 3).trim();
  const fields: Record<string, string> = {};

  for (const line of frontmatter.split("\n")) {
    const colonIdx = line.indexOf(":");
    if (colonIdx > 0) {
      const key = line.substring(0, colonIdx).trim();
      const value = line.substring(colonIdx + 1).trim();
      fields[key] = value.replace(/^["']|["']$/g, "");
    }
  }

  return { fields, body };
}

// ─── SkillLoader Class ──────────────────────────────────────

export class SkillLoader {
  /**
   * Discover skills from standard paths. Returns metadata only (no body loading).
   * Paths scanned in priority order (later overrides earlier):
   *   1. Global: ~/.config/devagent/skills/, ~/.agents/skills/, ~/.claude/skills/
   *   2. Project: .agents/skills/, .github/skills/, .claude/skills/, .devagent/skills/
   */
  discover(options: DiscoverOptions): SkillMetadata[] {
    const { repoRoot } = options;
    const found = new Map<string, SkillMetadata>();

    // Global paths (lowest priority)
    const globalPaths: Array<{ path: string; source: SkillSource }> =
      options.globalPaths
        ? options.globalPaths.map((p) => ({ path: p, source: "global" as const }))
        : [
            { path: join(homedir(), ".config", "devagent", "skills"), source: "global" },
            { path: join(homedir(), ".agents", "skills"), source: "global" },
            { path: join(homedir(), ".claude", "skills"), source: "global" },
          ];

    for (const { path, source } of globalPaths) {
      this.scanSkillsDirectory(path, source, found);
    }

    // Project paths (higher priority — overrides global)
    const projectPaths: Array<{ path: string; source: SkillSource }> = [
      { path: join(repoRoot, ".agents", "skills"), source: "project" },
      { path: join(repoRoot, ".github", "skills"), source: "project" },
      { path: join(repoRoot, ".claude", "skills"), source: "claude-compat" },
      { path: join(repoRoot, ".devagent", "skills"), source: "project" },
    ];

    for (const { path, source } of projectPaths) {
      this.scanSkillsDirectory(path, source, found);
    }

    return Array.from(found.values());
  }

  /**
   * Load full skill content (SKILL.md body + supporting directory detection).
   */
  loadSkillContent(metadata: SkillMetadata): Skill {
    const content = readFileSync(metadata.skillFilePath, "utf-8");
    const { body } = parseFrontmatter(content);

    return {
      ...metadata,
      instructions: body,
      hasScripts: existsSync(join(metadata.dirPath, "scripts")),
      hasReferences: existsSync(join(metadata.dirPath, "references")),
      hasAssets: existsSync(join(metadata.dirPath, "assets")),
    };
  }

  // ─── Private ────────────────────────────────────────────────

  private scanSkillsDirectory(
    dir: string,
    source: SkillSource,
    found: Map<string, SkillMetadata>,
  ): void {
    if (!existsSync(dir)) return;

    let entries: string[];
    try {
      entries = readdirSync(dir);
    } catch {
      return;
    }

    for (const entry of entries) {
      const entryPath = join(dir, entry);

      try {
        if (!statSync(entryPath).isDirectory()) continue;
      } catch {
        continue;
      }

      const skillFilePath = join(entryPath, "SKILL.md");
      if (!existsSync(skillFilePath)) continue;

      let content: string;
      try {
        content = readFileSync(skillFilePath, "utf-8");
      } catch {
        process.stderr.write(`[skills] Warning: cannot read ${skillFilePath}\n`);
        continue;
      }

      const { fields } = parseFrontmatter(content);
      const name = fields["name"];
      const description = fields["description"];

      if (!name || !description) {
        process.stderr.write(
          `[skills] Warning: ${skillFilePath} missing required name/description frontmatter\n`,
        );
        continue;
      }

      const dirName = basename(entryPath);
      if (name !== dirName) {
        process.stderr.write(
          `[skills] Warning: skill name "${name}" does not match directory "${dirName}" in ${skillFilePath}\n`,
        );
        continue;
      }

      if (!isValidSkillName(name)) {
        process.stderr.write(
          `[skills] Warning: invalid skill name "${name}" in ${skillFilePath}\n`,
        );
        continue;
      }

      const metadata: SkillMetadata = {
        name,
        description,
        source,
        dirPath: entryPath,
        skillFilePath,
        license: fields["license"],
        compatibility: fields["compatibility"]
          ? fields["compatibility"].split(",").map((s) => s.trim())
          : undefined,
        metadata: undefined,
      };

      found.set(name, metadata);
    }
  }
}
