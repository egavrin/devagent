/**
 * Skills system — Markdown-based skill discovery and loading.
 * Skills are Markdown files with YAML frontmatter stored in:
 *   - .devagent/skills/           (project-local)
 *   - ~/.config/devagent/skills/  (global)
 *
 * Progressive loading: metadata first, full instructions on invocation.
 * ArkTS-compatible: no `any`, explicit types.
 */

import { existsSync, readdirSync, readFileSync } from "node:fs";
import { join, resolve, basename } from "node:path";
import { homedir } from "node:os";

// ─── Skill Types ─────────────────────────────────────────────

export interface SkillMetadata {
  readonly name: string;
  readonly description: string;
  readonly source: "project" | "global";
  readonly filePath: string;
}

export interface Skill extends SkillMetadata {
  readonly instructions: string;
}

// ─── YAML Frontmatter Parser ─────────────────────────────────

/**
 * Simple YAML frontmatter parser. Extracts `name` and `description`
 * from the `---` delimited frontmatter block.
 * No dependency on external YAML libraries.
 */
function parseFrontmatter(content: string): {
  metadata: Record<string, string>;
  body: string;
} {
  const trimmed = content.trimStart();
  if (!trimmed.startsWith("---")) {
    return { metadata: {}, body: content };
  }

  const endIndex = trimmed.indexOf("---", 3);
  if (endIndex === -1) {
    return { metadata: {}, body: content };
  }

  const frontmatter = trimmed.substring(3, endIndex).trim();
  const body = trimmed.substring(endIndex + 3).trim();
  const metadata: Record<string, string> = {};

  for (const line of frontmatter.split("\n")) {
    const colonIdx = line.indexOf(":");
    if (colonIdx > 0) {
      const key = line.substring(0, colonIdx).trim();
      const value = line.substring(colonIdx + 1).trim();
      // Remove quotes if present
      metadata[key] = value.replace(/^["']|["']$/g, "");
    }
  }

  return { metadata, body };
}

// ─── Skill Registry ──────────────────────────────────────────

export class SkillRegistry {
  private readonly skills = new Map<string, SkillMetadata>();
  private readonly cache = new Map<string, Skill>();

  /**
   * Discover skills from project and global directories.
   * Project skills override global skills with the same name.
   */
  discover(repoRoot: string): void {
    this.skills.clear();
    this.cache.clear();

    // Global skills (lower priority)
    const globalDir = join(homedir(), ".config", "devagent", "skills");
    this.scanDirectory(globalDir, "global");

    // Project skills (higher priority — overrides global)
    const projectDir = join(repoRoot, ".devagent", "skills");
    this.scanDirectory(projectDir, "project");
  }

  /**
   * Scan a directory for .md skill files and register their metadata.
   */
  private scanDirectory(dir: string, source: "project" | "global"): void {
    if (!existsSync(dir)) return;

    let entries: string[];
    try {
      entries = readdirSync(dir);
    } catch {
      return; // Directory not readable
    }

    for (const entry of entries) {
      if (!entry.endsWith(".md")) continue;

      const filePath = join(dir, entry);
      try {
        const content = readFileSync(filePath, "utf-8");
        const { metadata } = parseFrontmatter(content);

        const name = metadata["name"] ?? basename(entry, ".md");
        const description = metadata["description"] ?? "";

        this.skills.set(name, {
          name,
          description,
          source,
          filePath,
        });
      } catch {
        // Skip unreadable files — fail fast at invoke time, not discovery
      }
    }
  }

  /**
   * Get metadata for all discovered skills.
   */
  list(): ReadonlyArray<SkillMetadata> {
    return Array.from(this.skills.values());
  }

  /**
   * Check if a skill exists.
   */
  has(name: string): boolean {
    return this.skills.has(name);
  }

  /**
   * Get skill metadata without loading full content.
   */
  getMetadata(name: string): SkillMetadata | undefined {
    return this.skills.get(name);
  }

  /**
   * Load a skill's full content (instructions).
   * Caches loaded content for subsequent calls.
   * Throws if skill not found.
   */
  load(name: string): Skill {
    // Check cache first
    const cached = this.cache.get(name);
    if (cached) return cached;

    const meta = this.skills.get(name);
    if (!meta) {
      const available = Array.from(this.skills.keys()).join(", ");
      throw new Error(
        `Skill "${name}" not found. Available: ${available || "none"}`,
      );
    }

    const content = readFileSync(meta.filePath, "utf-8");
    const { body } = parseFrontmatter(content);

    const skill: Skill = {
      ...meta,
      instructions: body,
    };

    this.cache.set(name, skill);
    return skill;
  }

  /**
   * Get the number of discovered skills.
   */
  get size(): number {
    return this.skills.size;
  }

  /**
   * Clear all discovered skills and cache.
   */
  clear(): void {
    this.skills.clear();
    this.cache.clear();
  }
}
