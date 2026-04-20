/**
 * SkillLoader — Discovers and parses skills following the Agent Skills standard.
 * Scans multiple directory paths for <name>/SKILL.md skill directories.
 */

import { existsSync, readdirSync, readFileSync, statSync } from "node:fs";
import { join, basename } from "node:path";

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
  readonly fields: Record<string, string | ReadonlyArray<string>>;
  readonly body: string;
}

interface ParsedSkillSupportRoots {
  readonly supportRootPath: string;
  readonly sourceRepoPath?: string;
  readonly sourceSkillDirPath?: string;
}

function defaultGlobalSkillPaths(): ReadonlyArray<string> {
  const home = process.env["HOME"];
  return home ? [join(home, ".agents", "skills")] : [];
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
  const fields: Record<string, string | ReadonlyArray<string>> = {};
  let activeListKey: string | null = null;

  for (const rawLine of frontmatter.split("\n")) {
    const line = rawLine.trimEnd();
    const trimmed = line.trim();
    if (trimmed.length === 0 || trimmed.startsWith("#")) {
      continue;
    }

    if (activeListKey && trimmed.startsWith("- ")) {
      const nextValue = trimmed.slice(2).trim().replace(/^["']|["']$/g, "");
      const existing = fields[activeListKey];
      if (Array.isArray(existing)) {
        fields[activeListKey] = [...existing, nextValue];
      } else if (typeof existing === "string" && existing.length > 0) {
        fields[activeListKey] = [existing, nextValue];
      } else {
        fields[activeListKey] = [nextValue];
      }
      continue;
    }

    const colonIdx = trimmed.indexOf(":");
    if (colonIdx <= 0) {
      activeListKey = null;
      continue;
    }

    const key = trimmed.substring(0, colonIdx).trim();
    const value = trimmed.substring(colonIdx + 1).trim();
    if (value.length === 0) {
      fields[key] = [];
      activeListKey = key;
      continue;
    }

    fields[key] = value.replace(/^["']|["']$/g, "");
    activeListKey = null;
  }

  return { fields, body };
}

function normalizeStringArray(
  value: string | ReadonlyArray<string> | undefined,
): ReadonlyArray<string> | undefined {
  if (value === undefined) {
    return undefined;
  }

  let items: ReadonlyArray<string>;
  if (Array.isArray(value)) {
    items = value;
  } else if (typeof value === "string") {
    items = value.split(",").map((entry: string) => entry.trim());
  } else {
    return undefined;
  }
  const normalized = items
    .map((entry: string) => entry.trim())
    .filter((entry: string) => entry.length > 0);

  return normalized.length > 0 ? normalized : undefined;
}

function normalizeOptionalString(
  value: string | ReadonlyArray<string> | unknown,
): string | undefined {
  if (Array.isArray(value)) {
    return undefined;
  }
  if (typeof value !== "string") {
    return undefined;
  }
  const trimmed = value.trim();
  return trimmed.length > 0 ? trimmed : undefined;
}

function parseSkillSupportRoots(skillDirPath: string): ParsedSkillSupportRoots {
  const metadataPath = join(skillDirPath, ".arkts-agent-kit-source.json");
  if (!existsSync(metadataPath)) {
    return { supportRootPath: skillDirPath };
  }

  let parsed: unknown;
  try {
    parsed = JSON.parse(readFileSync(metadataPath, "utf-8"));
  } catch {
    return { supportRootPath: skillDirPath };
  }

  if (!parsed || typeof parsed !== "object") {
    return { supportRootPath: skillDirPath };
  }

  const sourceRepoPath = normalizeOptionalString(
    (parsed as Record<string, unknown>)["source_repo"],
  );
  const sourceSkillDirPath = normalizeOptionalString(
    (parsed as Record<string, unknown>)["source_dir"],
  );

  return {
    supportRootPath: sourceRepoPath ?? skillDirPath,
    sourceRepoPath,
    sourceSkillDirPath,
  };
}

// ─── SkillLoader Class ──────────────────────────────────────

export class SkillLoader {
  /**
   * Discover skills from standard paths. Returns metadata only (no body loading).
   * Paths scanned in priority order (later overrides earlier):
   *   1. Global: ~/.agents/skills/
   *   2. Project: .agents/skills/
   */
  discover(options: DiscoverOptions): SkillMetadata[] {
    const { repoRoot } = options;
    const found = new Map<string, SkillMetadata>();

    // Global paths (lowest priority)
    const globalPaths: Array<{ path: string; source: SkillSource }> =
      options.globalPaths
        ? options.globalPaths.map((p) => ({ path: p, source: "global" as const }))
        : defaultGlobalSkillPaths().map((path) => ({ path, source: "global" as const }));

    for (const { path, source } of globalPaths) {
      this.scanSkillsDirectory(path, source, found);
    }

    // Project paths (higher priority — overrides global)
    const projectPaths: Array<{ path: string; source: SkillSource }> = [
      { path: join(repoRoot, ".agents", "skills"), source: "project" },
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
    const roots = [metadata.dirPath];
    if (
      metadata.supportRootPath &&
      metadata.supportRootPath !== metadata.dirPath
    ) {
      roots.push(metadata.supportRootPath);
    }

    return {
      ...metadata,
      instructions: body,
      hasScripts: roots.some((root) => existsSync(join(root, "scripts"))),
      hasReferences: roots.some((root) => existsSync(join(root, "references"))),
      hasAssets: roots.some((root) => existsSync(join(root, "assets"))),
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
      const name = normalizeOptionalString(fields["name"]);
      const description = normalizeOptionalString(fields["description"]);

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

      const supportRoots = parseSkillSupportRoots(entryPath);

      const metadata: SkillMetadata = {
        name,
        description,
        triggers: normalizeStringArray(fields["triggers"]),
        paths: normalizeStringArray(fields["paths"]),
        examples: normalizeStringArray(fields["examples"]),
        source,
        dirPath: entryPath,
        skillFilePath,
        supportRootPath: supportRoots.supportRootPath,
        sourceRepoPath: supportRoots.sourceRepoPath,
        sourceSkillDirPath: supportRoots.sourceSkillDirPath,
        license: normalizeOptionalString(fields["license"]),
        compatibility: normalizeStringArray(fields["compatibility"]),
        metadata: undefined,
      };

      found.set(name, metadata);
    }
  }
}
