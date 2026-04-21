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

const SKILL_SOURCE_METADATA_FILENAME = ".devagent-skill-source.json";

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
    const parsed = parseFrontmatterLine(rawLine, activeListKey);
    if (parsed.kind === "skip") continue;
    if (parsed.kind === "list-item") {
      appendFrontmatterListItem(fields, parsed.key, parsed.value);
      continue;
    }
    if (parsed.kind === "invalid") {
      activeListKey = null;
      continue;
    }

    fields[parsed.key] = buildEmptyListValue(parsed.value);
    activeListKey = parsed.value.length === 0 ? parsed.key : null;
  }

  return { fields, body };
}

type ParsedFrontmatterLine =
  | { readonly kind: "skip" }
  | { readonly kind: "invalid" }
  | { readonly kind: "list-item"; readonly key: string; readonly value: string }
  | { readonly kind: "field"; readonly key: string; readonly value: string };

function parseFrontmatterLine(
  rawLine: string,
  activeListKey: string | null,
): ParsedFrontmatterLine {
  const trimmed = rawLine.trimEnd().trim();
  if (trimmed.length === 0 || trimmed.startsWith("#")) return { kind: "skip" };
  if (activeListKey && trimmed.startsWith("- ")) {
    return {
      kind: "list-item",
      key: activeListKey,
      value: stripQuotes(trimmed.slice(2).trim()),
    };
  }

  const colonIdx = trimmed.indexOf(":");
  if (colonIdx <= 0) return { kind: "invalid" };

  const key = trimmed.substring(0, colonIdx).trim();
  const value = trimmed.substring(colonIdx + 1).trim();
  return { kind: "field", key, value: stripQuotes(value) };
}

function appendFrontmatterListItem(
  fields: Record<string, string | ReadonlyArray<string>>,
  key: string,
  value: string,
): void {
  const existing = fields[key];
  if (Array.isArray(existing)) {
    fields[key] = [...existing, value];
    return;
  }
  fields[key] = typeof existing === "string" && existing.length > 0
    ? [existing, value]
    : [value];
}

function stripQuotes(value: string): string {
  return value.replace(/^["']|["']$/g, "");
}

function buildEmptyListValue(value: string): string | ReadonlyArray<string> {
  return value.length === 0 ? [] : value;
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
  const metadataPath = join(skillDirPath, SKILL_SOURCE_METADATA_FILENAME);
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
      const metadata = this.scanSkillEntry(dir, entry, source);
      if (metadata) found.set(metadata.name, metadata);
    }
  }

  private scanSkillEntry(
    dir: string,
    entry: string,
    source: SkillSource,
  ): SkillMetadata | null {
    const entryPath = join(dir, entry);
    if (!isDirectory(entryPath)) return null;

    const skillFilePath = join(entryPath, "SKILL.md");
    if (!existsSync(skillFilePath)) return null;

    const content = readSkillFile(skillFilePath);
    if (content === null) return null;

    const { fields } = parseFrontmatter(content);
    const identity = validateSkillIdentity(entryPath, skillFilePath, fields);
    if (!identity) return null;

    const supportRoots = parseSkillSupportRoots(entryPath);
    return buildSkillMetadata({
      fields,
      identity,
      source,
      entryPath,
      skillFilePath,
      supportRoots,
    });
  }
}

function isDirectory(path: string): boolean {
  try {
    return statSync(path).isDirectory();
  } catch {
    return false;
  }
}

function readSkillFile(skillFilePath: string): string | null {
  try {
    return readFileSync(skillFilePath, "utf-8");
  } catch {
    process.stderr.write(`[skills] Warning: cannot read ${skillFilePath}\n`);
    return null;
  }
}

function validateSkillIdentity(
  entryPath: string,
  skillFilePath: string,
  fields: Record<string, string | ReadonlyArray<string>>,
): { readonly name: string; readonly description: string } | null {
  const name = normalizeOptionalString(fields["name"]);
  const description = normalizeOptionalString(fields["description"]);
  if (!name || !description) {
    process.stderr.write(
      `[skills] Warning: ${skillFilePath} missing required name/description frontmatter\n`,
    );
    return null;
  }
  if (!validateSkillDirectoryName(name, entryPath, skillFilePath)) return null;
  if (!validateSkillName(name, skillFilePath)) return null;
  return { name, description };
}

function validateSkillDirectoryName(
  name: string,
  entryPath: string,
  skillFilePath: string,
): boolean {
  const dirName = basename(entryPath);
  if (name === dirName) return true;
  process.stderr.write(
    `[skills] Warning: skill name "${name}" does not match directory "${dirName}" in ${skillFilePath}\n`,
  );
  return false;
}

function validateSkillName(name: string, skillFilePath: string): boolean {
  if (isValidSkillName(name)) return true;
  process.stderr.write(`[skills] Warning: invalid skill name "${name}" in ${skillFilePath}\n`);
  return false;
}

function buildSkillMetadata(input: {
  readonly fields: Record<string, string | ReadonlyArray<string>>;
  readonly identity: { readonly name: string; readonly description: string };
  readonly source: SkillSource;
  readonly entryPath: string;
  readonly skillFilePath: string;
  readonly supportRoots: ParsedSkillSupportRoots;
}): SkillMetadata {
  return {
    name: input.identity.name,
    description: input.identity.description,
    triggers: normalizeStringArray(input.fields["triggers"]),
    paths: normalizeStringArray(input.fields["paths"]),
    examples: normalizeStringArray(input.fields["examples"]),
    source: input.source,
    dirPath: input.entryPath,
    skillFilePath: input.skillFilePath,
    supportRootPath: input.supportRoots.supportRootPath,
    sourceRepoPath: input.supportRoots.sourceRepoPath,
    sourceSkillDirPath: input.supportRoots.sourceSkillDirPath,
    license: normalizeOptionalString(input.fields["license"]),
    compatibility: normalizeStringArray(input.fields["compatibility"]),
    metadata: undefined,
  };
}
