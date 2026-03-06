/**
 * Skill types — Agent Skills standard (agentskills.io) compatible.
 */

// ─── Name Validation ─────────────────────────────────────────

/** Agent Skills standard: 1-64 chars, lowercase alphanumeric + hyphens,
 *  no leading/trailing/consecutive hyphens. */
const SKILL_NAME_REGEX = /^[a-z0-9](?:[a-z0-9-]*[a-z0-9])?$/;
const SKILL_NAME_MAX_LENGTH = 64;

export function isValidSkillName(name: string): boolean {
  return (
    name.length >= 1 &&
    name.length <= SKILL_NAME_MAX_LENGTH &&
    SKILL_NAME_REGEX.test(name) &&
    !name.includes("--")
  );
}

// ─── Types ───────────────────────────────────────────────────

/** Agent Skills standard frontmatter (agentskills.io). */
export interface SkillFrontmatter {
  readonly name: string;
  readonly description: string;
  readonly license?: string;
  readonly compatibility?: ReadonlyArray<string>;
  readonly metadata?: Readonly<Record<string, string>>;
}

/** Discovery source tracking. */
export type SkillSource = "project" | "claude-compat" | "global";

/** Metadata loaded at discovery time (cheap — no file body read). */
export interface SkillMetadata extends SkillFrontmatter {
  readonly source: SkillSource;
  readonly dirPath: string;
  readonly skillFilePath: string;
}

/** Full skill loaded on demand (reads SKILL.md body). */
export interface Skill extends SkillMetadata {
  readonly instructions: string;
  readonly hasScripts: boolean;
  readonly hasReferences: boolean;
  readonly hasAssets: boolean;
}

/** Resolved skill after argument + shell substitution. */
export interface ResolvedSkill extends Skill {
  readonly resolvedInstructions: string;
}
