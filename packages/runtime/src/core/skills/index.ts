/**
 * Skills subsystem — Agent Skills standard (agentskills.io) compatible.
 *
 * Modules:
 * - types: Type definitions and name validation
 * - loader: Filesystem discovery and SKILL.md parsing
 * - registry: In-memory store with async lazy loading
 * - resolver: Argument substitution and shell preprocessing
 */

export { isValidSkillName } from "./types.js";
export type {
  SkillFrontmatter,
  SkillSource,
  SkillMetadata,
  Skill,
  ResolvedSkill,
} from "./types.js";

export { SkillLoader } from "./loader.js";
export type { DiscoverOptions } from "./loader.js";

export { SkillRegistry } from "./registry.js";

export { SkillResolver } from "./resolver.js";
export type { ResolveContext, SkillResolverOptions } from "./resolver.js";

export {
  formatSkillMatchLine,
  formatSkillPromptGuidance,
} from "./prompt-format.js";

export {
  SkillAccessManager,
  INVOKED_SKILL_KNOWLEDGE_PREFIX,
} from "./access.js";
export type { SkillAccessPersistence } from "./access.js";
