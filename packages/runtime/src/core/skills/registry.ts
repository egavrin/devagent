/**
 * SkillRegistry — In-memory skill store with async lazy loading and caching.
 * Pure data store: no filesystem discovery logic (that's SkillLoader's job).
 */

import { SkillLoader } from "./loader.js";
import type { SkillMetadata, Skill } from "./types.js";

export class SkillRegistry {
  private readonly skills = new Map<string, SkillMetadata>();
  private readonly cache = new Map<string, Skill>();
  private readonly loader = new SkillLoader();

  /** Register discovered skill metadata. Replaces any existing entries. */
  register(metadata: ReadonlyArray<SkillMetadata>): void {
    for (const m of metadata) {
      this.skills.set(m.name, m);
    }
  }

  /** Get metadata for all registered skills. */
  list(): ReadonlyArray<SkillMetadata> {
    return Array.from(this.skills.values());
  }

  /** Check if a skill is registered. */
  has(name: string): boolean {
    return this.skills.has(name);
  }

  /** Get metadata without loading full content. */
  getMetadata(name: string): SkillMetadata | undefined {
    return this.skills.get(name);
  }

  /**
   * Load full skill content (SKILL.md body + supporting dirs).
   * Caches result for subsequent calls.
   * Throws if skill not found.
   */
  async load(name: string): Promise<Skill> {
    const cached = this.cache.get(name);
    if (cached) return cached;

    const meta = this.skills.get(name);
    if (!meta) {
      const available = Array.from(this.skills.keys()).join(", ");
      throw new Error(
        `Skill "${name}" not found. Available: ${available || "none"}`,
      );
    }

    const skill = this.loader.loadSkillContent(meta);
    this.cache.set(name, skill);
    return skill;
  }

  /** Number of registered skills. */
  get size(): number {
    return this.skills.size;
  }

  /** Clear all skills and cache. */
  clear(): void {
    this.skills.clear();
    this.cache.clear();
  }
}
