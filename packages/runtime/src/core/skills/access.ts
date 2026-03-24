import type { SkillMetadata } from "./types.js";
import { SkillRegistry } from "./registry.js";

export const INVOKED_SKILL_KNOWLEDGE_PREFIX = "invoked-skill:";

export interface SkillAccessPersistence {
  readonly loadUnlockedSkillNames?: () => ReadonlyArray<string>;
  readonly persistUnlockedSkill?: (skillName: string) => void;
}

export class SkillAccessManager {
  private readonly unlocked = new Set<string>();
  private hydrated = false;

  constructor(
    private readonly registry: SkillRegistry,
    private readonly persistence?: SkillAccessPersistence,
  ) {}

  unlock(skillName: string): void {
    const metadata = this.getKnownSkill(skillName);
    if (this.unlocked.has(metadata.name)) return;
    this.unlocked.add(metadata.name);
    this.persistence?.persistUnlockedSkill?.(metadata.name);
  }

  isUnlocked(skillName: string): boolean {
    this.hydrate();
    return this.unlocked.has(skillName);
  }

  requireUnlocked(skillName: string): SkillMetadata {
    const metadata = this.getKnownSkill(skillName);
    if (!this.unlocked.has(metadata.name)) {
      throw new Error(
        `Skill "${metadata.name}" is not unlocked. Call invoke_skill with this exact skill name first.`,
      );
    }
    return metadata;
  }

  listAvailable(): ReadonlyArray<string> {
    return this.registry.list().map((skill) => skill.name).sort();
  }

  private getKnownSkill(skillName: string): SkillMetadata {
    this.hydrate();
    const metadata = this.registry.getMetadata(skillName);
    if (metadata) {
      return metadata;
    }

    const available = this.listAvailable().join(", ");
    throw new Error(
      `Skill "${skillName}" not found. Available: ${available || "none"}`,
    );
  }

  private hydrate(): void {
    if (this.hydrated) {
      return;
    }
    this.hydrated = true;
    const restored = this.persistence?.loadUnlockedSkillNames?.() ?? [];
    for (const skillName of restored) {
      if (this.registry.has(skillName)) {
        this.unlocked.add(skillName);
      }
    }
  }
}
