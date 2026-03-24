import {
  SkillLoader,
  SkillRegistry,
  SkillResolver,
  SkillAccessManager,
  INVOKED_SKILL_KNOWLEDGE_PREFIX,
  type SessionState,
} from "@devagent/runtime";
import {
  createDefaultToolRegistry,
  createFindFilesTool,
  createReadFileTool,
  createSearchFilesTool,
  type ToolRegistry,
} from "@devagent/runtime";

export interface SkillInfrastructure {
  readonly skills: SkillRegistry;
  readonly skillResolver: SkillResolver;
  readonly skillAccess: SkillAccessManager;
  readonly toolRegistry: ToolRegistry;
}

export function createSkillInfrastructure(
  projectRoot: string,
  sessionState: SessionState,
): SkillInfrastructure {
  const skillLoader = new SkillLoader();
  const skillMetadata = skillLoader.discover({ repoRoot: projectRoot });
  const skills = new SkillRegistry();
  skills.register(skillMetadata);
  const skillResolver = new SkillResolver();
  const skillAccess = new SkillAccessManager(skills, {
    loadUnlockedSkillNames: () => sessionState.getKnowledge()
      .filter((entry) => entry.key.startsWith(INVOKED_SKILL_KNOWLEDGE_PREFIX))
      .map((entry) => entry.key.slice(INVOKED_SKILL_KNOWLEDGE_PREFIX.length)),
    persistUnlockedSkill: (skillName) => {
      sessionState.addKnowledge(
        `${INVOKED_SKILL_KNOWLEDGE_PREFIX}${skillName}`,
        `Unlocked skill support tree for ${skillName}`,
        0,
      );
    },
  });
  const toolRegistry = createDefaultToolRegistry({
    overrides: [
      createReadFileTool({ skillAccess }),
      createFindFilesTool({ skillAccess }),
      createSearchFilesTool({ skillAccess }),
    ],
  });

  return {
    skills,
    skillResolver,
    skillAccess,
    toolRegistry,
  };
}
