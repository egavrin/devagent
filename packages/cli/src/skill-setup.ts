import {
  SkillLoader,
  SkillRegistry,
  SkillResolver,
  SkillAccessManager,
  INVOKED_SKILL_KNOWLEDGE_PREFIX,
  type SessionState,
  createToolSearchTool,
  probeShellTools,
  formatProbeResults,
} from "@devagent/runtime";
import {
  createDefaultToolRegistry,
  createFindFilesTool,
  createReadFileTool,
  createSearchFilesTool,
  type ToolRegistry,
} from "@devagent/runtime";

interface SkillInfrastructure {
  readonly skills: SkillRegistry;
  readonly skillResolver: SkillResolver;
  readonly skillAccess: SkillAccessManager;
  readonly toolRegistry: ToolRegistry;
}

interface SkillInfrastructureOptions {
  readonly additionalDeferredToolNames?: ReadonlySet<string>;
}

/** Tools deferred by default — loaded on demand via tool_search. */
const DEFAULT_DEFERRED_TOOLS = new Set([
  "git_status",
  "git_diff",
  "git_commit",
]);

export function createSkillInfrastructure(
  projectRoot: string,
  sessionState: SessionState,
  options?: SkillInfrastructureOptions,
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
  const deferredToolNames = new Set(DEFAULT_DEFERRED_TOOLS);
  for (const toolName of options?.additionalDeferredToolNames ?? []) {
    deferredToolNames.add(toolName);
  }
  const toolRegistry = createDefaultToolRegistry({
    overrides: [
      createReadFileTool({ skillAccess }),
      createFindFilesTool({ skillAccess }),
      createSearchFilesTool({ skillAccess }),
    ],
    deferredToolNames,
  });

  // Register tool_search so the LLM can discover deferred tools on demand
  toolRegistry.register(createToolSearchTool(toolRegistry));

  // Probe shell tool availability and store as env fact
  try {
    const probeResults = probeShellTools();
    const summary = formatProbeResults(probeResults);
    if (summary) {
      sessionState.addEnvFact("shell-tools", summary);
    }
  } catch {
    // Non-fatal: shell probing failure shouldn't block startup
  }

  return {
    skills,
    skillResolver,
    skillAccess,
    toolRegistry,
  };
}
