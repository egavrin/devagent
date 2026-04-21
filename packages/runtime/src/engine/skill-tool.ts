/**
 * invoke_skill — LLM-callable tool to load skill instructions on demand.
 * Follows the same factory pattern as the other runtime tool constructors.
 */

import type { ToolContext, ToolSpec , SkillAccessManager, SkillRegistry, SkillResolver } from "../core/index.js";
import { extractErrorMessage } from "../core/index.js";

interface SkillToolOptions {
  readonly skillAccess?: SkillAccessManager;
}

const SKILL_TOOL_PARAM_SCHEMA = {
  type: "object",
  properties: {
    name: {
      type: "string",
      description: "The skill name to invoke (must match a discovered skill name)",
    },
    arguments: {
      type: "string",
      description: "Optional arguments to pass to the skill (substituted into $ARGUMENTS, $0, $1, etc.)",
    },
  },
  required: ["name"],
};

const SKILL_TOOL_RESULT_SCHEMA = {
  type: "object",
  properties: {
    instructions: { type: "string" },
    skillName: { type: "string" },
    skillDir: { type: "string" },
    hasScripts: { type: "boolean" },
    hasReferences: { type: "boolean" },
    hasAssets: { type: "boolean" },
  },
};

export function createSkillTool(
  registry: SkillRegistry,
  resolver: SkillResolver,
  options?: SkillToolOptions,
): ToolSpec {
  return {
    name: "invoke_skill",
    description:
      "Load a skill's full instructions. Skills are reusable instruction sets that " +
      "guide how to approach specific tasks. When a user's task matches an available " +
      "skill (listed in 'Available Skills'), invoke this tool to load its guidance " +
      "before proceeding. Always invoke a relevant skill before starting work.",
    category: "readonly",
    errorGuidance: {
      common: "Check the skill name matches one listed in 'Available Skills'. Use invoke_skill with the exact skill name.",
      patterns: [
        {
          match: "not found",
          hint: "The skill name was not found. List available skills by checking the 'Available Skills' section in your system prompt.",
        },
      ],
    },
    paramSchema: SKILL_TOOL_PARAM_SCHEMA,
    resultSchema: SKILL_TOOL_RESULT_SCHEMA,
    handler: async (params, context) => invokeSkill(registry, resolver, options, params, context),
  };
}

async function invokeSkill(
  registry: SkillRegistry,
  resolver: SkillResolver,
  options: SkillToolOptions | undefined,
  params: Record<string, unknown>,
  context: ToolContext,
) {
  const name = params["name"] as string | undefined;
  if (!name) return missingSkillNameResult();

  try {
    const skill = await registry.load(name);
    const resolved = await resolver.resolve(skill, (params["arguments"] as string | undefined) ?? "", {
      sessionId: context.sessionId,
      allowShellPreprocess: skill.source === "project",
    });
    options?.skillAccess?.unlock(skill.name);

    return {
      success: true,
      output: formatSkillInstructions(skill, resolved.resolvedInstructions, options),
      error: null,
      artifacts: [],
    };
  } catch (err) {
    return {
      success: false,
      output: "",
      error: extractErrorMessage(err),
      artifacts: [],
    };
  }
}

function missingSkillNameResult() {
  return {
    success: false,
    output: "",
    error: "Missing required parameter: name",
    artifacts: [],
  };
}

function formatSkillInstructions(
  skill: Awaited<ReturnType<SkillRegistry["load"]>>,
  resolvedInstructions: string,
  options: SkillToolOptions | undefined,
) {
  const header = `# Skill: ${skill.name}\n\n`;
  const meta = buildSkillMetadata(skill, options);
  return `${header}${meta}\n\n---\n\n${resolvedInstructions}`;
}

function buildSkillMetadata(
  skill: Awaited<ReturnType<SkillRegistry["load"]>>,
  options: SkillToolOptions | undefined,
) {
  return [
    `**Source:** ${skill.source}`,
    `**Directory:** ${skill.dirPath}`,
    skill.hasScripts ? "**Scripts:** available in scripts/" : null,
    skill.hasReferences ? "**References:** available in references/" : null,
    skill.hasAssets ? "**Assets:** available in assets/" : null,
    hasBackedSupportRoot(skill) ? "**Support files:** additional support content is available through the backing skill tree" : null,
    shouldShowSkillFileHint(skill, options)
      ? `**Skill files:** use \`read_file\`, \`find_files\`, or \`search_files\` with \`skill://${skill.name}/...\``
      : null,
  ]
    .filter(Boolean)
    .join("\n");
}

function hasBackedSupportRoot(skill: Awaited<ReturnType<SkillRegistry["load"]>>) {
  return Boolean(skill.supportRootPath && skill.supportRootPath !== skill.dirPath);
}

function shouldShowSkillFileHint(
  skill: Awaited<ReturnType<SkillRegistry["load"]>>,
  options: SkillToolOptions | undefined,
) {
  const hasSupportFiles = skill.hasScripts || skill.hasReferences || skill.hasAssets;
  return Boolean((hasSupportFiles || hasBackedSupportRoot(skill)) && options?.skillAccess);
}
