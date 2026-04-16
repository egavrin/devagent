/**
 * invoke_skill — LLM-callable tool to load skill instructions on demand.
 * Follows the same factory pattern as the other runtime tool constructors.
 */

import type { ToolSpec } from "../core/index.js";
import type { SkillAccessManager, SkillRegistry, SkillResolver } from "../core/index.js";
import { extractErrorMessage } from "../core/index.js";

interface SkillToolOptions {
  readonly skillAccess?: SkillAccessManager;
}

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
    paramSchema: {
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
    },
    resultSchema: {
      type: "object",
      properties: {
        instructions: { type: "string" },
        skillName: { type: "string" },
        skillDir: { type: "string" },
        hasScripts: { type: "boolean" },
        hasReferences: { type: "boolean" },
        hasAssets: { type: "boolean" },
      },
    },
    handler: async (params, context) => {
      const name = params["name"] as string | undefined;
      if (!name) {
        return {
          success: false,
          output: "",
          error: "Missing required parameter: name",
          artifacts: [],
        };
      }

      const args = (params["arguments"] as string | undefined) ?? "";

      try {
        const skill = await registry.load(name);
        const resolved = await resolver.resolve(skill, args, {
          sessionId: context.sessionId,
          allowShellPreprocess: skill.source === "project",
        });
        options?.skillAccess?.unlock(skill.name);

        const header = `# Skill: ${skill.name}\n\n`;
        const hasSupportFiles = skill.hasScripts || skill.hasReferences || skill.hasAssets;
        const hasBackedSupportRoot = Boolean(
          skill.supportRootPath && skill.supportRootPath !== skill.dirPath,
        );
        const meta = [
          `**Source:** ${skill.source}`,
          `**Directory:** ${skill.dirPath}`,
          skill.hasScripts ? "**Scripts:** available in scripts/" : null,
          skill.hasReferences ? "**References:** available in references/" : null,
          skill.hasAssets ? "**Assets:** available in assets/" : null,
          hasBackedSupportRoot ? "**Support files:** additional support content is available through the backing skill tree" : null,
          (hasSupportFiles || hasBackedSupportRoot) && options?.skillAccess
            ? `**Skill files:** use \`read_file\`, \`find_files\`, or \`search_files\` with \`skill://${skill.name}/...\``
            : null,
        ]
          .filter(Boolean)
          .join("\n");

        const output = `${header}${meta}\n\n---\n\n${resolved.resolvedInstructions}`;

        return {
          success: true,
          output,
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
    },
  };
}
