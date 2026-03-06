/**
 * invoke_skill — LLM-callable tool to load skill instructions on demand.
 * Follows the factory pattern used by createPlanTool, createMemoryTools, etc.
 */

import type { ToolSpec } from "@devagent/core";
import type { SkillRegistry, SkillResolver } from "@devagent/core";
import { extractErrorMessage } from "@devagent/core";

export function createSkillTool(
  registry: SkillRegistry,
  resolver: SkillResolver,
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

        const header = `# Skill: ${skill.name}\n\n`;
        const meta = [
          `**Source:** ${skill.source}`,
          `**Directory:** ${skill.dirPath}`,
          skill.hasScripts ? "**Scripts:** available in scripts/" : null,
          skill.hasReferences ? "**References:** available in references/" : null,
          skill.hasAssets ? "**Assets:** available in assets/" : null,
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
