/**
 * Feature development plugin — multi-phase workflow.
 * Command: /feature
 *
 * 7-phase workflow (from Claude Code inspiration):
 * 1. Understand — Parse the feature request
 * 2. Explore — Analyze codebase structure
 * 3. Plan — Design implementation approach
 * 4. Implement — Write code changes
 * 5. Verify — Run tests and validation
 * 6. Review — Self-review changes
 * 7. Summarize — Report results
 *
 * Each phase can optionally use subagents via the delegate tool.
 */

import type { Plugin, PluginContext, CommandHandler } from "@devagent/core";

export interface FeaturePhase {
  readonly name: string;
  readonly description: string;
  readonly prompt: string;
}

/**
 * Get the 7-phase workflow definition for a feature.
 */
export function getFeaturePhases(featureDescription: string, repoRoot: string): ReadonlyArray<FeaturePhase> {
  return [
    {
      name: "understand",
      description: "Parse and clarify the feature request",
      prompt: `Analyze this feature request and identify:
1. Core requirements (what must be built)
2. Acceptance criteria (how to verify it works)
3. Questions or ambiguities to resolve

Feature request: "${featureDescription}"
Working directory: ${repoRoot}`,
    },
    {
      name: "explore",
      description: "Analyze codebase structure and relevant files",
      prompt: `Explore the codebase to understand the architecture relevant to this feature.
Use find_files and search_files to identify:
1. Files that will need changes
2. Existing patterns to follow
3. Dependencies and interfaces to understand

Feature: "${featureDescription}"
Working directory: ${repoRoot}`,
    },
    {
      name: "plan",
      description: "Design implementation approach",
      prompt: `Based on your exploration, create a detailed implementation plan for this feature.
Include:
1. Files to create or modify (in order)
2. Key design decisions and patterns to follow
3. Testing strategy
4. Potential risks or edge cases

Feature: "${featureDescription}"`,
    },
    {
      name: "implement",
      description: "Write code changes",
      prompt: `Implement the feature according to the plan.
Use write_file and replace_in_file to make changes.
Follow existing code patterns and style.
Write clean, well-documented code.

Feature: "${featureDescription}"`,
    },
    {
      name: "verify",
      description: "Run tests and validation",
      prompt: `Verify the implementation:
1. Run existing tests to check for regressions
2. Verify the new code compiles without errors
3. Check that the feature works as expected

Use run_command to execute tests and validation commands.`,
    },
    {
      name: "review",
      description: "Self-review changes",
      prompt: `Review all changes made for this feature:
1. Use git_diff to see all changes
2. Check for code quality issues
3. Verify all acceptance criteria are met
4. Identify any missing edge cases or tests

Be critical and thorough.`,
    },
    {
      name: "summarize",
      description: "Report results",
      prompt: `Summarize what was accomplished for this feature:
1. What files were created or modified
2. Key design decisions made
3. Test results
4. Any known limitations or follow-up tasks

Feature: "${featureDescription}"`,
    },
  ];
}

const featureCommand: CommandHandler = {
  description: "Multi-phase feature development workflow",
  usage: '/feature "<description>"',

  async execute(args: string, context: PluginContext): Promise<string> {
    const featureDescription = args.trim().replace(/^["']|["']$/g, "");

    if (!featureDescription) {
      return 'Usage: /feature "<description>"\nProvide a feature description to start the workflow.';
    }

    const phases = getFeaturePhases(featureDescription, context.repoRoot);

    const output: string[] = [
      `## Feature Development Workflow`,
      `**Feature**: ${featureDescription}`,
      "",
      "### Phases:",
    ];

    for (let i = 0; i < phases.length; i++) {
      const phase = phases[i]!;
      output.push(`${i + 1}. **${phase.name}** — ${phase.description}`);
    }

    output.push("");
    output.push("To execute this workflow with the AI agent, use:");
    output.push(`\`devagent "Implement this feature using the 7-phase workflow: ${featureDescription}"\``);
    output.push("");
    output.push("Or run individual phases:");
    output.push(`\`devagent --plan "Explore and plan: ${featureDescription}"\``);
    output.push(`\`devagent "Implement: ${featureDescription}"\``);

    return output.join("\n");
  },
};

export function createFeatureDevPlugin(): Plugin {
  return {
    name: "feature-dev",
    version: "1.0.0",
    description: "Multi-phase feature development workflow",
    commands: { feature: featureCommand },
    activate() {
      // No event subscriptions needed
    },
  };
}
