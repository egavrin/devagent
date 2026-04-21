/**
 * System prompt assembly — composes modular prompt sections
 * from markdown files into a single system prompt.
 */

import { formatBriefing, formatDeferredToolsSection } from "@devagent/runtime";

import {
  PROMPT_BASE,
  PROMPT_TOOLS,
  PROMPT_MODE_ACT,
  PROMPT_REVIEW,
} from "./embedded.js";
import {
  buildRootPromptFragments,
  deriveRootPromptCapabilities,
} from "./fragments.js";
import { loadProjectContext } from "./project-context.js";
import type { SkillRegistry ,
  AgentType,
  ReasoningEffort,
  TaskMode,
  ToolSpec,
  TurnBriefing,
} from "@devagent/runtime";

interface AssemblePromptOptions {
  readonly mode: TaskMode;
  readonly repoRoot: string;
  readonly skills: SkillRegistry;
  readonly availableTools?: ReadonlyArray<Pick<ToolSpec, "name" | "category">>;
  /** Deferred tool stubs for prompt injection (available via tool_search). */
  readonly deferredTools?: ReadonlyArray<{ name: string; description: string }>;
  readonly approvalMode?: string;
  readonly provider?: string;
  readonly model?: string;
  readonly agentModelOverrides?: Partial<Record<AgentType, string>>;
  readonly agentReasoningOverrides?: Partial<Record<AgentType, ReasoningEffort>>;
  readonly includeReview?: boolean;
  /** Structured briefing from a prior turn (enables turn isolation). */
  readonly briefing?: TurnBriefing;
}

function formatOptionalSkillList(
  label: string,
  values: ReadonlyArray<string> | undefined,
  formatter: (value: string) => string = (value) => `"${value}"`,
): string | null {
  if (!values || values.length === 0) {
    return null;
  }
  return `${label}: ${values.map(formatter).join(", ")}`;
}

function formatSkillMatchLine(skill: {
  readonly name: string;
  readonly description: string;
  readonly source?: string;
  readonly triggers?: ReadonlyArray<string>;
  readonly paths?: ReadonlyArray<string>;
  readonly examples?: ReadonlyArray<string>;
}): string {
  const details = [
    formatOptionalSkillList("triggers", skill.triggers),
    formatOptionalSkillList("paths", skill.paths, (value) => `\`${value}\``),
    formatOptionalSkillList("examples", skill.examples),
  ].filter((detail): detail is string => Boolean(detail));

  const source = skill.source ?? "project";
  if (details.length === 0) {
    return `- \`${skill.name}\`: ${skill.description} (${source})`;
  }
  return `- \`${skill.name}\`: ${skill.description} (${source}; ${details.join("; ")})`;
}

function formatSkillPromptGuidance(): string {
  return [
    "Match skills using user intent, touched paths, and expected output shape.",
    "Invoke the broadest relevant workflow skill first, then specialist follow-up skills only when the task clearly enters that area.",
    "Precedence examples: `surface-change-e2e` before `validate-user-surface`; `provider-adapter-change` before `security-checklist`; `release-train` before `validate-user-surface`.",
  ].join(" ");
}

/**
 * Assemble the full system prompt from static markdown sections plus
 * capability-aware/task-shape fragments.
 */
export function assembleSystemPrompt(opts: AssemblePromptOptions): string {
  const sections: string[] = [];
  const capabilities = deriveRootPromptCapabilities(opts.availableTools);

  // Core identity and behavior
  sections.push(PROMPT_BASE);

  // Generic tool usage strategy
  sections.push(PROMPT_TOOLS);

  // Mode-specific constraints
  sections.push(PROMPT_MODE_ACT);

  // Capability-aware task-shape and tool-specific guidance
  sections.push(
    ...buildRootPromptFragments({
      mode: opts.mode,
      capabilities,
      agentModelOverrides: opts.agentModelOverrides,
      agentReasoningOverrides: opts.agentReasoningOverrides,
    }),
  );

  // Review guidelines (loaded conditionally)
  if (opts.includeReview) {
    sections.push(PROMPT_REVIEW);
  }

  // Environment context
  const envLines = [
    `Working directory: ${opts.repoRoot}`,
    `Task mode: ${opts.mode}`,
  ];
  if (opts.approvalMode) {
    envLines.push(`Safety mode: ${opts.approvalMode}`);
  }
  if (opts.provider && opts.model) {
    envLines.push(`Provider: ${opts.provider} / ${opts.model}`);
  } else if (opts.provider) {
    envLines.push(`Provider: ${opts.provider}`);
  } else if (opts.model) {
    envLines.push(`Model: ${opts.model}`);
  }
  envLines.push(`Date: ${new Date().toISOString().split("T")[0]}`);
  sections.push(`## Environment\n\n${envLines.join("\n")}`);

  // Project-level instructions (AGENTS.md, CLAUDE.md, etc.)
  const projectContext = loadProjectContext(opts.repoRoot);
  if (projectContext) {
    sections.push(projectContext);
  }

  // Skills (Agent Skills standard)
  const skillList = opts.skills.list();
  if (skillList.length > 0) {
    const skillLines = skillList
      .map((s) => formatSkillMatchLine(s))
      .join("\n");
    sections.push(
      `## Available Skills\n\n${skillLines}\n\n` +
      `Use the \`invoke_skill\` tool to load a skill's full instructions when the ` +
      `user's task matches an available skill. ${formatSkillPromptGuidance()}`,
    );
  }

  if (opts.deferredTools && opts.deferredTools.length > 0) {
    sections.push(formatDeferredToolsSection(opts.deferredTools));
  }

  // Session context briefing (turn isolation — replaces raw history)
  if (opts.briefing) {
    sections.push(
      `## Session Context\n\nYou are continuing a conversation. Here is a summary of prior work:\n\n${formatBriefing(opts.briefing)}`,
    );
  }

  return sections.join("\n\n");
}
