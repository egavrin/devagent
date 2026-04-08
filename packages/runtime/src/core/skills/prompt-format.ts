import type { SkillMetadata } from "./types.js";

function formatOptionalList(
  label: string,
  values: ReadonlyArray<string> | undefined,
  formatter: (value: string) => string = (value) => `"${value}"`,
): string | null {
  if (!values || values.length === 0) {
    return null;
  }
  return `${label}: ${values.map(formatter).join(", ")}`;
}

function formatPathCue(path: string): string {
  return `\`${path}\``;
}

export function formatSkillMatchLine(skill: SkillMetadata): string {
  const details = [
    formatOptionalList("triggers", skill.triggers),
    formatOptionalList("paths", skill.paths, formatPathCue),
    formatOptionalList("examples", skill.examples),
  ].filter((detail): detail is string => Boolean(detail));

  if (details.length === 0) {
    return `- \`${skill.name}\`: ${skill.description} (${skill.source})`;
  }

  return `- \`${skill.name}\`: ${skill.description} (${skill.source}; ${details.join("; ")})`;
}

export function formatSkillPromptGuidance(): string {
  return [
    "Match skills using user intent, touched paths, and expected output shape.",
    "Invoke the broadest relevant workflow skill first, then specialist follow-up skills only when the task clearly enters that area.",
    "Precedence examples: `surface-change-e2e` before `live-validation-authoring`; `provider-adapter-change` before `security-checklist`; `release-train` before `validate-user-surface`.",
  ].join(" ");
}
