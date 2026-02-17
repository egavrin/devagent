/**
 * Review plugin — code review with structured output.
 * Command: /review
 * Uses read-only tools to analyze code and provide feedback.
 */

import type { Plugin, PluginContext, CommandHandler } from "@devagent/core";
import { execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";

const reviewCommand: CommandHandler = {
  description: "Review code for issues, style, and improvements",
  usage: "/review <file|path> [--diff]",

  async execute(args: string, context: PluginContext): Promise<string> {
    const parts = args.trim().split(/\s+/);
    const isDiff = parts.includes("--diff");
    const target = parts.filter((p) => !p.startsWith("--"))[0];

    if (!target && !isDiff) {
      return "Usage: /review <file|path> [--diff]\nProvide a file path to review, or use --diff to review staged changes.";
    }

    try {
      if (isDiff) {
        return reviewDiff(context.repoRoot);
      }

      const filePath = resolve(context.repoRoot, target!);
      if (!existsSync(filePath)) {
        return `File not found: ${target}`;
      }

      return reviewFile(filePath, target!);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      return `Review failed: ${msg}`;
    }
  },
};

function reviewDiff(repoRoot: string): string {
  const diff = execSync("git diff --cached", {
    cwd: repoRoot,
    encoding: "utf-8",
  }).trim();

  if (!diff) {
    return "No staged changes to review. Stage files with `git add` first.";
  }

  const lines = diff.split("\n");
  const additions = lines.filter((l) => l.startsWith("+") && !l.startsWith("+++")).length;
  const deletions = lines.filter((l) => l.startsWith("-") && !l.startsWith("---")).length;

  const output: string[] = [
    "## Staged Changes Review",
    "",
    `**Changes**: +${additions} / -${deletions} lines`,
    "",
    "### Files Changed:",
  ];

  // Extract file names from diff headers
  const fileHeaders = lines.filter((l) => l.startsWith("+++ b/"));
  for (const header of fileHeaders) {
    const file = header.replace("+++ b/", "");
    output.push(`- ${file}`);
  }

  output.push("");
  output.push("Use the agent to get a detailed AI-powered review:");
  output.push('`devagent "Review the staged changes and suggest improvements"`');

  return output.join("\n");
}

function reviewFile(filePath: string, displayPath: string): string {
  const content = readFileSync(filePath, "utf-8");
  const lines = content.split("\n");
  const lineCount = lines.length;
  const nonEmptyLines = lines.filter((l) => l.trim().length > 0).length;

  const output: string[] = [
    `## Review: ${displayPath}`,
    "",
    `**Lines**: ${lineCount} (${nonEmptyLines} non-empty)`,
  ];

  // Basic heuristic checks
  const issues: string[] = [];

  // Long lines
  const longLines = lines.filter((l) => l.length > 120);
  if (longLines.length > 0) {
    issues.push(`${longLines.length} line(s) exceed 120 characters`);
  }

  // TODO/FIXME/HACK comments
  const todoComments = lines.filter((l) =>
    /\b(TODO|FIXME|HACK|XXX)\b/i.test(l),
  );
  if (todoComments.length > 0) {
    issues.push(`${todoComments.length} TODO/FIXME/HACK comment(s) found`);
  }

  // console.log statements
  const consoleLogs = lines.filter((l) => l.includes("console.log"));
  if (consoleLogs.length > 0) {
    issues.push(`${consoleLogs.length} console.log statement(s) found`);
  }

  // any type usage (ArkTS concern)
  const anyTypes = lines.filter((l) => /:\s*any\b/.test(l));
  if (anyTypes.length > 0) {
    issues.push(`${anyTypes.length} \`any\` type usage(s) found (ArkTS incompatible)`);
  }

  if (issues.length > 0) {
    output.push("");
    output.push("### Issues:");
    for (const issue of issues) {
      output.push(`- ⚠️ ${issue}`);
    }
  } else {
    output.push("");
    output.push("✅ No common issues found.");
  }

  output.push("");
  output.push("For a detailed AI-powered review, use:");
  output.push(`\`devagent "Review ${displayPath} for code quality, potential bugs, and improvements"\``);

  return output.join("\n");
}

export function createReviewPlugin(): Plugin {
  return {
    name: "review",
    version: "1.0.0",
    description: "Code review with structured output",
    commands: { review: reviewCommand },
    activate() {
      // No event subscriptions needed
    },
  };
}
