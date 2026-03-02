/**
 * Review plugin — code review with structured output.
 * Command: /review
 *
 * Supports two modes:
 *  - Quick heuristic checks for individual files
 *  - LLM-powered rule-based review for patch files (--rule, --patch)
 */

import type { Plugin, PluginContext, CommandHandler } from "@devagent/core";
import { extractErrorMessage } from "@devagent/core";
import { execSync } from "node:child_process";
import { readFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import { PatchParser } from "@devagent/tools/builtins/patch-parser";
import { formatPatchDataset } from "../review/chunker.js";

const reviewCommand: CommandHandler = {
  description: "Review code for issues, style, and improvements",
  usage: "/review <file|path> [--diff] [--patch <file> --rule <rule_file>]",

  async execute(args: string, context: PluginContext): Promise<string> {
    const parts = args.trim().split(/\s+/);
    const isDiff = parts.includes("--diff");
    const patchIdx = parts.indexOf("--patch");
    const ruleIdx = parts.indexOf("--rule");
    const target = parts.filter(
      (p, i) =>
        !p.startsWith("--") &&
        (patchIdx === -1 || i !== patchIdx + 1) &&
        (ruleIdx === -1 || i !== ruleIdx + 1),
    )[0];

    // Patch + rule mode: parse and format patch for LLM review
    if (patchIdx !== -1 && ruleIdx !== -1) {
      const patchFile = parts[patchIdx + 1];
      const ruleFile = parts[ruleIdx + 1];
      if (!patchFile || !ruleFile) {
        return "Usage: /review --patch <patch_file> --rule <rule_file>";
      }
      return reviewPatchWithRule(context.repoRoot, patchFile, ruleFile);
    }

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
      const msg = extractErrorMessage(err);
      return `Review failed: ${msg}`;
    }
  },
};

function reviewPatchWithRule(repoRoot: string, patchFile: string, ruleFile: string): string {
  const patchPath = resolve(repoRoot, patchFile);
  const rulePath = resolve(repoRoot, ruleFile);

  if (!existsSync(patchPath)) return `Patch file not found: ${patchFile}`;
  if (!existsSync(rulePath)) return `Rule file not found: ${ruleFile}`;

  const patchContent = readFileSync(patchPath, "utf-8");
  const ruleContent = readFileSync(rulePath, "utf-8");

  const parser = new PatchParser(patchContent, true);
  const parsed = parser.parse();

  const output: string[] = [
    `## Patch Review: ${patchFile}`,
    `**Rule**: ${ruleFile}`,
    "",
    `**Files**: ${parsed.summary.totalFiles} | **+${parsed.summary.totalAdditions}** / **-${parsed.summary.totalDeletions}**`,
    "",
  ];

  if (parsed.files.length === 0) {
    output.push("No files found in patch.");
    return output.join("\n");
  }

  output.push("### Parsed Patch Dataset");
  output.push("```");
  output.push(formatPatchDataset(parsed));
  output.push("```");
  output.push("");
  output.push("### Rule Content");
  output.push("```markdown");
  output.push(ruleContent);
  output.push("```");
  output.push("");
  output.push("Use `devagent review <patch> --rule <rule> --json` for automated LLM review.");

  return output.join("\n");
}

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
      output.push(`- ${issue}`);
    }
  } else {
    output.push("");
    output.push("No common issues found.");
  }

  output.push("");
  output.push("For a detailed AI-powered review, use:");
  output.push(`\`devagent "Review ${displayPath} for code quality, potential bugs, and improvements"\``);

  return output.join("\n");
}

export function createReviewPlugin(): Plugin {
  return {
    name: "review",
    version: "2.0.0",
    description: "Code review with structured output and rule-based patch review",
    commands: { review: reviewCommand },
    activate() {
      // No event subscriptions needed
    },
  };
}
