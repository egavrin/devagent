/**
 * Commit plugin — smart commit message generation.
 * Command: /commit
 * Uses git diff to generate meaningful commit messages.
 */

import type { Plugin, PluginContext, CommandHandler } from "@devagent/core";
import { extractErrorMessage } from "@devagent/core";
import { execSync } from "node:child_process";

function generateCommitMessage(diff: string, status: string): string {
  // Extract changed files from status
  const files = status
    .split("\n")
    .filter((line) => line.trim().length > 0)
    .map((line) => line.trim());

  const addedFiles = files.filter((f) => f.startsWith("A ") || f.startsWith("?? "));
  const modifiedFiles = files.filter((f) => f.startsWith("M "));
  const deletedFiles = files.filter((f) => f.startsWith("D "));

  const parts: string[] = [];

  if (addedFiles.length > 0) {
    parts.push(`Add ${addedFiles.length} file(s)`);
  }
  if (modifiedFiles.length > 0) {
    parts.push(`Update ${modifiedFiles.length} file(s)`);
  }
  if (deletedFiles.length > 0) {
    parts.push(`Remove ${deletedFiles.length} file(s)`);
  }

  if (parts.length === 0) {
    return "Update code";
  }

  return parts.join(", ");
}

const commitCommand: CommandHandler = {
  description: "Generate a smart commit message from staged changes",
  usage: "/commit [--dry-run] [message]",

  async execute(args: string, context: PluginContext): Promise<string> {
    const isDryRun = args.includes("--dry-run");
    const customMessage = args.replace("--dry-run", "").trim();

    try {
      // Get git status and diff
      const status = execSync("git status --porcelain", {
        cwd: context.repoRoot,
        encoding: "utf-8",
      }).trim();

      if (!status) {
        return "No changes to commit.";
      }

      const diff = execSync("git diff --cached --stat", {
        cwd: context.repoRoot,
        encoding: "utf-8",
      }).trim();

      // If nothing staged, stage all tracked changes
      if (!diff) {
        return "No staged changes. Use `git add` to stage files first.";
      }

      const message = customMessage || generateCommitMessage(diff, status);

      if (isDryRun) {
        return `[dry-run] Would commit with message: "${message}"\n\nStaged changes:\n${diff}`;
      }

      // Execute commit
      const output = execSync(`git commit -m "${message.replace(/"/g, '\\"')}"`, {
        cwd: context.repoRoot,
        encoding: "utf-8",
      }).trim();

      return `Committed: "${message}"\n\n${output}`;
    } catch (err) {
      const msg = extractErrorMessage(err);
      return `Commit failed: ${msg}`;
    }
  },
};

export function createCommitPlugin(): Plugin {
  return {
    name: "commit",
    version: "1.0.0",
    description: "Smart commit message generation",
    commands: { commit: commitCommand },
    activate() {
      // No event subscriptions needed
    },
  };
}
