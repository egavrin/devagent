/**
 * Git tools — git_status, git_diff, git_commit.
 * Category: readonly (status/diff) and mutating (commit).
 */

import { execSync } from "node:child_process";
import { resolve } from "node:path";
import type { ToolSpec } from "@devagent/core";
import { ToolError } from "@devagent/core";

function execGit(args: string, repoRoot: string): string {
  try {
    return execSync(`git ${args}`, {
      cwd: repoRoot,
      encoding: "utf-8",
      maxBuffer: 1024 * 1024, // 1MB
      timeout: 30_000,
    }).trim();
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new ToolError("git", `git ${args} failed: ${message}`);
  }
}

export const gitStatusTool: ToolSpec = {
  name: "git_status",
  description: "Show the working tree status (modified, staged, untracked files).",
  category: "readonly",
  paramSchema: {
    type: "object",
    properties: {},
  },
  resultSchema: {
    type: "object",
    properties: {
      status: { type: "string" },
    },
  },
  handler: async (_params, context) => {
    const output = execGit("status --short", context.repoRoot);
    return {
      success: true,
      output: output || "Working tree clean",
      error: null,
      artifacts: [],
    };
  },
};

export const gitDiffTool: ToolSpec = {
  name: "git_diff",
  description:
    "Show changes between commits, working tree, etc. Optionally specify a file path or ref.",
  category: "readonly",
  paramSchema: {
    type: "object",
    properties: {
      path: { type: "string", description: "File path to diff" },
      staged: { type: "boolean", description: "Show staged changes (default: false)" },
      ref: { type: "string", description: "Git ref to diff against (e.g. HEAD~1)" },
    },
  },
  resultSchema: {
    type: "object",
    properties: {
      diff: { type: "string" },
    },
  },
  handler: async (params, context) => {
    const path = params["path"] as string | undefined;
    const staged = (params["staged"] as boolean | undefined) ?? false;
    const ref = params["ref"] as string | undefined;

    let args = "diff";
    if (staged) args += " --cached";
    if (ref) args += ` ${ref}`;
    if (path) args += ` -- ${path}`;

    const output = execGit(args, context.repoRoot);
    return {
      success: true,
      output: output || "No changes",
      error: null,
      artifacts: [],
    };
  },
};

export const gitCommitTool: ToolSpec = {
  name: "git_commit",
  description:
    "Stage files and create a git commit. Requires a commit message.",
  category: "mutating",
  paramSchema: {
    type: "object",
    properties: {
      message: { type: "string", description: "Commit message" },
      files: {
        type: "string",
        description: "Files to stage (space-separated). Use '.' to stage all.",
      },
    },
    required: ["message"],
  },
  resultSchema: {
    type: "object",
    properties: {
      hash: { type: "string" },
    },
  },
  handler: async (params, context) => {
    const message = params["message"] as string;
    const files = (params["files"] as string | undefined) ?? ".";

    // Stage files
    execGit(`add ${files}`, context.repoRoot);

    // Commit
    const output = execGit(
      `commit -m "${message.replace(/"/g, '\\"')}"`,
      context.repoRoot,
    );

    // Get the commit hash
    const hash = execGit("rev-parse --short HEAD", context.repoRoot);

    return {
      success: true,
      output: `Committed: ${hash}\n${output}`,
      error: null,
      artifacts: [],
    };
  },
};
