/**
 * Git tools — git_status, git_diff, git_commit.
 * Category: readonly (status/diff) and mutating (commit).
 */

import { execFileSync } from "node:child_process";
import type { ToolSpec } from "../../core/index.js";
import { ToolError , extractErrorMessage } from "../../core/index.js";

const DISALLOWED_ARG_CHARS = /[;&|`$<>\n\r\0]/;

function assertSafeArg(
  toolName: string,
  field: string,
  value: string,
): void {
  if (DISALLOWED_ARG_CHARS.test(value)) {
    throw new ToolError(
      toolName,
      `Invalid ${field}: contains disallowed shell metacharacters.`,
    );
  }
}

function parseQuotedArgs(input: string): string[] {
  const out: string[] = [];
  let current = "";
  let quote: '"' | "'" | null = null;
  let escaping = false;

  for (const ch of input) {
    if (escaping) {
      current += ch;
      escaping = false;
      continue;
    }

    if (ch === "\\") {
      escaping = true;
      continue;
    }

    if (quote) {
      if (ch === quote) {
        quote = null;
      } else {
        current += ch;
      }
      continue;
    }

    if (ch === '"' || ch === "'") {
      quote = ch;
      continue;
    }

    if (/\s/.test(ch)) {
      if (current.length > 0) {
        out.push(current);
        current = "";
      }
      continue;
    }

    current += ch;
  }

  if (quote) {
    throw new ToolError(
      "git_commit",
      "Invalid files: unterminated quote.",
    );
  }

  if (escaping) {
    current += "\\";
  }

  if (current.length > 0) {
    out.push(current);
  }

  return out;
}

function parseFilesArg(files: string): string[] {
  const parsed = parseQuotedArgs(files).filter((part) => part.length > 0);
  if (parsed.length === 0) return ["."];
  for (const file of parsed) {
    assertSafeArg("git_commit", "files", file);
  }
  return parsed;
}

function execGit(args: ReadonlyArray<string>, repoRoot: string): string {
  try {
    return execFileSync("git", [...args], {
      cwd: repoRoot,
      encoding: "utf-8",
      maxBuffer: 1024 * 1024, // 1MB
      timeout: 30_000,
    }).trim();
  } catch (err) {
    const message = extractErrorMessage(err);
    throw new ToolError("git", `git ${args.join(" ")} failed: ${message}`);
  }
}

export const gitStatusTool: ToolSpec = {
  name: "git_status",
  description: "Show the working tree status (modified, staged, untracked files). Use before committing to verify changed files.",
  category: "readonly",
  errorGuidance: {
    common: "Ensure you are in a git repository.",
  },
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
    const output = execGit(["status", "--short"], context.repoRoot);
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
    "Show changes between commits, working tree, etc. Optionally specify a file path or ref. Use after edits to verify modifications, or before committing to review changes.",
  category: "readonly",
  errorGuidance: {
    common: "Check that the ref or file path exists. Use git_status to see current state.",
  },
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

    if (path) {
      assertSafeArg("git_diff", "path", path);
    }
    if (ref) {
      assertSafeArg("git_diff", "ref", ref);
      if (ref.startsWith("-")) {
        throw new ToolError(
          "git_diff",
          "Invalid ref: option-style refs are not allowed.",
        );
      }
    }

    const args = ["diff"];
    if (staged) args.push("--cached");
    if (ref) args.push(ref);
    if (path) args.push("--", path);

    const output = execGit(args, context.repoRoot);
    const result = output || "No changes";

    // Advisory for very large diffs: suggest per-file diffing for context efficiency
    if (!path && result.length > 20_000) {
      const fileCount = (result.match(/^diff --git/gm) || []).length;
      return {
        success: true,
        output: result + `\n\n[ADVISORY: This diff contains ${fileCount} files (${result.length} chars). For better context efficiency, use git_diff with a specific file path to review files individually.]`,
        error: null,
        artifacts: [],
      };
    }

    return {
      success: true,
      output: result,
      error: null,
      artifacts: [],
    };
  },
};

export const gitCommitTool: ToolSpec = {
  name: "git_commit",
  description:
    "Stage files and create a git commit. Requires a commit message. Only use when explicitly requested by the user. Specify individual files rather than '.'. Never commit .env or secrets.",
  category: "mutating",
  errorGuidance: {
    common: "Verify file paths with git_status before committing. Ensure files are not in .gitignore.",
  },
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
    const parsedFiles = parseFilesArg(files);

    // Stage files
    execGit(["add", "--", ...parsedFiles], context.repoRoot);

    // Commit
    const output = execGit(
      ["commit", "-m", message],
      context.repoRoot,
    );

    // Get the commit hash
    const hash = execGit(["rev-parse", "--short", "HEAD"], context.repoRoot);

    return {
      success: true,
      output: `Committed: ${hash}\n${output}`,
      error: null,
      artifacts: [],
    };
  },
};
