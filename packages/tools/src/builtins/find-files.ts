/**
 * find_files — Find files matching a glob pattern.
 * Category: readonly.
 */

import type { ToolSpec } from "@devagent/core";
import { resolvePathInRepo, resolveRepoRoot } from "./path-guard.js";
import { globToRegex, normalizeGlobPattern } from "./glob-utils.js";
import { walkDirectory } from "./walk-directory.js";

export const findFilesTool: ToolSpec = {
  name: "find_files",
  description:
    "Find files matching a glob-like pattern. Supports * and ** wildcards. Returns relative paths. Use to discover project structure before reading specific files. Skips node_modules, .git, dist, .cache.",
  category: "readonly",
  paramSchema: {
    type: "object",
    properties: {
      pattern: {
        type: "string",
        description: "Glob pattern (e.g. '**/*.ts', 'src/*.js')",
      },
      path: {
        type: "string",
        description: "Directory to search in (relative to repo root, default: '.')",
      },
      max_results: {
        type: "number",
        description: "Maximum number of results (default: 100)",
      },
    },
    required: ["pattern"],
  },
  resultSchema: {
    type: "object",
    properties: {
      files: { type: "string", description: "Array of matching file paths" },
      count: { type: "number" },
    },
  },
  handler: async (params, context) => {
    const pattern = params["pattern"] as string;
    const searchPath = params["path"] as string | undefined ?? ".";
    const maxResults = (params["max_results"] as number | undefined) ?? 100;

    const baseDir = resolvePathInRepo(
      context.repoRoot,
      searchPath,
      "find_files",
    );
    const resolvedRoot = resolveRepoRoot(context.repoRoot);
    const effectivePattern = normalizeGlobPattern(pattern);
    const regex = globToRegex(effectivePattern);
    const matches: string[] = [];

    for (const entry of walkDirectory(baseDir, resolvedRoot)) {
      if (regex.test(entry.relativePath)) {
        matches.push(entry.relativePath);
        if (matches.length >= maxResults) break;
      }
    }

    return {
      success: true,
      output: matches.length > 0
        ? matches.join("\n")
        : "No files matched the pattern.",
      error: null,
      artifacts: [],
    };
  },
};

