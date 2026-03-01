/**
 * find_files — Find files matching a glob pattern.
 * Category: readonly.
 */

import { readdirSync, realpathSync, statSync } from "node:fs";
import { join, relative, resolve } from "node:path";
import type { ToolSpec } from "@devagent/core";
import { resolvePathInRepo } from "./path-guard.js";
import { globToRegex, normalizeGlobPattern } from "./glob-utils.js";

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
    const resolvedRoot = realpathSync(resolve(context.repoRoot));
    const effectivePattern = normalizeGlobPattern(pattern);
    const regex = globToRegex(effectivePattern);
    const matches: string[] = [];

    walkDir(baseDir, resolvedRoot, regex, matches, maxResults);

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

function walkDir(
  dir: string,
  repoRoot: string,
  regex: RegExp,
  results: string[],
  maxResults: number,
): void {
  if (results.length >= maxResults) return;

  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return; // Skip directories we can't read
  }

  for (const entry of entries) {
    if (results.length >= maxResults) return;

    // Skip common ignored directories
    if (
      entry === "node_modules" ||
      entry === ".git" ||
      entry === "dist" ||
      entry === ".cache"
    ) {
      continue;
    }

    const fullPath = join(dir, entry);
    let stat;
    try {
      stat = statSync(fullPath);
    } catch {
      continue;
    }

    const relPath = relative(repoRoot, fullPath);

    if (stat.isDirectory()) {
      walkDir(fullPath, repoRoot, regex, results, maxResults);
    } else if (regex.test(relPath)) {
      results.push(relPath);
    }
  }
}

