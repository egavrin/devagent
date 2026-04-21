/**
 * find_files — Find files matching a glob pattern.
 * Category: readonly.
 */

import { globToRegex, normalizeGlobPattern } from "./glob-utils.js";
import { resolveReadonlyPath, toRootRelativePath, type ReadonlyToolOptions } from "./readonly-paths.js";
import { walkDirectory } from "./walk-directory.js";
import type { ToolSpec } from "../../core/types.js";

const FIND_FILES_PARAM_SCHEMA = {
  type: "object",
  properties: {
    pattern: {
      type: "string",
      description: "Glob pattern (e.g. '**/*.ts', 'src/*.js')",
    },
    path: {
      type: "string",
      description: "Directory to search in (repo-relative or skill://<skill-name>/..., default: '.')",
    },
    max_results: {
      type: "number",
      description: "Maximum number of results (default: 100)",
    },
  },
  required: ["pattern"],
};

export function createFindFilesTool(options?: ReadonlyToolOptions): ToolSpec {
  return {
    name: "find_files",
    description:
      "Find files matching a glob-like pattern. Supports * and ** wildcards. Returns root-relative paths. Search within the repo or under skill://<skill-name>/... for files from an invoked skill. Skips node_modules, .git, dist, .cache.",
    category: "readonly",
    paramSchema: FIND_FILES_PARAM_SCHEMA,
    errorGuidance: {
      common: "Try a broader glob pattern. Check that you are searching in the correct directory.",
      patterns: [
        { match: "invoke_skill", hint: "Unlock the skill first by calling invoke_skill with the exact skill name, then retry the search." },
      ],
    },
    resultSchema: {
      type: "object",
      properties: {
        files: { type: "string", description: "Array of matching file paths" },
        count: { type: "number" },
      },
    },
    handler: async (params, context) => runFindFiles(params, context.repoRoot, options),
  };
}

export const findFilesTool: ToolSpec = createFindFilesTool();

function runFindFiles(
  params: Record<string, unknown>,
  repoRoot: string,
  options: ReadonlyToolOptions | undefined,
) {
  const pattern = params["pattern"] as string;
  const searchPath = params["path"] as string | undefined ?? ".";
  const maxResults = (params["max_results"] as number | undefined) ?? 100;
  const resolved = resolveReadonlyPath(repoRoot, searchPath, "find_files", options);
  const regex = globToRegex(normalizeGlobPattern(pattern));
  const matches: string[] = [];

  for (const entry of walkDirectory(resolved.resolvedPath, resolved.rootPath)) {
    if (!regex.test(entry.relativePath)) continue;
    matches.push(toRootRelativePath(resolved.rootPath, entry.fullPath));
    if (matches.length >= maxResults) break;
  }

  return {
    success: true,
    output: matches.length > 0 ? matches.join("\n") : "No files matched the pattern.",
    error: null,
    artifacts: [],
  };
}
