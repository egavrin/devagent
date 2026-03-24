/**
 * search_files — Search for text/regex in files.
 * Category: readonly.
 */

import { readFileSync } from "node:fs";
import type { ToolSpec } from "../../core/index.js";
import { resolveReadonlyPath, toRootRelativePath, type ReadonlyToolOptions } from "./readonly-paths.js";
import { escapeRegex, globToRegex, normalizeGlobPattern } from "./glob-utils.js";
import { walkDirectory } from "./walk-directory.js";

export interface SearchMatch {
  readonly file: string;
  readonly line: number;
  readonly content: string;
}

export function createSearchFilesTool(options?: ReadonlyToolOptions): ToolSpec {
  return {
    name: "search_files",
    description:
      "Search for a text pattern or regex in files. Returns matching lines with file paths and line numbers. Use file_pattern to scope the search. Search within the repo or under skill://<skill-name>/... for support files from an invoked skill.",
    category: "readonly",
    paramSchema: {
      type: "object",
      properties: {
        pattern: {
          type: "string",
          description: "Search pattern (string or regex)",
        },
        path: {
          type: "string",
          description: "Directory to search in (repo-relative or skill://<skill-name>/..., default: '.')",
        },
        file_pattern: {
          type: "string",
          description: "Glob pattern to filter files (e.g. '*.ts')",
        },
        max_results: {
          type: "number",
          description: "Maximum number of matching lines (default: 50)",
        },
      },
      required: ["pattern"],
    },
    errorGuidance: {
      common: "Try a different search pattern or broader file_pattern. Use find_files to verify the project structure first.",
      patterns: [
        { match: "invoke_skill", hint: "Unlock the skill first by calling invoke_skill with the exact skill name, then retry the search." },
      ],
    },
    resultSchema: {
      type: "object",
      properties: {
        matches: { type: "string" },
        count: { type: "number" },
      },
    },
    handler: async (params, context) => {
      const pattern = params["pattern"] as string;
      const searchPath = params["path"] as string | undefined ?? ".";
      const filePattern = params["file_pattern"] as string | undefined;
      const maxResults = (params["max_results"] as number | undefined) ?? 50;

      const resolved = resolveReadonlyPath(
        context.repoRoot,
        searchPath,
        "search_files",
        options,
      );
      const effectivePattern = filePattern ? normalizeGlobPattern(filePattern) : null;
      const fileRegex = effectivePattern ? globToRegex(effectivePattern) : null;
      let regex: RegExp;

      try {
        regex = new RegExp(pattern, "gi");
      } catch {
        regex = new RegExp(escapeRegex(pattern), "gi");
      }

      const matches: SearchMatch[] = [];

      outer:
      for (const entry of walkDirectory(resolved.resolvedPath, resolved.rootPath)) {
        if (matches.length >= maxResults) break;

        if (fileRegex && !fileRegex.test(entry.relativePath)) continue;
        if (isBinaryPath(entry.relativePath)) continue;

        let content: string;
        try {
          content = readFileSync(entry.fullPath, "utf-8");
        } catch {
          continue;
        }

        const lines = content.split("\n");
        for (let i = 0; i < lines.length; i++) {
          if (matches.length >= maxResults) break outer;
          const line = lines[i]!;
          regex.lastIndex = 0;
          if (regex.test(line)) {
            matches.push({
              file: toRootRelativePath(resolved.rootPath, entry.fullPath),
              line: i + 1,
              content: line.length > 200 ? line.substring(0, 200) + "..." : line,
            });
          }
        }
      }

      if (matches.length === 0) {
        return {
          success: true,
          output: "No matches found.",
          error: null,
          artifacts: [],
        };
      }

      const output = matches
        .map((m) => `${m.file}:${m.line}: ${m.content.trim()}`)
        .join("\n");

      return {
        success: true,
        output: `${matches.length} match(es):\n${output}`,
        error: null,
        artifacts: [],
      };
    },
  };
}

export const searchFilesTool: ToolSpec = createSearchFilesTool();

function isBinaryPath(path: string): boolean {
  const binaryExtensions = [
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".svg",
    ".woff", ".woff2", ".ttf", ".eot",
    ".zip", ".gz", ".tar", ".bz2",
    ".pdf", ".doc", ".docx",
    ".exe", ".dll", ".so", ".dylib",
    ".db", ".sqlite",
    ".mp3", ".mp4", ".avi", ".mov",
  ];
  const lower = path.toLowerCase();
  return binaryExtensions.some((ext) => lower.endsWith(ext));
}
