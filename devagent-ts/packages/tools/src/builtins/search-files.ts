/**
 * search_files — Search for text/regex in files.
 * Category: readonly.
 */

import { readdirSync, readFileSync, statSync } from "node:fs";
import { resolve, join, relative } from "node:path";
import type { ToolSpec } from "@devagent/core";

export interface SearchMatch {
  readonly file: string;
  readonly line: number;
  readonly content: string;
}

export const searchFilesTool: ToolSpec = {
  name: "search_files",
  description:
    "Search for a text pattern or regex in files. Returns matching lines with file paths and line numbers. Use file_pattern to scope the search. Prefer this over reading entire files to find symbols or patterns.",
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
        description: "Directory to search in (relative to repo root, default: '.')",
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

    const baseDir = resolve(context.repoRoot, searchPath);
    const fileRegex = filePattern ? globToRegex(filePattern) : null;
    let regex: RegExp;

    try {
      regex = new RegExp(pattern, "gi");
    } catch {
      // If invalid regex, treat as literal string
      regex = new RegExp(escapeRegex(pattern), "gi");
    }

    const matches: SearchMatch[] = [];
    searchDir(baseDir, context.repoRoot, regex, fileRegex, matches, maxResults);

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

function searchDir(
  dir: string,
  repoRoot: string,
  pattern: RegExp,
  fileRegex: RegExp | null,
  results: SearchMatch[],
  maxResults: number,
): void {
  if (results.length >= maxResults) return;

  let entries: string[];
  try {
    entries = readdirSync(dir);
  } catch {
    return;
  }

  for (const entry of entries) {
    if (results.length >= maxResults) return;

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

    if (stat.isDirectory()) {
      searchDir(fullPath, repoRoot, pattern, fileRegex, results, maxResults);
    } else {
      const relPath = relative(repoRoot, fullPath);
      if (fileRegex && !fileRegex.test(relPath)) continue;

      // Skip binary files (rough heuristic: check extension)
      if (isBinaryPath(relPath)) continue;

      let content: string;
      try {
        content = readFileSync(fullPath, "utf-8");
      } catch {
        continue;
      }

      const lines = content.split("\n");
      for (let i = 0; i < lines.length; i++) {
        if (results.length >= maxResults) return;
        const line = lines[i]!;
        // Reset regex lastIndex for each line
        pattern.lastIndex = 0;
        if (pattern.test(line)) {
          results.push({
            file: relPath,
            line: i + 1,
            content: line.length > 200 ? line.substring(0, 200) + "..." : line,
          });
        }
      }
    }
  }
}

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

function globToRegex(pattern: string): RegExp {
  const regexStr = pattern
    .replace(/\./g, "\\.")
    .replace(/\*\*\//g, "{{GLOBSTARSLASH}}")
    .replace(/\*\*/g, "{{GLOBSTAR}}")
    .replace(/\*/g, "[^/]*")
    .replace(/\?/g, "[^/]")
    .replace(/\{\{GLOBSTARSLASH\}\}/g, "(.*/)?")
    .replace(/\{\{GLOBSTAR\}\}/g, ".*");
  return new RegExp(`^${regexStr}$`);
}

function escapeRegex(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
