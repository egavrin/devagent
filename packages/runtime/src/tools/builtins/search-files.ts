/**
 * search_files — Search for text/regex in files.
 * Category: readonly.
 */

import { readFileSync } from "node:fs";

import { escapeRegex, globToRegex, normalizeGlobPattern } from "./glob-utils.js";
import { resolveReadonlyPath, toRootRelativePath, type ReadonlyToolOptions } from "./readonly-paths.js";
import { walkDirectory } from "./walk-directory.js";
import type { ToolSpec } from "../../core/types.js";

interface SearchMatch {
  readonly file: string;
  readonly line: number;
  readonly content: string;
}

interface SearchFilesRequest {
  readonly pattern: string;
  readonly searchPath: string;
  readonly filePattern?: string;
  readonly maxResults: number;
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
      const request = parseSearchFilesRequest(params);
      const resolved = resolveReadonlyPath(
        context.repoRoot,
        request.searchPath,
        "search_files",
        options,
      );
      const matches = collectSearchMatches(request, resolved);
      return {
        success: true,
        output: formatSearchMatches(matches),
        error: null,
        artifacts: [],
      };
    },
  };
}

export const searchFilesTool: ToolSpec = createSearchFilesTool();

function parseSearchFilesRequest(params: Record<string, unknown>): SearchFilesRequest {
  return {
    pattern: params["pattern"] as string,
    searchPath: params["path"] as string | undefined ?? ".",
    filePattern: params["file_pattern"] as string | undefined,
    maxResults: (params["max_results"] as number | undefined) ?? 50,
  };
}

function collectSearchMatches(
  request: SearchFilesRequest,
  resolved: { readonly resolvedPath: string; readonly rootPath: string },
): SearchMatch[] {
  const regex = createSearchRegex(request.pattern);
  const fileRegex = createFilePatternRegex(request.filePattern);
  const matches: SearchMatch[] = [];

  outer:
  for (const entry of walkDirectory(resolved.resolvedPath, resolved.rootPath)) {
    if (matches.length >= request.maxResults) break;
    if (shouldSkipSearchEntry(entry.relativePath, fileRegex)) continue;

    const content = readTextFileOrNull(entry.fullPath);
    if (content === null) continue;

    for (const match of findLineMatches(content, regex, {
      file: toRootRelativePath(resolved.rootPath, entry.fullPath),
    })) {
      if (matches.length >= request.maxResults) break outer;
      matches.push(match);
    }
  }

  return matches;
}

function createSearchRegex(pattern: string): RegExp {
  try {
    return new RegExp(pattern, "gi");
  } catch {
    return new RegExp(escapeRegex(pattern), "gi");
  }
}

function createFilePatternRegex(filePattern: string | undefined): RegExp | null {
  return filePattern ? globToRegex(normalizeGlobPattern(filePattern)) : null;
}

function shouldSkipSearchEntry(relativePath: string, fileRegex: RegExp | null): boolean {
  return (fileRegex !== null && !fileRegex.test(relativePath)) || isBinaryPath(relativePath);
}

function readTextFileOrNull(path: string): string | null {
  try {
    return readFileSync(path, "utf-8");
  } catch {
    return null;
  }
}

function findLineMatches(
  content: string,
  regex: RegExp,
  input: { readonly file: string },
): SearchMatch[] {
  const matches: SearchMatch[] = [];
  const lines = content.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    regex.lastIndex = 0;
    if (!regex.test(line)) continue;
    matches.push({
      file: input.file,
      line: i + 1,
      content: line.length > 200 ? line.substring(0, 200) + "..." : line,
    });
  }
  return matches;
}

function formatSearchMatches(matches: ReadonlyArray<SearchMatch>): string {
  if (matches.length === 0) return "No matches found.";
  const output = matches
    .map((m) => `${m.file}:${m.line}: ${m.content.trim()}`)
    .join("\n");
  return `${matches.length} match(es):\n${output}`;
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
