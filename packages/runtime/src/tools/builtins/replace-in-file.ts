/**
 * replace_in_file — Search and replace text in a file.
 * Category: mutating.
 *
 * Uses cascading fuzzy matching when exact search fails. The matcher itself
 * lives in replace-in-file-fuzzy.ts so this module stays focused on tool I/O.
 */

import { existsSync, readFileSync, writeFileSync } from "node:fs";

import { FileTime } from "./file-time.js";
import { resolvePathInRepo } from "./path-guard.js";
import {
  BlockAnchorReplacer,
  IndentationFlexibleReplacer,
  LineTrimmedReplacer,
  WhitespaceNormalizedReplacer,
  fuzzyReplace,
  levenshtein,
  makeCtx,
} from "./replace-in-file-fuzzy.js";
import { ToolError, extractErrorMessage } from "../../core/errors.js";
import { buildToolFileChangePreview } from "../../core/tool-file-change.js";
import type { ToolResult, ToolSpec } from "../../core/types.js";

interface ReplacementPair {
  readonly search: string;
  readonly replace: string;
}

interface ReplacementResult extends ReplacementPair {
  readonly count: number;
  readonly success: boolean;
  readonly error?: string;
}

interface ReplaceRequest {
  readonly filePath: string;
  readonly displayPath: string;
  readonly replacements?: ReplacementPair[];
  readonly search?: string;
  readonly replace?: string;
  readonly replaceAll: boolean;
  readonly expectedReplacements?: number;
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 3) + "..." : s;
}

function parseReplaceRequest(
  params: Record<string, unknown>,
  repoRoot: string,
): ReplaceRequest {
  const parseError = params["_parseError"] as string | undefined;
  if (parseError) throw new ToolError("replace_in_file", parseError);

  const displayPath = params["path"] as string;
  const search = params["search"] as string | undefined;
  const replace = params["replace"] as string | undefined;
  const replacements = params["replacements"] as ReplacementPair[] | undefined;

  if (replacements !== undefined && (search !== undefined || replace !== undefined)) {
    throw new ToolError(
      "replace_in_file",
      "Cannot use both 'replacements' array and 'search'/'replace' params. Pick one mode.",
    );
  }

  return {
    filePath: resolvePathInRepo(repoRoot, displayPath, "replace_in_file"),
    displayPath,
    replacements,
    search,
    replace,
    replaceAll: (params["all"] as boolean | undefined) ?? false,
    expectedReplacements: params["expected_replacements"] as number | undefined,
  };
}

function assertFileReady(request: ReplaceRequest): void {
  if (!existsSync(request.filePath)) {
    throw new ToolError("replace_in_file", `File not found: ${request.displayPath}`);
  }

  FileTime.assert(request.filePath);
}

function validateBatchReplacements(replacements: ReplacementPair[] | undefined): ReplacementPair[] {
  if (!Array.isArray(replacements) || replacements.length === 0) {
    throw new ToolError("replace_in_file", "Missing or empty 'replacements' array.");
  }
  return replacements;
}

function applyBatchReplacement(
  content: string,
  replacement: ReplacementPair,
): { readonly content: string; readonly result: ReplacementResult } {
  const { search, replace } = replacement;
  if (search === replace) {
    return {
      content,
      result: { search, replace, count: 0, success: true },
    };
  }

  try {
    const result = fuzzyReplace(content, search, replace, true);
    if (result.newContent === content) {
      return {
        content,
        result: {
          search,
          replace,
          count: 0,
          success: false,
          error: "No-op: content unchanged after replacement",
        },
      };
    }
    return {
      content: result.newContent,
      result: { search, replace, count: result.count, success: true },
    };
  } catch (err) {
    return {
      content,
      result: {
        search,
        replace,
        count: 0,
        success: false,
        error: extractErrorMessage(err),
      },
    };
  }
}

function runBatchReplacements(
  content: string,
  replacements: ReadonlyArray<ReplacementPair>,
): { readonly content: string; readonly results: ReplacementResult[]; readonly totalCount: number } {
  let nextContent = content;
  let totalCount = 0;
  const results: ReplacementResult[] = [];

  for (const replacement of replacements) {
    const next = applyBatchReplacement(nextContent, replacement);
    nextContent = next.content;
    totalCount += next.result.count;
    results.push(next.result);
  }

  return { content: nextContent, results, totalCount };
}

function formatBatchLine(result: ReplacementResult): string {
  const search = truncate(result.search, 50);
  const replace = truncate(result.replace, 50);
  if (result.success && result.count > 0) {
    return `  ✓ '${search}' → '${replace}' (${result.count})`;
  }
  if (result.success) {
    return `  - '${search}' → '${replace}' (no-op, identical)`;
  }
  return `  ✗ '${search}': ${result.error}`;
}

function buildBatchSummary(
  displayPath: string,
  totalCount: number,
  results: ReadonlyArray<ReplacementResult>,
): string {
  const lines = results.map(formatBatchLine);
  return `Applied ${totalCount} replacement(s) in ${displayPath}:\n${lines.join("\n")}`;
}

function buildFileEditMetadata(
  displayPath: string,
  before: string,
  after: string,
): { readonly fileEdits: unknown[] } {
  return {
    fileEdits: [
      buildToolFileChangePreview({
        path: displayPath,
        kind: "update",
        before,
        after,
      }),
    ],
  };
}

function handleBatchMode(request: ReplaceRequest): ToolResult {
  const replacements = validateBatchReplacements(request.replacements);
  const originalContent = readFileSync(request.filePath, "utf-8");
  const batch = runBatchReplacements(originalContent, replacements);

  if (batch.totalCount > 0) {
    writeFileSync(request.filePath, batch.content, "utf-8");
    FileTime.recordWrite(request.filePath);
  }

  const hasFailure = batch.results.some((result) => !result.success);
  const summary = buildBatchSummary(request.displayPath, batch.totalCount, batch.results);

  if (hasFailure) {
    return {
      success: false,
      output: summary,
      error: "Some replacements failed",
      artifacts: batch.totalCount > 0 ? [request.filePath] : [],
    };
  }

  return {
    success: true,
    output: summary,
    error: null,
    artifacts: [request.filePath],
    metadata: buildFileEditMetadata(request.displayPath, originalContent, batch.content),
  };
}

function validateSingleRequest(request: ReplaceRequest): asserts request is ReplaceRequest & {
  readonly search: string;
  readonly replace: string;
} {
  if (request.search === undefined || request.replace === undefined) {
    throw new ToolError(
      "replace_in_file",
      `Missing required parameters: search=${request.search === undefined ? "missing" : "present"}, replace=${request.replace === undefined ? "missing" : "present"}. Check that tool arguments are valid JSON.`,
    );
  }

  if (request.search === request.replace) {
    throw new ToolError(
      "replace_in_file",
      "No changes to apply: search and replace strings are identical.",
    );
  }
}

function assertExpectedReplacementCount(
  expectedReplacements: number | undefined,
  actualCount: number,
): void {
  if (expectedReplacements === undefined || actualCount === expectedReplacements) return;
  throw new ToolError(
    "replace_in_file",
    `Expected ${expectedReplacements} replacement(s), but made ${actualCount}. The edit was not applied.`,
  );
}

function assertContentChanged(content: string, nextContent: string): void {
  if (nextContent !== content) return;
  throw new ToolError(
    "replace_in_file",
    "No-op replacement: fuzzy matching found text but replacement produced identical content. The file may already contain the target text.",
  );
}

function handleSingleMode(request: ReplaceRequest): ToolResult {
  validateSingleRequest(request);

  const content = readFileSync(request.filePath, "utf-8");
  const result = fuzzyReplace(content, request.search, request.replace, request.replaceAll);

  assertExpectedReplacementCount(request.expectedReplacements, result.count);
  assertContentChanged(content, result.newContent);

  writeFileSync(request.filePath, result.newContent, "utf-8");
  FileTime.recordWrite(request.filePath);

  return {
    success: true,
    output: `Replaced ${result.count} occurrence(s) in ${request.displayPath}`,
    error: null,
    artifacts: [request.filePath],
    metadata: buildFileEditMetadata(request.displayPath, content, result.newContent),
  };
}

function handleReplaceInFile(params: Record<string, unknown>, repoRoot: string): ToolResult {
  const request = parseReplaceRequest(params, repoRoot);
  assertFileReady(request);
  return request.replacements === undefined
    ? handleSingleMode(request)
    : handleBatchMode(request);
}

export const replaceInFileTool: ToolSpec = {
  name: "replace_in_file",
  description:
    "Replace text in a file using fuzzy matching. Two modes:\n" +
    "• Single: provide search + replace (+ optional all, expected_replacements)\n" +
    "• Batch: provide replacements array of {search, replace} pairs — " +
    "applied sequentially with replaceAll, partial writes on failure.\n" +
    "Always read_file first.",
  category: "mutating",
  paramSchema: {
    type: "object",
    properties: {
      path: { type: "string", description: "File path (relative to repo root)" },
      search: { type: "string", description: "Text to search for (single mode)" },
      replace: { type: "string", description: "Replacement text (single mode)" },
      all: { type: "boolean", description: "Replace all occurrences (single mode, default: false)" },
      expected_replacements: {
        type: "number",
        description: "Optional safety check (single mode). Fails if count differs.",
      },
      replacements: {
        type: "array",
        description: "Batch mode: array of {search, replace} pairs applied sequentially. Mutually exclusive with search/replace.",
        items: {
          type: "object",
          properties: {
            search: { type: "string", description: "Text to search for" },
            replace: { type: "string", description: "Replacement text" },
          },
          required: ["search", "replace"],
          additionalProperties: false,
        },
      },
    },
    required: ["path"],
  },
  errorGuidance: {
    common: "Re-read the file with read_file and copy the exact current text as your search parameter.",
    patterns: [
      { match: "not found", hint: "The search text doesn't match. Re-read the file — it may have changed since your last read. Copy the exact text from read_file output." },
      { match: "multiple matches", hint: "The search text matches multiple locations. Add 1-3 surrounding context lines to make it unique." },
      { match: "No-op", hint: "The replacement produced identical content. Check that your replace text is actually different from the search text." },
      { match: "Missing required", hint: "Check that tool arguments are valid JSON with both 'search' and 'replace' fields." },
    ],
  },
  resultSchema: {
    type: "object",
    properties: {
      replacements: { type: "number" },
    },
  },
  handler: async (params, context) => {
    try {
      return handleReplaceInFile(params, context.repoRoot);
    } catch (err) {
      if (err instanceof ToolError) throw err;
      throw new ToolError("replace_in_file", extractErrorMessage(err));
    }
  },
};

export {
  fuzzyReplace,
  levenshtein,
  makeCtx,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
};
