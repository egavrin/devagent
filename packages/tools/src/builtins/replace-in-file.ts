/**
 * replace_in_file — Search and replace text in a file.
 * Category: mutating.
 *
 * Uses cascading fuzzy matching when exact search fails:
 * 1. Exact match
 * 2. Line-trimmed (whitespace at line start/end)
 * 3. Block-anchor (first/last line anchors + Levenshtein similarity)
 * 4. Whitespace-normalized (collapse internal whitespace)
 * 5. Indentation-flexible (different indent levels)
 * 6. Escape-normalized (handle escape sequences)
 * 7. Trimmed-boundary (leading/trailing block whitespace)
 * 8. Context-aware (block anchors + 50% middle-line match)
 *
 * Approach sourced from OpenCode, Cline, and Gemini CLI.
 */

import { readFileSync, writeFileSync, existsSync, statSync } from "node:fs";
import type { ToolSpec } from "@devagent/core";
import { ToolError } from "@devagent/core";
import { FileTime } from "./file-time.js";
import { resolvePathInRepo } from "./path-guard.js";

// ─── Levenshtein Distance ───────────────────────────────────

function levenshtein(a: string, b: string): number {
  if (a === b) return 0;
  if (a.length === 0) return b.length;
  if (b.length === 0) return a.length;

  const matrix: number[][] = [];
  for (let i = 0; i <= a.length; i++) {
    matrix[i] = [i];
  }
  for (let j = 0; j <= b.length; j++) {
    matrix[0]![j] = j;
  }

  for (let i = 1; i <= a.length; i++) {
    for (let j = 1; j <= b.length; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      matrix[i]![j] = Math.min(
        matrix[i - 1]![j]! + 1,
        matrix[i]![j - 1]! + 1,
        matrix[i - 1]![j - 1]! + cost,
      );
    }
  }

  return matrix[a.length]![b.length]!;
}

// ─── Replacer Types ─────────────────────────────────────────

/** A replacer yields candidate exact substrings from content that match the search intent. */
type Replacer = (content: string, find: string) => Generator<string>;

// Similarity thresholds for block-anchor matching
const SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.3;
const MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.7;

// ─── Individual Replacers ───────────────────────────────────

/** 1. Exact match — yields the search string itself. */
const SimpleReplacer: Replacer = function* (_content, find) {
  yield find;
};

/** 2. Line-trimmed — matches when line-level whitespace differs. */
const LineTrimmedReplacer: Replacer = function* (content, find) {
  const originalLines = content.split("\n");
  const searchLines = find.split("\n");

  if (searchLines[searchLines.length - 1] === "") {
    searchLines.pop();
  }

  for (let i = 0; i <= originalLines.length - searchLines.length; i++) {
    let matches = true;

    for (let j = 0; j < searchLines.length; j++) {
      if (originalLines[i + j]!.trim() !== searchLines[j]!.trim()) {
        matches = false;
        break;
      }
    }

    if (matches) {
      let matchStartIndex = 0;
      for (let k = 0; k < i; k++) {
        matchStartIndex += originalLines[k]!.length + 1;
      }

      let matchEndIndex = matchStartIndex;
      for (let k = 0; k < searchLines.length; k++) {
        matchEndIndex += originalLines[i + k]!.length;
        if (k < searchLines.length - 1) {
          matchEndIndex += 1;
        }
      }

      yield content.substring(matchStartIndex, matchEndIndex);
    }
  }
};

/** 3. Block-anchor — first/last lines as anchors, Levenshtein on middle. */
const BlockAnchorReplacer: Replacer = function* (content, find) {
  const originalLines = content.split("\n");
  const searchLines = find.split("\n");

  if (searchLines.length < 3) return;

  if (searchLines[searchLines.length - 1] === "") {
    searchLines.pop();
  }

  const firstLineSearch = searchLines[0]!.trim();
  const lastLineSearch = searchLines[searchLines.length - 1]!.trim();
  const searchBlockSize = searchLines.length;

  // Collect candidate positions where both anchors match
  const candidates: Array<{ startLine: number; endLine: number }> = [];
  for (let i = 0; i < originalLines.length; i++) {
    if (originalLines[i]!.trim() !== firstLineSearch) continue;

    for (let j = i + 2; j < originalLines.length; j++) {
      if (originalLines[j]!.trim() === lastLineSearch) {
        candidates.push({ startLine: i, endLine: j });
        break;
      }
    }
  }

  if (candidates.length === 0) return;

  function extractSubstring(startLine: number, endLine: number): string {
    let matchStartIndex = 0;
    for (let k = 0; k < startLine; k++) {
      matchStartIndex += originalLines[k]!.length + 1;
    }
    let matchEndIndex = matchStartIndex;
    for (let k = startLine; k <= endLine; k++) {
      matchEndIndex += originalLines[k]!.length;
      if (k < endLine) matchEndIndex += 1;
    }
    return content.substring(matchStartIndex, matchEndIndex);
  }

  function calcSimilarity(startLine: number, endLine: number, threshold: number): number {
    const actualBlockSize = endLine - startLine + 1;
    const linesToCheck = Math.min(searchBlockSize - 2, actualBlockSize - 2);

    if (linesToCheck <= 0) return 1.0;

    let similarity = 0;
    for (let j = 1; j < searchBlockSize - 1 && j < actualBlockSize - 1; j++) {
      const originalLine = originalLines[startLine + j]!.trim();
      const searchLine = searchLines[j]!.trim();
      const maxLen = Math.max(originalLine.length, searchLine.length);
      if (maxLen === 0) continue;
      const distance = levenshtein(originalLine, searchLine);
      similarity += (1 - distance / maxLen) / linesToCheck;
      if (similarity >= threshold) break;
    }
    return similarity;
  }

  if (candidates.length === 1) {
    const { startLine, endLine } = candidates[0]!;
    if (calcSimilarity(startLine, endLine, SINGLE_CANDIDATE_SIMILARITY_THRESHOLD) >= SINGLE_CANDIDATE_SIMILARITY_THRESHOLD) {
      yield extractSubstring(startLine, endLine);
    }
    return;
  }

  // Multiple candidates — pick the best
  let bestMatch: { startLine: number; endLine: number } | null = null;
  let maxSimilarity = -1;

  for (const candidate of candidates) {
    const sim = calcSimilarity(candidate.startLine, candidate.endLine, 0);
    if (sim > maxSimilarity) {
      maxSimilarity = sim;
      bestMatch = candidate;
    }
  }

  if (maxSimilarity >= MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD && bestMatch) {
    yield extractSubstring(bestMatch.startLine, bestMatch.endLine);
  }
};

/** 4. Whitespace-normalized — collapse all whitespace and match. */
const WhitespaceNormalizedReplacer: Replacer = function* (content, find) {
  const normalize = (text: string) => text.replace(/\s+/g, " ").trim();
  const normalizedFind = normalize(find);

  const lines = content.split("\n");

  // Single-line matches
  for (const line of lines) {
    if (normalize(line) === normalizedFind) {
      yield line;
    } else {
      const normalizedLine = normalize(line);
      if (normalizedLine.includes(normalizedFind)) {
        const words = find.trim().split(/\s+/);
        if (words.length > 0) {
          const pattern = words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("\\s+");
          try {
            const match = line.match(new RegExp(pattern));
            if (match) yield match[0];
          } catch {
            // Invalid regex, skip
          }
        }
      }
    }
  }

  // Multi-line matches
  const findLines = find.split("\n");
  if (findLines.length > 1) {
    for (let i = 0; i <= lines.length - findLines.length; i++) {
      const block = lines.slice(i, i + findLines.length);
      if (normalize(block.join("\n")) === normalizedFind) {
        yield block.join("\n");
      }
    }
  }
};

/** 5. Indentation-flexible — matches when indent level differs. */
const IndentationFlexibleReplacer: Replacer = function* (content, find) {
  const removeIndentation = (text: string) => {
    const lines = text.split("\n");
    const nonEmpty = lines.filter((l) => l.trim().length > 0);
    if (nonEmpty.length === 0) return text;

    const minIndent = Math.min(
      ...nonEmpty.map((l) => {
        const match = l.match(/^(\s*)/);
        return match ? match[1]!.length : 0;
      }),
    );

    return lines.map((l) => (l.trim().length === 0 ? l : l.slice(minIndent))).join("\n");
  };

  const normalizedFind = removeIndentation(find);
  const contentLines = content.split("\n");
  const findLines = find.split("\n");

  for (let i = 0; i <= contentLines.length - findLines.length; i++) {
    const block = contentLines.slice(i, i + findLines.length).join("\n");
    if (removeIndentation(block) === normalizedFind) {
      yield block;
    }
  }
};

/** 6. Escape-normalized — handle escape sequence differences. */
const EscapeNormalizedReplacer: Replacer = function* (content, find) {
  const unescape = (str: string): string =>
    str.replace(/\\(n|t|r|'|"|`|\\|\n|\$)/g, (_match, ch: string) => {
      switch (ch) {
        case "n": return "\n";
        case "t": return "\t";
        case "r": return "\r";
        case "'": return "'";
        case '"': return '"';
        case "`": return "`";
        case "\\": return "\\";
        case "\n": return "\n";
        case "$": return "$";
        default: return _match;
      }
    });

  const unescapedFind = unescape(find);

  if (content.includes(unescapedFind)) {
    yield unescapedFind;
  }

  const lines = content.split("\n");
  const findLines = unescapedFind.split("\n");

  for (let i = 0; i <= lines.length - findLines.length; i++) {
    const block = lines.slice(i, i + findLines.length).join("\n");
    if (unescape(block) === unescapedFind) {
      yield block;
    }
  }
};

/** 7. Trimmed-boundary — leading/trailing whitespace on the overall block. */
const TrimmedBoundaryReplacer: Replacer = function* (content, find) {
  const trimmedFind = find.trim();
  if (trimmedFind === find) return; // Already trimmed

  if (content.includes(trimmedFind)) {
    yield trimmedFind;
  }

  const lines = content.split("\n");
  const findLines = find.split("\n");

  for (let i = 0; i <= lines.length - findLines.length; i++) {
    const block = lines.slice(i, i + findLines.length).join("\n");
    if (block.trim() === trimmedFind) {
      yield block;
    }
  }
};

/** 8. Context-aware — block anchors + 50% middle-line similarity. */
const ContextAwareReplacer: Replacer = function* (content, find) {
  const findLines = find.split("\n");
  if (findLines.length < 3) return;

  if (findLines[findLines.length - 1] === "") {
    findLines.pop();
  }

  const contentLines = content.split("\n");
  const firstLine = findLines[0]!.trim();
  const lastLine = findLines[findLines.length - 1]!.trim();

  for (let i = 0; i < contentLines.length; i++) {
    if (contentLines[i]!.trim() !== firstLine) continue;

    for (let j = i + 2; j < contentLines.length; j++) {
      if (contentLines[j]!.trim() !== lastLine) continue;

      const blockLines = contentLines.slice(i, j + 1);

      if (blockLines.length === findLines.length) {
        let matchingLines = 0;
        let totalNonEmpty = 0;

        for (let k = 1; k < blockLines.length - 1; k++) {
          const blockLine = blockLines[k]!.trim();
          const findLine = findLines[k]!.trim();
          if (blockLine.length > 0 || findLine.length > 0) {
            totalNonEmpty++;
            if (blockLine === findLine) matchingLines++;
          }
        }

        if (totalNonEmpty === 0 || matchingLines / totalNonEmpty >= 0.5) {
          yield blockLines.join("\n");
          return;
        }
      }
      break;
    }
  }
};

// ─── Cascading Replace ──────────────────────────────────────

const REPLACERS: ReadonlyArray<Replacer> = [
  SimpleReplacer,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
  EscapeNormalizedReplacer,
  TrimmedBoundaryReplacer,
  ContextAwareReplacer,
];

/**
 * Try each replacer in order. The first one that yields a match found in content wins.
 * Returns `{ newContent, count }` or throws if no replacer matches.
 */
export function fuzzyReplace(
  content: string,
  oldString: string,
  newString: string,
  replaceAll: boolean,
): { newContent: string; count: number } {
  for (const replacer of REPLACERS) {
    for (const candidate of replacer(content, oldString)) {
      const index = content.indexOf(candidate);
      if (index === -1) continue;

      // Skip no-op: if the fuzzy candidate IS the replacement, replacing would change nothing
      if (candidate === newString) continue;

      if (replaceAll) {
        const count = content.split(candidate).length - 1;
        const newContent = content.replaceAll(candidate, newString);
        // Guard against no-op even after replaceAll (e.g., candidate found but result identical)
        if (newContent === content) continue;
        return { newContent, count };
      }

      // Single replace: require unique match
      const lastIndex = content.lastIndexOf(candidate);
      if (index !== lastIndex) continue; // Ambiguous — skip this candidate

      return {
        newContent: content.substring(0, index) + newString + content.substring(index + candidate.length),
        count: 1,
      };
    }
  }

  // All replacers exhausted — build rich error context
  const lines = content.split("\n");
  const firstSearchLine = oldString.split("\n")[0]!.trim();
  let hint = "";

  if (firstSearchLine.length > 5) {
    const token = firstSearchLine.substring(0, Math.min(40, firstSearchLine.length));
    for (let i = 0; i < lines.length; i++) {
      if (lines[i]!.includes(token)) {
        const start = Math.max(0, i - 2);
        const end = Math.min(lines.length, i + 3);
        hint = `\nPartial match near line ${i + 1}:\n` +
          lines.slice(start, end).map((l, idx) => `${start + idx + 1}: ${l}`).join("\n");
        break;
      }
    }
  }

  if (!hint) {
    const preview = lines.slice(0, 15).map((l, i) => `${i + 1}: ${l}`).join("\n");
    hint = `\nFile has ${lines.length} lines. First 15:\n${preview}`;
  }

  // Distinguish: no match at all vs. multiple ambiguous matches
  for (const replacer of REPLACERS) {
    for (const candidate of replacer(content, oldString)) {
      const index = content.indexOf(candidate);
      if (index !== -1) {
        const lastIndex = content.lastIndexOf(candidate);
        if (index !== lastIndex) {
          throw new Error(
            "Found multiple matches for search string. Provide more surrounding context to make the match unique.",
          );
        }
      }
    }
  }

  throw new Error(`Search string not found. Re-read the file for exact text.${hint}`);
}

// ─── Tool Definition ────────────────────────────────────────

export const replaceInFileTool: ToolSpec = {
  name: "replace_in_file",
  description:
    "Replace occurrences of a search string with a replacement string in a file. Uses fuzzy matching (whitespace, indentation, escape sequences) when exact match fails. Always read_file first to get text. Default mode is single replacement; set all=true for global replacement.",
  category: "mutating",
  paramSchema: {
    type: "object",
    properties: {
      path: { type: "string", description: "File path (relative to repo root)" },
      search: { type: "string", description: "Text to search for" },
      replace: { type: "string", description: "Replacement text" },
      all: { type: "boolean", description: "Replace all occurrences (default: false)" },
      expected_replacements: {
        type: "number",
        description: "Optional safety check. Fails if the number of replacements differs from this value.",
      },
    },
    required: ["path", "search", "replace"],
  },
  resultSchema: {
    type: "object",
    properties: {
      replacements: { type: "number" },
    },
  },
  handler: async (params, context) => {
    // Detect malformed arguments from JSON parse fallback
    const parseError = params["_parseError"] as string | undefined;
    if (parseError) {
      throw new ToolError("replace_in_file", parseError);
    }

    const filePath = resolvePathInRepo(
      context.repoRoot,
      params["path"] as string,
      "replace_in_file",
    );
    const search = params["search"] as string;
    const replace = params["replace"] as string;
    const replaceAll = (params["all"] as boolean | undefined) ?? false;
    const expectedReplacements = params["expected_replacements"] as number | undefined;

    // Guard against undefined params (from malformed JSON that parsed to {})
    if (search === undefined || replace === undefined) {
      throw new ToolError(
        "replace_in_file",
        `Missing required parameters: search=${search === undefined ? "missing" : "present"}, replace=${replace === undefined ? "missing" : "present"}. Check that tool arguments are valid JSON.`,
      );
    }

    if (!existsSync(filePath)) {
      throw new ToolError(
        "replace_in_file",
        `File not found: ${params["path"] as string}`,
      );
    }

    // Enforce pre-read: file must have been read in this session
    FileTime.assert(filePath);

    if (search === replace) {
      throw new ToolError(
        "replace_in_file",
        "No changes to apply: search and replace strings are identical.",
      );
    }

    const content = readFileSync(filePath, "utf-8");

    try {
      const result = fuzzyReplace(content, search, replace, replaceAll);

      if (
        expectedReplacements !== undefined &&
        result.count !== expectedReplacements
      ) {
        throw new ToolError(
          "replace_in_file",
          `Expected ${expectedReplacements} replacement(s), but made ${result.count}. The edit was not applied.`,
        );
      }

      // Fail fast: if fuzzyReplace returned but content is unchanged, report the no-op
      if (result.newContent === content) {
        throw new ToolError(
          "replace_in_file",
          "No-op replacement: fuzzy matching found text but replacement produced identical content. The file may already contain the target text.",
        );
      }

      writeFileSync(filePath, result.newContent, "utf-8");
      FileTime.recordWrite(filePath);

      return {
        success: true,
        output: `Replaced ${result.count} occurrence(s) in ${params["path"] as string}`,
        error: null,
        artifacts: [filePath],
      };
    } catch (err) {
      if (err instanceof ToolError) throw err;
      throw new ToolError(
        "replace_in_file",
        err instanceof Error ? err.message : String(err),
      );
    }
  },
};

// Export replacers and helpers for testing
export {
  levenshtein,
  SimpleReplacer,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
  EscapeNormalizedReplacer,
  TrimmedBoundaryReplacer,
  ContextAwareReplacer,
};
export type { Replacer };
