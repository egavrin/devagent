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
import type { ToolSpec } from "../../core/types.js";
import { ToolError, extractErrorMessage } from "../../core/errors.js";
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

/** Context passed to every replacer — lines are pre-split once for performance. */
interface ReplacerCtx {
  readonly content: string;
  readonly lines: string[];
  readonly find: string;
  readonly findLines: string[];
}

/** A replacer yields candidate exact substrings from content that match the search intent. */
type Replacer = (ctx: ReplacerCtx) => Generator<string>;

// Similarity thresholds for block-anchor matching
const SINGLE_CANDIDATE_SIMILARITY_THRESHOLD = 0.3;
const MULTIPLE_CANDIDATES_SIMILARITY_THRESHOLD = 0.7;

// ─── Individual Replacers ───────────────────────────────────

/** 1. Exact match — yields the search string itself. */
const SimpleReplacer: Replacer = function* ({ find }) {
  yield find;
};

/** 2. Line-trimmed — matches when line-level whitespace differs. */
const LineTrimmedReplacer: Replacer = function* ({ content, lines: originalLines, findLines }) {
  const searchLines = [...findLines];

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
const BlockAnchorReplacer: Replacer = function* ({ content, lines: originalLines, findLines }) {
  const searchLines = [...findLines];

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
const WhitespaceNormalizedReplacer: Replacer = function* ({ find, lines, findLines }) {
  const normalize = (text: string) => text.replace(/\s+/g, " ").trim();
  const normalizedFind = normalize(find);

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
const IndentationFlexibleReplacer: Replacer = function* ({ find, lines: contentLines, findLines }) {
  const removeIndentation = (text: string) => {
    const textLines = text.split("\n");
    const nonEmpty = textLines.filter((l) => l.trim().length > 0);
    if (nonEmpty.length === 0) return text;

    const minIndent = Math.min(
      ...nonEmpty.map((l) => {
        const match = l.match(/^(\s*)/);
        return match ? match[1]!.length : 0;
      }),
    );

    return textLines.map((l) => (l.trim().length === 0 ? l : l.slice(minIndent))).join("\n");
  };

  const normalizedFind = removeIndentation(find);

  for (let i = 0; i <= contentLines.length - findLines.length; i++) {
    const block = contentLines.slice(i, i + findLines.length).join("\n");
    if (removeIndentation(block) === normalizedFind) {
      yield block;
    }
  }
};

/** 6. Escape-normalized — handle escape sequence differences. */
const EscapeNormalizedReplacer: Replacer = function* ({ content, lines, find }) {
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

  const unescapedFindLines = unescapedFind.split("\n");

  for (let i = 0; i <= lines.length - unescapedFindLines.length; i++) {
    const block = lines.slice(i, i + unescapedFindLines.length).join("\n");
    if (unescape(block) === unescapedFind) {
      yield block;
    }
  }
};

/** 7. Trimmed-boundary — leading/trailing whitespace on the overall block. */
const TrimmedBoundaryReplacer: Replacer = function* ({ content, lines, find, findLines }) {
  const trimmedFind = find.trim();
  if (trimmedFind === find) return; // Already trimmed

  if (content.includes(trimmedFind)) {
    yield trimmedFind;
  }

  for (let i = 0; i <= lines.length - findLines.length; i++) {
    const block = lines.slice(i, i + findLines.length).join("\n");
    if (block.trim() === trimmedFind) {
      yield block;
    }
  }
};

/** 8. Context-aware — block anchors + 50% middle-line similarity. */
const ContextAwareReplacer: Replacer = function* ({ lines: contentLines, findLines }) {
  const searchLines = [...findLines];
  if (searchLines.length < 3) return;

  if (searchLines[searchLines.length - 1] === "") {
    searchLines.pop();
  }

  const firstLine = searchLines[0]!.trim();
  const lastLine = searchLines[searchLines.length - 1]!.trim();

  for (let i = 0; i < contentLines.length; i++) {
    if (contentLines[i]!.trim() !== firstLine) continue;

    for (let j = i + 2; j < contentLines.length; j++) {
      if (contentLines[j]!.trim() !== lastLine) continue;

      const blockLines = contentLines.slice(i, j + 1);

      if (blockLines.length === searchLines.length) {
        let matchingLines = 0;
        let totalNonEmpty = 0;

        for (let k = 1; k < blockLines.length - 1; k++) {
          const blockLine = blockLines[k]!.trim();
          const findLine = searchLines[k]!.trim();
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

/** Build a ReplacerCtx from content and find strings, pre-splitting lines once. */
function makeCtx(content: string, find: string): ReplacerCtx {
  return {
    content,
    lines: content.split("\n"),
    find,
    findLines: find.split("\n"),
  };
}

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
  const ctx = makeCtx(content, oldString);

  for (const replacer of REPLACERS) {
    for (const candidate of replacer(ctx)) {
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
  const { lines } = ctx;
  const firstSearchLine = ctx.findLines[0]!.trim();
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
    for (const candidate of replacer(ctx)) {
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

// ─── Batch-mode types ────────────────────────────────────────

interface ReplacementPair {
  readonly search: string;
  readonly replace: string;
}

interface ReplacementResult {
  readonly search: string;
  readonly replace: string;
  readonly count: number;
  readonly success: boolean;
  readonly error?: string;
}

function truncate(s: string, max: number): string {
  return s.length > max ? s.slice(0, max - 3) + "..." : s;
}

// ─── Tool Definition ────────────────────────────────────────

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
    const replacements = params["replacements"] as ReplacementPair[] | undefined;
    const search = params["search"] as string | undefined;
    const replace = params["replace"] as string | undefined;

    // Mutual exclusion: batch mode vs single mode
    if (replacements !== undefined && (search !== undefined || replace !== undefined)) {
      throw new ToolError(
        "replace_in_file",
        "Cannot use both 'replacements' array and 'search'/'replace' params. Pick one mode.",
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

    // ─── Batch mode ────────────────────────────────────────
    if (replacements !== undefined) {
      if (!Array.isArray(replacements) || replacements.length === 0) {
        throw new ToolError(
          "replace_in_file",
          "Missing or empty 'replacements' array.",
        );
      }

      let content = readFileSync(filePath, "utf-8");
      const results: ReplacementResult[] = [];
      let totalCount = 0;
      let hasFailure = false;

      for (const { search: s, replace: r } of replacements) {
        if (s === r) {
          results.push({ search: s, replace: r, count: 0, success: true });
          continue;
        }

        try {
          const result = fuzzyReplace(content, s, r, true);
          if (result.newContent === content) {
            results.push({
              search: s, replace: r, count: 0, success: false,
              error: "No-op: content unchanged after replacement",
            });
            hasFailure = true;
            continue;
          }
          content = result.newContent;
          totalCount += result.count;
          results.push({ search: s, replace: r, count: result.count, success: true });
        } catch (err) {
          const message = extractErrorMessage(err);
          results.push({ search: s, replace: r, count: 0, success: false, error: message });
          hasFailure = true;
        }
      }

      if (totalCount > 0) {
        writeFileSync(filePath, content, "utf-8");
        FileTime.recordWrite(filePath);
      }

      const lines: string[] = [];
      for (const r of results) {
        if (r.success && r.count > 0) {
          lines.push(`  ✓ '${truncate(r.search, 50)}' → '${truncate(r.replace, 50)}' (${r.count})`);
        } else if (r.success && r.count === 0) {
          lines.push(`  - '${truncate(r.search, 50)}' → '${truncate(r.replace, 50)}' (no-op, identical)`);
        } else {
          lines.push(`  ✗ '${truncate(r.search, 50)}': ${r.error}`);
        }
      }

      const summary = `Applied ${totalCount} replacement(s) in ${params["path"] as string}:\n${lines.join("\n")}`;

      if (hasFailure) {
        return {
          success: false,
          output: summary,
          error: "Some replacements failed",
          artifacts: totalCount > 0 ? [filePath] : [],
        };
      }

      return {
        success: true,
        output: summary,
        error: null,
        artifacts: [filePath],
      };
    }

    // ─── Single mode (original behavior) ───────────────────

    // Guard against undefined params (from malformed JSON that parsed to {})
    if (search === undefined || replace === undefined) {
      throw new ToolError(
        "replace_in_file",
        `Missing required parameters: search=${search === undefined ? "missing" : "present"}, replace=${replace === undefined ? "missing" : "present"}. Check that tool arguments are valid JSON.`,
      );
    }

    const replaceAll = (params["all"] as boolean | undefined) ?? false;
    const expectedReplacements = params["expected_replacements"] as number | undefined;

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
        extractErrorMessage(err),
      );
    }
  },
};

// Export replacers and helpers for testing
export {
  levenshtein,
  makeCtx,
  SimpleReplacer,
  LineTrimmedReplacer,
  BlockAnchorReplacer,
  WhitespaceNormalizedReplacer,
  IndentationFlexibleReplacer,
  EscapeNormalizedReplacer,
  TrimmedBoundaryReplacer,
  ContextAwareReplacer,
};
export type { Replacer, ReplacerCtx };
