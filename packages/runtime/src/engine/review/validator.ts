/**
 * Validation & deduplication for LLM review responses.
 * Ensures violations reference real patch lines and normalises severity /
 * change-type values.  Ported from the Python validation implementation.
 */

import type {
  Violation,
  ReviewResult,
  ReviewSummary,
  Severity,
  ReviewChangeType,
} from "./schema.js";
import type { FileEntry } from "../../tools/builtins/patch-parser.js";

// ── Lookup tables returned by collectPatchReviewData ─────────────────────────

export interface PatchReviewData {
  /** file path -> { lineNumber -> content } for added lines. */
  addedLines: Record<string, Record<number, string>>;
  /** file path -> { lineNumber -> content } for removed lines. */
  removedLines: Record<string, Record<number, string>>;
  /** Set of all file paths present in the patch. */
  parsedFiles: Set<string>;
}

// ── Data collection ──────────────────────────────────────────────────────────

/**
 * Build lookup tables of added/removed lines per file from the parsed patch
 * file entries.
 */
export function collectPatchReviewData(files: FileEntry[]): PatchReviewData {
  const addedLines: Record<string, Record<number, string>> = {};
  const removedLines: Record<string, Record<number, string>> = {};
  const parsedFiles = new Set<string>();

  for (const fileEntry of files) {
    const path = fileEntry.path;
    if (typeof path !== "string") continue;

    parsedFiles.add(path);

    const added: Record<number, string> = {};
    const removed: Record<number, string> = {};

    for (const hunk of fileEntry.hunks ?? []) {
      for (const a of hunk.addedLines ?? []) {
        const ln = a.lineNumber;
        const c = a.content;
        if (typeof ln === "number" && typeof c === "string") {
          added[ln] = c;
        }
      }
      for (const r of hunk.removedLines ?? []) {
        const ln = r.lineNumber;
        const c = r.content;
        if (typeof ln === "number" && typeof c === "string") {
          removed[ln] = c;
        }
      }
    }

    if (Object.keys(added).length > 0) {
      addedLines[path] = added;
    }
    if (Object.keys(removed).length > 0) {
      removedLines[path] = removed;
    }
  }

  return { addedLines, removedLines, parsedFiles };
}

// ── Severity / change-type normalisation maps ────────────────────────────────

const SEVERITY_ALIASES: Record<string, Severity> = {
  critical: "error",
  high: "error",
  major: "error",
  medium: "warning",
  moderate: "warning",
  minor: "info",
  low: "info",
};

const ALLOWED_SEVERITIES = new Set<Severity>(["error", "warning", "info"]);

const CHANGE_TYPE_ALIASES: Record<string, ReviewChangeType> = {
  addition: "added",
  add: "added",
  added: "added",
  modification: "added",
  update: "added",
  deletion: "removed",
  delete: "removed",
  deleted: "removed",
  removed: "removed",
  removal: "removed",
};

// ── Validation ───────────────────────────────────────────────────────────────

/**
 * Validate an LLM review response against the actual patch data.
 *
 * Each reported violation is checked for:
 *   - valid file path and line number types
 *   - the file actually being in the patch
 *   - the line actually being an added line in the patch
 *   - code snippet matching the real content (when provided)
 *   - removed lines are discarded (we only flag additions)
 *
 * Severity and change-type values are normalised via alias maps.
 */
export function validateReviewResponse(
  response: unknown,
  addedLines: Record<string, Record<number, string>>,
  removedLines: Record<string, Record<number, string>>,
  parsedFiles: Set<string>,
): ReviewResult {
  if (typeof response !== "object" || response === null) {
    throw new Error("Reviewer output is not valid JSON.");
  }

  const responseObj = response as Record<string, unknown>;
  const violations = responseObj.violations;

  if (!Array.isArray(violations)) {
    throw new Error("Reviewer output missing 'violations' array.");
  }

  const summary = (responseObj.summary ?? {}) as Record<string, unknown>;

  const validViolations: Violation[] = [];
  const discarded: string[] = [];

  for (const entry of violations) {
    if (typeof entry !== "object" || entry === null) continue;

    const raw = entry as Record<string, unknown>;

    const filePath = raw.file;
    const lineNumber = raw.line;
    const snippet = raw.codeSnippet ?? raw.code_snippet;

    // Normalise change type
    const ctRaw = String(raw.changeType ?? raw.change_type ?? "added")
      .trim()
      .toLowerCase();
    const changeType: ReviewChangeType = CHANGE_TYPE_ALIASES[ctRaw] ?? "added";

    // Normalise severity
    let normSev: Severity = "warning";
    const sevRaw = raw.severity;
    if (typeof sevRaw === "string") {
      const candidate = sevRaw.trim().toLowerCase();
      const aliased = SEVERITY_ALIASES[candidate] ?? candidate;
      if (ALLOWED_SEVERITIES.has(aliased as Severity)) {
        normSev = aliased as Severity;
      }
    }

    // Basic type checks
    if (typeof filePath !== "string" || typeof lineNumber !== "number") {
      discarded.push(JSON.stringify(entry));
      continue;
    }

    // Discard violations on removed lines
    if (changeType === "removed") {
      discarded.push(`${filePath}:${lineNumber} (removed)`);
      continue;
    }

    // File must be in the patch
    const addedForFile = addedLines[filePath];
    if (addedForFile === undefined) {
      discarded.push(`${filePath}:${lineNumber} (not in patch)`);
      continue;
    }

    // Line must be an added line
    const actualLine = addedForFile[lineNumber];
    if (actualLine === undefined) {
      discarded.push(`${filePath}:${lineNumber} (line not added)`);
      continue;
    }

    // Snippet content match (when provided)
    if (
      typeof snippet === "string" &&
      snippet.trim() !== actualLine.trim()
    ) {
      discarded.push(`${filePath}:${lineNumber} (content mismatch)`);
      continue;
    }

    const sanitized: Violation = {
      file: filePath,
      line: lineNumber,
      severity: normSev,
      message: typeof raw.message === "string" ? raw.message : "",
      changeType,
    };

    if (typeof raw.rule === "string") {
      sanitized.rule = raw.rule;
    }

    if (typeof snippet === "string") {
      sanitized.codeSnippet = snippet;
    }

    validViolations.push(sanitized);
  }

  // Build normalised summary
  const normSummary: ReviewSummary = {
    totalViolations: validViolations.length,
    filesReviewed: parsedFiles.size,
    ruleName:
      typeof summary.ruleName === "string"
        ? summary.ruleName
        : typeof summary.rule_name === "string"
          ? summary.rule_name
          : "",
  };

  if (discarded.length > 0) {
    normSummary.discardedViolations = discarded.length;
  }

  return { violations: validViolations, summary: normSummary };
}

// ── Deduplication ────────────────────────────────────────────────────────────

/**
 * Remove duplicate violations keyed by (file, line, severity, message).
 */
export function deduplicateViolations(violations: Violation[]): Violation[] {
  const seen = new Set<string>();
  const result: Violation[] = [];

  for (const v of violations) {
    const key = `${v.file}::${v.line}::${v.severity}::${v.message}`;
    if (!seen.has(key)) {
      seen.add(key);
      result.push(v);
    }
  }

  return result;
}
