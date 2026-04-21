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
    const { added, removed } = collectFileReviewLines(fileEntry);

    if (Object.keys(added).length > 0) {
      addedLines[path] = added;
    }
    if (Object.keys(removed).length > 0) {
      removedLines[path] = removed;
    }
  }

  return { addedLines, removedLines, parsedFiles };
}

function collectFileReviewLines(fileEntry: FileEntry): {
  readonly added: Record<number, string>;
  readonly removed: Record<number, string>;
} {
  const added: Record<number, string> = {};
  const removed: Record<number, string> = {};

  for (const hunk of fileEntry.hunks ?? []) {
    copyLineEntries(added, hunk.addedLines ?? []);
    copyLineEntries(removed, hunk.removedLines ?? []);
  }

  return { added, removed };
}

function copyLineEntries(
  target: Record<number, string>,
  entries: ReadonlyArray<{ readonly lineNumber?: number; readonly content?: string }>,
): void {
  for (const entry of entries) {
    if (typeof entry.lineNumber === "number" && typeof entry.content === "string") {
      target[entry.lineNumber] = entry.content;
    }
  }
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
    const validation = validateViolationEntry(entry, addedLines);
    if (validation.kind === "skip") continue;
    if (validation.kind === "discard") {
      discarded.push(validation.reason);
      continue;
    }
    validViolations.push(validation.violation);
  }

  const normSummary = buildReviewSummary(summary, validViolations, discarded, parsedFiles);
  return { violations: validViolations, summary: normSummary };
}

type ViolationValidation =
  | { readonly kind: "skip" }
  | { readonly kind: "discard"; readonly reason: string }
  | { readonly kind: "valid"; readonly violation: Violation };

function validateViolationEntry(
  entry: unknown,
  addedLines: Record<string, Record<number, string>>,
): ViolationValidation {
  if (typeof entry !== "object" || entry === null) return { kind: "skip" };

  const raw = entry as Record<string, unknown>;
  const filePath = raw.file;
  const lineNumber = raw.line;
  const snippet = raw.codeSnippet ?? raw.code_snippet;
  const changeType = normalizeChangeType(raw);
  const severity = normalizeSeverity(raw.severity);

  if (typeof filePath !== "string" || typeof lineNumber !== "number") {
    return { kind: "discard", reason: JSON.stringify(entry) };
  }

  if (changeType === "removed") {
    return { kind: "discard", reason: `${filePath}:${lineNumber} (removed)` };
  }

  const addedForFile = addedLines[filePath];
  if (addedForFile === undefined) {
    return { kind: "discard", reason: `${filePath}:${lineNumber} (not in patch)` };
  }

  const actualLine = addedForFile[lineNumber];
  if (actualLine === undefined) {
    return { kind: "discard", reason: `${filePath}:${lineNumber} (line not added)` };
  }

  if (isSnippetMismatch(snippet, actualLine)) {
    return { kind: "discard", reason: `${filePath}:${lineNumber} (content mismatch)` };
  }

  return { kind: "valid", violation: buildViolation(raw, {
    filePath,
    lineNumber,
    severity,
    changeType,
    snippet,
  }) };
}

function isSnippetMismatch(snippet: unknown, actualLine: string): boolean {
  return typeof snippet === "string" && snippet.trim() !== actualLine.trim();
}

function normalizeChangeType(raw: Record<string, unknown>): ReviewChangeType {
  const ctRaw = String(raw.changeType ?? raw.change_type ?? "added")
    .trim()
    .toLowerCase();
  return CHANGE_TYPE_ALIASES[ctRaw] ?? "added";
}

function normalizeSeverity(severity: unknown): Severity {
  if (typeof severity !== "string") return "warning";
  const candidate = severity.trim().toLowerCase();
  const aliased = SEVERITY_ALIASES[candidate] ?? candidate;
  return ALLOWED_SEVERITIES.has(aliased as Severity) ? aliased as Severity : "warning";
}

function buildViolation(
  raw: Record<string, unknown>,
  input: {
    readonly filePath: string;
    readonly lineNumber: number;
    readonly severity: Severity;
    readonly changeType: ReviewChangeType;
    readonly snippet: unknown;
  },
): Violation {
  const violation: Violation = {
    file: input.filePath,
    line: input.lineNumber,
    severity: input.severity,
    message: typeof raw.message === "string" ? raw.message : "",
    changeType: input.changeType,
  };

  if (typeof raw.rule === "string") violation.rule = raw.rule;
  if (typeof input.snippet === "string") violation.codeSnippet = input.snippet;
  return violation;
}

function buildReviewSummary(
  summary: Record<string, unknown>,
  validViolations: ReadonlyArray<Violation>,
  discarded: ReadonlyArray<string>,
  parsedFiles: Set<string>,
): ReviewSummary {
  const normSummary: ReviewSummary = {
    totalViolations: validViolations.length,
    filesReviewed: parsedFiles.size,
    ruleName: normalizeRuleName(summary),
  };

  if (discarded.length > 0) normSummary.discardedViolations = discarded.length;
  return normSummary;
}

function normalizeRuleName(summary: Record<string, unknown>): string {
  if (typeof summary.ruleName === "string") return summary.ruleName;
  if (typeof summary.rule_name === "string") return summary.rule_name;
  return "";
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
