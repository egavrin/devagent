import { Buffer } from "node:buffer";

import {
  countCommonPrefix,
  countCommonSuffix,
  diffSnapshotLines,
} from "./tool-file-change-diff.js";
import type { DiffOperation } from "./tool-file-change-diff.js";
import type {
  ToolFileChangePreview,
  ToolFileChangeLine,
  ToolFileChangeHunk,
  ToolFileStructuredDiff,
} from "./types.js";

const DEFAULT_MAX_DIFF_LINES = 40;
const DEFAULT_MAX_DIFF_BYTES = 8 * 1024;
const DEFAULT_MAX_FILES = 3;
const DEFAULT_MAX_SNAPSHOT_BYTES = 64 * 1024;
const DIFF_CONTEXT_LINES = 3;
const TRUNCATION_MARKER = "... diff truncated ...";

interface BuildToolFileChangePreviewOptions {
  readonly path: string;
  readonly kind: ToolFileChangePreview["kind"];
  readonly before: string;
  readonly after: string;
  readonly maxDiffLines?: number;
  readonly maxDiffBytes?: number;
  readonly maxSnapshotBytes?: number;
}

interface ToolFileChangePreviewSummary {
  readonly fileEdits?: ReadonlyArray<ToolFileChangePreview>;
  readonly hiddenFileCount: number;
}

interface ToolFileDiffAnalysis {
  readonly additions: number;
  readonly deletions: number;
  readonly diffLines: ReadonlyArray<string>;
  readonly structuredDiff?: ToolFileStructuredDiff;
}

export function buildToolFileUnifiedDiff(
  options: BuildToolFileChangePreviewOptions,
): string {
  return analyzeToolFileChange(options).diffLines.join("\n");
}

export function buildToolFileStructuredDiff(
  before: string,
  after: string,
): ToolFileStructuredDiff | undefined {
  const beforeLines = splitFileLines(before);
  const afterLines = splitFileLines(after);
  const operations = diffSnapshotLines(beforeLines, afterLines);
  if (!operations) return undefined;

  return {
    hunks: groupLinesIntoHunks(numberDiffLines(operations)),
  };
}
export function buildToolFileStructuredDiffFromUnifiedDiff(
  diff: string,
): ToolFileStructuredDiff | undefined {
  const hunks: ToolFileChangeHunk[] = [];
  let currentLines: ToolFileChangeLine[] = [];
  let oldLine: number | null = null;
  let newLine: number | null = null;

  const flushHunk = (): void => {
    if (currentLines.length === 0) return;
    hunks.push(buildHunkFromLines(currentLines));
    currentLines = [];
  };

  for (const line of diff.split("\n")) {
    if (shouldSkipUnifiedDiffLine(line)) continue;
    if (line.startsWith("@@")) {
      flushHunk();
      const parsed = parseHunkHeader(line);
      oldLine = parsed.oldStart;
      newLine = parsed.newStart;
      continue;
    }
    const parsedLine = parseUnifiedDiffContentLine(line, oldLine, newLine);
    currentLines.push(parsedLine.line);
    oldLine = parsedLine.oldLine;
    newLine = parsedLine.newLine;
  }

  flushHunk();

  return hunks.length > 0 ? { hunks } : undefined;
}

function shouldSkipUnifiedDiffLine(line: string): boolean {
  return line.length === 0 ||
    line.startsWith("---") ||
    line.startsWith("+++") ||
    line === TRUNCATION_MARKER;
}

function parseUnifiedDiffContentLine(
  line: string,
  oldLine: number | null,
  newLine: number | null,
): {
  readonly line: ToolFileChangeLine;
  readonly oldLine: number | null;
  readonly newLine: number | null;
} {
  if (line.startsWith("+")) {
    return {
      line: { type: "add", text: line.slice(1), oldLine: null, newLine },
      oldLine,
      newLine: (newLine ?? 0) + 1,
    };
  }
  if (line.startsWith("-")) {
    return {
      line: { type: "delete", text: line.slice(1), oldLine, newLine: null },
      oldLine: (oldLine ?? 0) + 1,
      newLine,
    };
  }
  return {
    line: {
      type: "context",
      text: line.startsWith(" ") ? line.slice(1) : line,
      oldLine,
      newLine,
    },
    oldLine: (oldLine ?? 0) + 1,
    newLine: (newLine ?? 0) + 1,
  };
}

export function buildToolFileChangePreview(
  options: BuildToolFileChangePreviewOptions,
): ToolFileChangePreview {
  const analysis = analyzeToolFileChange(options);
  const clipped = clipUnifiedDiff(
    analysis.diffLines,
    options.maxDiffLines ?? DEFAULT_MAX_DIFF_LINES,
    options.maxDiffBytes ?? DEFAULT_MAX_DIFF_BYTES,
  );
  const maxSnapshotBytes = options.maxSnapshotBytes ?? DEFAULT_MAX_SNAPSHOT_BYTES;
  const includeSnapshots =
    Buffer.byteLength(options.before, "utf8") <= maxSnapshotBytes
    && Buffer.byteLength(options.after, "utf8") <= maxSnapshotBytes;
  const includeStructuredDiff =
    analysis.structuredDiff
    && Buffer.byteLength(JSON.stringify(analysis.structuredDiff), "utf8") <= maxSnapshotBytes;

  return {
    path: options.path,
    kind: options.kind,
    additions: analysis.additions,
    deletions: analysis.deletions,
    unifiedDiff: clipped.unifiedDiff,
    truncated: clipped.truncated,
    ...(includeStructuredDiff ? { structuredDiff: analysis.structuredDiff } : {}),
    ...(includeSnapshots ? { before: options.before, after: options.after } : {}),
  };
}

export function extractToolFileChangePreviewSummary(
  metadata: Record<string, unknown> | undefined,
  maxFiles: number = DEFAULT_MAX_FILES,
): ToolFileChangePreviewSummary {
  const raw = metadata?.["fileEdits"];
  if (!Array.isArray(raw)) {
    return {
      fileEdits: undefined,
      hiddenFileCount: 0,
    };
  }

  const previews: ToolFileChangePreview[] = [];
  let totalValid = 0;

  for (const entry of raw) {
    const preview = parseToolFileChangePreview(entry);
    if (!preview) continue;

    totalValid++;
    if (previews.length < maxFiles) {
      previews.push(preview);
    }
  }

  return {
    fileEdits: previews.length > 0 ? previews : undefined,
    hiddenFileCount: Math.max(0, totalValid - previews.length),
  };
}

export function extractToolFileChangePreviews(
  metadata: Record<string, unknown> | undefined,
  maxFiles: number = DEFAULT_MAX_FILES,
): ReadonlyArray<ToolFileChangePreview> | undefined {
  return extractToolFileChangePreviewSummary(metadata, maxFiles).fileEdits;
}

export function stripToolFileChangePresentationData(
  preview: ToolFileChangePreview,
): ToolFileChangePreview {
  const { before: _before, after: _after, structuredDiff: _structuredDiff, ...rest } = preview;
  return rest;
}

function splitFileLines(text: string): string[] {
  if (text.length === 0) return [];
  const lines = text.replace(/\r\n/g, "\n").split("\n");
  if (lines.length > 0 && lines[lines.length - 1] === "") {
    lines.pop();
  }
  return lines;
}

function clipUnifiedDiff(
  lines: ReadonlyArray<string>,
  maxLines: number,
  maxBytes: number,
): {
  readonly unifiedDiff: string;
  readonly truncated: boolean;
} {
  const clipped = [...lines];
  let truncated = false;

  if (clipped.length > maxLines) {
    clipped.splice(Math.max(0, maxLines - 1));
    truncated = true;
  }

  if (truncated) {
    clipped.push(TRUNCATION_MARKER);
  }

  while (clipped.length > 1 && Buffer.byteLength(clipped.join("\n"), "utf8") > maxBytes) {
    truncated = true;
    const markerIndex = clipped.lastIndexOf(TRUNCATION_MARKER);
    if (markerIndex >= 0) {
      clipped.splice(Math.max(0, markerIndex - 1), 1);
    } else {
      clipped.pop();
    }
    if (clipped[clipped.length - 1] !== TRUNCATION_MARKER) {
      clipped.push(TRUNCATION_MARKER);
    }
  }

  return {
    unifiedDiff: clipped.join("\n"),
    truncated,
  };
}
function analyzeToolFileChange(options: BuildToolFileChangePreviewOptions): ToolFileDiffAnalysis {
  const structuredDiff = buildToolFileStructuredDiff(options.before, options.after);
  if (structuredDiff) {
    return analyzeStructuredToolFileChange(options, structuredDiff);
  }

  const beforeLines = splitFileLines(options.before);
  const afterLines = splitFileLines(options.after);
  return analyzeLineRangeToolFileChange(options, beforeLines, afterLines);
}

function analyzeStructuredToolFileChange(
  options: BuildToolFileChangePreviewOptions,
  structuredDiff: ToolFileStructuredDiff,
): ToolFileDiffAnalysis {
  return {
    additions: countStructuredDiffLines(structuredDiff, "add"),
    deletions: countStructuredDiffLines(structuredDiff, "delete"),
    diffLines: buildUnifiedDiffLines(options.path, options.kind, structuredDiff),
    structuredDiff,
  };
}

function analyzeLineRangeToolFileChange(
  options: BuildToolFileChangePreviewOptions,
  beforeLines: ReadonlyArray<string>,
  afterLines: ReadonlyArray<string>,
): ToolFileDiffAnalysis {
  const prefix = countCommonPrefix(beforeLines, afterLines);
  const suffix = countCommonSuffix(beforeLines, afterLines, prefix);

  const changedBefore = beforeLines.slice(prefix, beforeLines.length - suffix);
  const changedAfter = afterLines.slice(prefix, afterLines.length - suffix);

  const contextStart = Math.max(0, prefix - DIFF_CONTEXT_LINES);
  const afterContextEnd = Math.min(afterLines.length - suffix, afterLines.length);
  const leadingContext = beforeLines.slice(contextStart, prefix);
  const trailingContext = afterLines.slice(
    afterLines.length - suffix,
    Math.min(afterContextEnd + DIFF_CONTEXT_LINES, afterLines.length),
  );

  const oldCount = leadingContext.length + changedBefore.length + trailingContext.length;
  const newCount = leadingContext.length + changedAfter.length + trailingContext.length;
  const oldStart = oldCount === 0 ? 0 : contextStart + 1;
  const newStart = newCount === 0 ? 0 : contextStart + 1;

  return {
    additions: changedAfter.length,
    deletions: changedBefore.length,
    diffLines: [
      options.kind === "create" ? "--- /dev/null" : `--- a/${options.path}`,
      options.kind === "delete" ? "+++ /dev/null" : `+++ b/${options.path}`,
      `@@ -${oldStart},${oldCount} +${newStart},${newCount} @@`,
      ...leadingContext.map((line) => ` ${line}`),
      ...changedBefore.map((line) => `-${line}`),
      ...changedAfter.map((line) => `+${line}`),
      ...trailingContext.map((line) => ` ${line}`),
    ],
  };
}

function buildUnifiedDiffLines(
  path: string,
  kind: ToolFileChangePreview["kind"],
  structuredDiff: ToolFileStructuredDiff,
): ReadonlyArray<string> {
  const lines: string[] = [
    kind === "create" ? "--- /dev/null" : `--- a/${path}`,
    kind === "delete" ? "+++ /dev/null" : `+++ b/${path}`,
  ];

  for (const hunk of structuredDiff.hunks) {
    lines.push(`@@ -${hunk.oldStart},${hunk.oldLines} +${hunk.newStart},${hunk.newLines} @@`);
    for (const line of hunk.lines) {
      lines.push(
        `${line.type === "add" ? "+" : line.type === "delete" ? "-" : " "}${line.text}`,
      );
    }
  }

  return lines;
}

function countStructuredDiffLines(
  structuredDiff: ToolFileStructuredDiff,
  type: ToolFileChangeLine["type"],
): number {
  let count = 0;
  for (const hunk of structuredDiff.hunks) {
    for (const line of hunk.lines) {
      if (line.type === type) count++;
    }
  }
  return count;
}

function numberDiffLines(
  operations: ReadonlyArray<DiffOperation>,
): ReadonlyArray<ToolFileChangeLine> {
  const lines: ToolFileChangeLine[] = [];
  let oldLine = 1;
  let newLine = 1;

  for (const operation of operations) {
    if (operation.type === "context") {
      lines.push({
        type: "context",
        text: operation.text,
        oldLine,
        newLine,
      });
      oldLine++;
      newLine++;
      continue;
    }

    if (operation.type === "delete") {
      lines.push({
        type: "delete",
        text: operation.text,
        oldLine,
        newLine: null,
      });
      oldLine++;
      continue;
    }

    lines.push({
      type: "add",
      text: operation.text,
      oldLine: null,
      newLine,
    });
    newLine++;
  }

  return lines;
}

function groupLinesIntoHunks(
  lines: ReadonlyArray<ToolFileChangeLine>,
): ReadonlyArray<ToolFileChangeHunk> {
  const changeIndexes = lines
    .map((line, index) => (line.type === "context" ? -1 : index))
    .filter((index) => index >= 0);

  if (changeIndexes.length === 0) {
    return lines.length > 0 ? [buildHunkFromLines(lines)] : [];
  }

  const hunks: ToolFileChangeHunk[] = [];
  let start = Math.max(0, changeIndexes[0]! - DIFF_CONTEXT_LINES);
  let end = Math.min(lines.length - 1, changeIndexes[0]! + DIFF_CONTEXT_LINES);

  for (let index = 1; index < changeIndexes.length; index++) {
    const changeIndex = changeIndexes[index]!;
    const nextStart = Math.max(0, changeIndex - DIFF_CONTEXT_LINES);
    const nextEnd = Math.min(lines.length - 1, changeIndex + DIFF_CONTEXT_LINES);

    if (nextStart <= end + 1) {
      end = Math.max(end, nextEnd);
      continue;
    }

    hunks.push(buildHunkFromLines(lines.slice(start, end + 1)));
    start = nextStart;
    end = nextEnd;
  }

  hunks.push(buildHunkFromLines(lines.slice(start, end + 1)));
  return hunks;
}

function buildHunkFromLines(
  lines: ReadonlyArray<ToolFileChangeLine>,
): ToolFileChangeHunk {
  const oldNumbers = lines
    .map((line) => line.oldLine)
    .filter((line): line is number => line !== null);
  const newNumbers = lines
    .map((line) => line.newLine)
    .filter((line): line is number => line !== null);

  const oldStart = oldNumbers.length > 0 ? oldNumbers[0]! : 0;
  const newStart = newNumbers.length > 0 ? newNumbers[0]! : 0;
  const oldLines = oldNumbers.length > 0 ? (oldNumbers[oldNumbers.length - 1]! - oldStart + 1) : 0;
  const newLines = newNumbers.length > 0 ? (newNumbers[newNumbers.length - 1]! - newStart + 1) : 0;

  return {
    oldStart,
    oldLines,
    newStart,
    newLines,
    lines,
  };
}

function parseHunkHeader(line: string): {
  readonly oldStart: number;
  readonly newStart: number;
} {
  const match = /^@@ -(\d+)(?:,\d+)? \+(\d+)(?:,\d+)? @@/.exec(line);
  if (!match) {
    return { oldStart: 0, newStart: 0 };
  }
  return {
    oldStart: Number(match[1]),
    newStart: Number(match[2]),
  };
}
function parseToolFileChangePreview(entry: unknown): ToolFileChangePreview | null {
  if (!isToolFileChangePreviewRecord(entry)) return null;
  const record = entry as Record<string, unknown>;
  const kind = record["kind"] as ToolFileChangePreview["kind"];

  const before = record["before"];
  const after = record["after"];
  const structuredDiff =
    parseToolFileStructuredDiff(record["structuredDiff"])
    ?? (typeof before === "string" && typeof after === "string"
      ? buildToolFileStructuredDiff(before, after)
      : buildToolFileStructuredDiffFromUnifiedDiff(record["unifiedDiff"] as string));

  return {
    path: record["path"] as string,
    kind,
    additions: record["additions"] as number,
    deletions: record["deletions"] as number,
    unifiedDiff: record["unifiedDiff"] as string,
    truncated: record["truncated"] as boolean,
    ...(structuredDiff ? { structuredDiff } : {}),
    ...(typeof before === "string" && typeof after === "string" ? { before, after } : {}),
  };
}

function isToolFileChangePreviewRecord(entry: unknown): entry is Record<string, unknown> & {
  readonly kind: ToolFileChangePreview["kind"];
} {
  if (!entry || typeof entry !== "object") return false;
  const record = entry as Record<string, unknown>;
  return typeof record["path"] === "string" &&
    isToolFileChangeKind(record["kind"]) &&
    typeof record["additions"] === "number" &&
    typeof record["deletions"] === "number" &&
    typeof record["unifiedDiff"] === "string" &&
    typeof record["truncated"] === "boolean";
}

function isToolFileChangeKind(kind: unknown): kind is ToolFileChangePreview["kind"] {
  return kind === "create" || kind === "update" || kind === "delete" || kind === "move";
}
function parseToolFileStructuredDiff(entry: unknown): ToolFileStructuredDiff | undefined {
  const rawHunks = getStructuredDiffHunks(entry);
  if (!rawHunks) {
    return undefined;
  }

  const hunks: ToolFileChangeHunk[] = [];
  for (const rawHunk of rawHunks) {
    const hunk = parseToolFileStructuredHunk(rawHunk);
    if (!hunk) return undefined;
    hunks.push(hunk);
  }

  return { hunks };
}

function getStructuredDiffHunks(entry: unknown): ReadonlyArray<unknown> | null {
  if (!entry || typeof entry !== "object") return null;
  const hunks = (entry as Record<string, unknown>)["hunks"];
  return Array.isArray(hunks) ? hunks : null;
}

function parseToolFileStructuredHunk(rawHunk: unknown): ToolFileChangeHunk | null {
  if (!isStructuredHunkRecord(rawHunk)) return null;
  const record = rawHunk as Record<string, unknown>;
  const lines = parseToolFileStructuredLines(record["lines"] as ReadonlyArray<unknown>);
  if (!lines) return null;
  return {
    oldStart: record["oldStart"] as number,
    oldLines: record["oldLines"] as number,
    newStart: record["newStart"] as number,
    newLines: record["newLines"] as number,
    lines,
  };
}

function isStructuredHunkRecord(rawHunk: unknown): boolean {
  if (!rawHunk || typeof rawHunk !== "object") return false;
  const record = rawHunk as Record<string, unknown>;
  return typeof record["oldStart"] === "number" &&
    typeof record["oldLines"] === "number" &&
    typeof record["newStart"] === "number" &&
    typeof record["newLines"] === "number" &&
    Array.isArray(record["lines"]);
}

function parseToolFileStructuredLines(
  rawLines: ReadonlyArray<unknown>,
): ToolFileChangeLine[] | null {
  const lines: ToolFileChangeLine[] = [];
  for (const rawLine of rawLines) {
    const line = parseToolFileStructuredLine(rawLine);
    if (!line) return null;
    lines.push(line);
  }
  return lines;
}

function parseToolFileStructuredLine(rawLine: unknown): ToolFileChangeLine | null {
  if (!rawLine || typeof rawLine !== "object") return null;
  const record = rawLine as Record<string, unknown>;
  const type = record["type"];
  const oldLine = record["oldLine"];
  const newLine = record["newLine"];
  if (!isToolFileChangeLineType(type) || typeof record["text"] !== "string") return null;
  if (!isStructuredLineNumber(oldLine) || !isStructuredLineNumber(newLine)) return null;
  return {
    type,
    text: record["text"],
    oldLine: oldLine ?? null,
    newLine: newLine ?? null,
  };
}

function isToolFileChangeLineType(type: unknown): type is ToolFileChangeLine["type"] {
  return type === "context" || type === "add" || type === "delete";
}

function isStructuredLineNumber(value: unknown): value is number | null | undefined {
  return value === undefined || value === null || typeof value === "number";
}
