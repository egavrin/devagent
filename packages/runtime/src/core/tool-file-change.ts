import { Buffer } from "node:buffer";
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
const MAX_LCS_CELLS = 200_000;
const TRUNCATION_MARKER = "... diff truncated ...";

export interface BuildToolFileChangePreviewOptions {
  readonly path: string;
  readonly kind: ToolFileChangePreview["kind"];
  readonly before: string;
  readonly after: string;
  readonly maxDiffLines?: number;
  readonly maxDiffBytes?: number;
  readonly maxSnapshotBytes?: number;
}

export interface ToolFileChangePreviewSummary {
  readonly fileEdits?: ReadonlyArray<ToolFileChangePreview>;
  readonly hiddenFileCount: number;
}

interface ToolFileDiffAnalysis {
  readonly additions: number;
  readonly deletions: number;
  readonly diffLines: ReadonlyArray<string>;
  readonly structuredDiff?: ToolFileStructuredDiff;
}

interface DiffOperation {
  readonly type: "context" | "add" | "delete";
  readonly text: string;
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
    if (line.length === 0 || line.startsWith("---") || line.startsWith("+++")) {
      continue;
    }
    if (line === TRUNCATION_MARKER) {
      continue;
    }
    if (line.startsWith("@@")) {
      flushHunk();
      const parsed = parseHunkHeader(line);
      oldLine = parsed.oldStart;
      newLine = parsed.newStart;
      continue;
    }
    if (line.startsWith("+")) {
      currentLines.push({
        type: "add",
        text: line.slice(1),
        oldLine: null,
        newLine,
      });
      newLine = (newLine ?? 0) + 1;
      continue;
    }
    if (line.startsWith("-")) {
      currentLines.push({
        type: "delete",
        text: line.slice(1),
        oldLine,
        newLine: null,
      });
      oldLine = (oldLine ?? 0) + 1;
      continue;
    }
    currentLines.push({
      type: "context",
      text: line.startsWith(" ") ? line.slice(1) : line,
      oldLine,
      newLine,
    });
    oldLine = (oldLine ?? 0) + 1;
    newLine = (newLine ?? 0) + 1;
  }

  flushHunk();

  return hunks.length > 0 ? { hunks } : undefined;
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
    const additions = countStructuredDiffLines(structuredDiff, "add");
    const deletions = countStructuredDiffLines(structuredDiff, "delete");
    return {
      additions,
      deletions,
      diffLines: buildUnifiedDiffLines(options.path, options.kind, structuredDiff),
      structuredDiff,
    };
  }

  const beforeLines = splitFileLines(options.before);
  const afterLines = splitFileLines(options.after);

  let prefix = 0;
  while (
    prefix < beforeLines.length &&
    prefix < afterLines.length &&
    beforeLines[prefix] === afterLines[prefix]
  ) {
    prefix++;
  }

  let suffix = 0;
  while (
    suffix < beforeLines.length - prefix &&
    suffix < afterLines.length - prefix &&
    beforeLines[beforeLines.length - 1 - suffix] === afterLines[afterLines.length - 1 - suffix]
  ) {
    suffix++;
  }

  const changedBefore = beforeLines.slice(prefix, beforeLines.length - suffix);
  const changedAfter = afterLines.slice(prefix, afterLines.length - suffix);

  const contextStart = Math.max(0, prefix - DIFF_CONTEXT_LINES);
  const beforeContextEnd = Math.min(beforeLines.length - suffix, beforeLines.length);
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

function diffSnapshotLines(
  before: ReadonlyArray<string>,
  after: ReadonlyArray<string>,
): ReadonlyArray<DiffOperation> | null {
  let prefix = 0;
  while (prefix < before.length && prefix < after.length && before[prefix] === after[prefix]) {
    prefix++;
  }

  let suffix = 0;
  while (
    suffix < before.length - prefix &&
    suffix < after.length - prefix &&
    before[before.length - 1 - suffix] === after[after.length - 1 - suffix]
  ) {
    suffix++;
  }

  const beforeMiddle = before.slice(prefix, before.length - suffix);
  const afterMiddle = after.slice(prefix, after.length - suffix);
  const middle = buildMiddleDiffOperations(beforeMiddle, afterMiddle);
  if (!middle) return null;

  return [
    ...before.slice(0, prefix).map((text) => ({ type: "context", text }) satisfies DiffOperation),
    ...middle,
    ...before.slice(before.length - suffix).map((text) => ({ type: "context", text }) satisfies DiffOperation),
  ];
}

function buildMiddleDiffOperations(
  before: ReadonlyArray<string>,
  after: ReadonlyArray<string>,
): ReadonlyArray<DiffOperation> | null {
  if (before.length === 0) {
    return after.map((text) => ({ type: "add", text }) satisfies DiffOperation);
  }
  if (after.length === 0) {
    return before.map((text) => ({ type: "delete", text }) satisfies DiffOperation);
  }

  if ((before.length + 1) * (after.length + 1) > MAX_LCS_CELLS) {
    return null;
  }

  const dp = Array.from({ length: before.length + 1 }, () => new Uint32Array(after.length + 1));

  for (let oldIndex = before.length - 1; oldIndex >= 0; oldIndex--) {
    for (let newIndex = after.length - 1; newIndex >= 0; newIndex--) {
      dp[oldIndex]![newIndex] = before[oldIndex] === after[newIndex]
        ? dp[oldIndex + 1]![newIndex + 1]! + 1
        : Math.max(dp[oldIndex + 1]![newIndex]!, dp[oldIndex]![newIndex + 1]!);
    }
  }

  const operations: DiffOperation[] = [];
  let oldIndex = 0;
  let newIndex = 0;

  while (oldIndex < before.length && newIndex < after.length) {
    if (before[oldIndex] === after[newIndex]) {
      operations.push({ type: "context", text: before[oldIndex]! });
      oldIndex++;
      newIndex++;
      continue;
    }

    if (dp[oldIndex + 1]![newIndex]! >= dp[oldIndex]![newIndex + 1]!) {
      operations.push({ type: "delete", text: before[oldIndex]! });
      oldIndex++;
      continue;
    }

    operations.push({ type: "add", text: after[newIndex]! });
    newIndex++;
  }

  while (oldIndex < before.length) {
    operations.push({ type: "delete", text: before[oldIndex]! });
    oldIndex++;
  }
  while (newIndex < after.length) {
    operations.push({ type: "add", text: after[newIndex]! });
    newIndex++;
  }

  return operations;
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
  if (
    !entry ||
    typeof entry !== "object" ||
    typeof (entry as Record<string, unknown>)["path"] !== "string" ||
    typeof (entry as Record<string, unknown>)["kind"] !== "string" ||
    typeof (entry as Record<string, unknown>)["additions"] !== "number" ||
    typeof (entry as Record<string, unknown>)["deletions"] !== "number" ||
    typeof (entry as Record<string, unknown>)["unifiedDiff"] !== "string" ||
    typeof (entry as Record<string, unknown>)["truncated"] !== "boolean"
  ) {
    return null;
  }

  const kind = (entry as Record<string, unknown>)["kind"];
  if (kind !== "create" && kind !== "update" && kind !== "delete" && kind !== "move") {
    return null;
  }

  const before = (entry as Record<string, unknown>)["before"];
  const after = (entry as Record<string, unknown>)["after"];
  const structuredDiff =
    parseToolFileStructuredDiff((entry as Record<string, unknown>)["structuredDiff"])
    ?? (typeof before === "string" && typeof after === "string"
      ? buildToolFileStructuredDiff(before, after)
      : buildToolFileStructuredDiffFromUnifiedDiff((entry as Record<string, unknown>)["unifiedDiff"] as string));

  return {
    path: (entry as Record<string, unknown>)["path"] as string,
    kind,
    additions: (entry as Record<string, unknown>)["additions"] as number,
    deletions: (entry as Record<string, unknown>)["deletions"] as number,
    unifiedDiff: (entry as Record<string, unknown>)["unifiedDiff"] as string,
    truncated: (entry as Record<string, unknown>)["truncated"] as boolean,
    ...(structuredDiff ? { structuredDiff } : {}),
    ...(typeof before === "string" && typeof after === "string" ? { before, after } : {}),
  };
}

function parseToolFileStructuredDiff(entry: unknown): ToolFileStructuredDiff | undefined {
  if (!entry || typeof entry !== "object" || !Array.isArray((entry as Record<string, unknown>)["hunks"])) {
    return undefined;
  }

  const hunks: ToolFileChangeHunk[] = [];
  for (const rawHunk of (entry as Record<string, unknown>)["hunks"] as ReadonlyArray<unknown>) {
    if (
      !rawHunk ||
      typeof rawHunk !== "object" ||
      typeof (rawHunk as Record<string, unknown>)["oldStart"] !== "number" ||
      typeof (rawHunk as Record<string, unknown>)["oldLines"] !== "number" ||
      typeof (rawHunk as Record<string, unknown>)["newStart"] !== "number" ||
      typeof (rawHunk as Record<string, unknown>)["newLines"] !== "number" ||
      !Array.isArray((rawHunk as Record<string, unknown>)["lines"])
    ) {
      return undefined;
    }

    const lines: ToolFileChangeLine[] = [];
    for (const rawLine of (rawHunk as Record<string, unknown>)["lines"] as ReadonlyArray<unknown>) {
      if (
        !rawLine ||
        typeof rawLine !== "object" ||
        typeof (rawLine as Record<string, unknown>)["type"] !== "string" ||
        typeof (rawLine as Record<string, unknown>)["text"] !== "string"
      ) {
        return undefined;
      }

      const type = (rawLine as Record<string, unknown>)["type"];
      const oldLine = (rawLine as Record<string, unknown>)["oldLine"];
      const newLine = (rawLine as Record<string, unknown>)["newLine"];
      if (type !== "context" && type !== "add" && type !== "delete") {
        return undefined;
      }
      if ((oldLine !== null && typeof oldLine !== "number") || (newLine !== null && typeof newLine !== "number")) {
        return undefined;
      }

      lines.push({
        type,
        text: (rawLine as Record<string, unknown>)["text"] as string,
        oldLine: (oldLine as number | null | undefined) ?? null,
        newLine: (newLine as number | null | undefined) ?? null,
      });
    }

    hunks.push({
      oldStart: (rawHunk as Record<string, unknown>)["oldStart"] as number,
      oldLines: (rawHunk as Record<string, unknown>)["oldLines"] as number,
      newStart: (rawHunk as Record<string, unknown>)["newStart"] as number,
      newLines: (rawHunk as Record<string, unknown>)["newLines"] as number,
      lines,
    });
  }

  return { hunks };
}
