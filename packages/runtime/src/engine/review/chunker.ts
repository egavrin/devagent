/**
 * Chunking logic for splitting large patches into manageable pieces for LLM review.
 * Ported from the Python chunking implementation.
 */

import { estimateTokens } from "../../core/index.js";
import type {
  FileEntry,
  Hunk,
  ParsedPatch,
} from "../../tools/builtins/patch-parser.js";

// ── Configuration ────────────────────────────────────────────────────────────

export interface ReviewConfig {
  maxLinesPerChunk: number;
  maxHunksPerChunk: number;
  maxFilesPerChunk: number;
  chunkOverlapLines: number;
  tokenBudget: number;
}

export const DEFAULT_REVIEW_CONFIG: ReviewConfig = {
  maxLinesPerChunk: 1500,
  maxHunksPerChunk: 50,
  maxFilesPerChunk: 50,
  chunkOverlapLines: 100,
  tokenBudget: 80_000,
};

// ── Impact estimation ────────────────────────────────────────────────────────

/**
 * Sum of added + removed + context lines in a single hunk.
 */
function estimateHunkImpact(hunk: Hunk): number {
  const added = hunk.addedLines?.length ?? 0;
  const removed = hunk.removedLines?.length ?? 0;
  const context = hunk.contextLines?.length ?? 0;
  return added + removed + context;
}

/**
 * Total line impact across all hunks in a file entry.
 */
function estimateEntryLines(fileEntry: FileEntry): number {
  const hunks = fileEntry.hunks ?? [];
  return hunks.reduce((sum, h) => sum + estimateHunkImpact(h), 0);
}

// ── Large-file splitting ─────────────────────────────────────────────────────

/**
 * Split files that have many hunks into smaller groups, annotating each with
 * `_chunkIndex` and `_chunkTotal` metadata so the reviewer knows a file was
 * segmented.
 */
export function splitLargeFileEntries(
  files: FileEntry[],
  config: Pick<ReviewConfig, "maxHunksPerChunk" | "maxLinesPerChunk">,
): FileEntry[] {
  const maxHunksPerGroup = config.maxHunksPerChunk;
  const maxLinesPerGroup = config.maxLinesPerChunk;

  if (maxHunksPerGroup <= 0 && maxLinesPerGroup <= 0) {
    return [...files];
  }

  const results: FileEntry[] = [];

  for (const fileEntry of files) {
    const hunks: Hunk[] = fileEntry.hunks ?? [];

    if (hunks.length === 0) {
      results.push(fileEntry);
      continue;
    }

    const groups: Hunk[][] = [];
    let current: Hunk[] = [];
    let hunkCounter = 0;
    let lineCounter = 0;

    for (const hunk of hunks) {
      const impact = estimateHunkImpact(hunk);
      const shouldSplit =
        current.length > 0 &&
        ((maxHunksPerGroup > 0 && hunkCounter >= maxHunksPerGroup) ||
          (maxLinesPerGroup > 0 && lineCounter + impact > maxLinesPerGroup));

      if (shouldSplit) {
        groups.push(current);
        current = [];
        hunkCounter = 0;
        lineCounter = 0;
      }

      current.push(hunk);
      hunkCounter += 1;
      lineCounter += impact;
    }

    if (current.length > 0) {
      groups.push(current);
    }

    if (groups.length <= 1) {
      results.push(fileEntry);
      continue;
    }

    const total = groups.length;
    for (let index = 0; index < groups.length; index++) {
      const group = groups[index]!;
      const newEntry: FileEntry = {
        ...fileEntry,
        hunks: group,
        _chunkIndex: index,
        _chunkTotal: total,
      };
      results.push(newEntry);
    }
  }

  return results;
}

// ── Chunk grouping ───────────────────────────────────────────────────────────

/**
 * Group file entries into chunks respecting per-chunk file count and line
 * limits.  Optionally applies overlap between adjacent chunks.
 */
export function chunkPatchFiles(
  files: FileEntry[],
  config: Pick<
    ReviewConfig,
    "maxFilesPerChunk" | "maxLinesPerChunk" | "chunkOverlapLines"
  >,
): FileEntry[][] {
  if (files.length === 0) return [];

  const maxFilesPerChunk = config.maxFilesPerChunk;
  const maxLinesPerChunk = config.maxLinesPerChunk;
  const overlapLines = config.chunkOverlapLines;

  const effectiveMaxFiles =
    maxFilesPerChunk > 0 ? maxFilesPerChunk : files.length;
  const effectiveMaxLines = maxLinesPerChunk > 0 ? maxLinesPerChunk : 0;

  const chunked: FileEntry[][] = [];
  let currentChunk: FileEntry[] = [];
  let currentLines = 0;

  for (const entry of files) {
    const entryLines = estimateEntryLines(entry);
    const shouldSplit =
      currentChunk.length > 0 &&
      ((effectiveMaxFiles > 0 &&
        currentChunk.length >= effectiveMaxFiles) ||
        (effectiveMaxLines > 0 &&
          currentLines + entryLines > effectiveMaxLines));

    if (shouldSplit) {
      chunked.push(currentChunk);
      currentChunk = [];
      currentLines = 0;
    }

    currentChunk.push(entry);
    currentLines += entryLines;
  }

  if (currentChunk.length > 0) {
    chunked.push(currentChunk);
  }

  if (overlapLines > 0 && chunked.length > 1) {
    return applyChunkOverlap(chunked, overlapLines);
  }

  return chunked;
}

// ── Chunk overlap ────────────────────────────────────────────────────────────

/**
 * Copy boundary files from the tail of chunk N into the head of chunk N+1 so
 * the reviewer has surrounding context.
 */
function applyChunkOverlap(
  chunks: FileEntry[][],
  overlapLines: number,
): FileEntry[][] {
  if (chunks.length <= 1 || overlapLines <= 0) {
    return chunks;
  }

  const overlapped: FileEntry[][] = [chunks[0]!];

  for (let i = 1; i < chunks.length; i++) {
    const prevChunk = chunks[i - 1]!;
    const currentChunk = [...chunks[i]!];

    const overlapFiles: FileEntry[] = [];
    let accumulatedLines = 0;

    for (let j = prevChunk.length - 1; j >= 0; j--) {
      const fileEntry = prevChunk[j]!;
      const fileLines = estimateEntryLines(fileEntry);

      if (accumulatedLines + fileLines > overlapLines && overlapFiles.length > 0) {
        break;
      }

      overlapFiles.unshift(fileEntry);
      accumulatedLines += fileLines;
    }

    const currentPaths = new Set(
      currentChunk.filter((f) => f.path).map((f) => f.path),
    );

    for (const overlapFile of overlapFiles) {
      if (!currentPaths.has(overlapFile.path)) {
        currentChunk.unshift(overlapFile);
      }
    }

    overlapped.push(currentChunk);
  }

  return overlapped;
}

// ── Dynamic limit computation ────────────────────────────────────────────────

/**
 * Shrink the per-chunk line limit when the total context (rule text + patch
 * size) is very large, preventing individual chunks from exceeding a
 * practical token ceiling.
 */
export function computeDynamicLineLimit(
  configuredLimit: number,
  ruleText: string,
  files: FileEntry[],
): number {
  const DEFAULT = 1500;
  let base = configuredLimit > 0 ? configuredLimit : DEFAULT;

  const total = files.reduce((s, e) => s + estimateEntryLines(e), 0);
  const ruleTokens = estimateTokens(ruleText);
  const combined = ruleTokens + total * 2;

  if (combined > 150_000) {
    base = Math.min(base, 400);
  } else if (combined > 100_000) {
    base = Math.min(base, 800);
  } else if (combined > 75_000) {
    base = Math.min(base, 1200);
  }

  if (configuredLimit <= 0 && base === DEFAULT) return 0;
  return base;
}

/**
 * Shrink the per-chunk file limit for massive contexts.
 */
export function computeDynamicFileLimit(
  configuredLimit: number,
  ruleText: string,
  files: FileEntry[],
): number {
  const DEFAULT = 50;
  let base = configuredLimit > 0 ? configuredLimit : DEFAULT;

  const total = files.reduce((s, e) => s + estimateEntryLines(e), 0);
  const ruleTokens = estimateTokens(ruleText);
  const combined = ruleTokens + total * 2;

  if (combined > 150_000) {
    base = Math.min(base, 10);
  } else if (combined > 100_000) {
    base = Math.min(base, 20);
  } else if (combined > 75_000) {
    base = Math.min(base, 35);
  }

  if (configuredLimit <= 0 && base === DEFAULT) return files.length || 1;
  return Math.max(1, base);
}

// ── Token estimation ─────────────────────────────────────────────────────────

/**
 * Estimate prompt tokens across multiple optional segments.
 * Delegates to the canonical `estimateTokens` from `@devagent/runtime`
 * (Math.ceil(length / 4)) so engine and core stay in sync.
 */
function estimatePromptTokens(...segments: (string | undefined | null)[]): number {
  let total = 0;
  for (const s of segments) {
    if (s) total += estimateTokens(s);
  }
  return total;
}

// ── Token-budget refinement ──────────────────────────────────────────────────

/**
 * Binary-split any chunk whose estimated prompt tokens exceed `tokenBudget`
 * until every chunk fits within budget (or is down to a single file).
 */
export function refineChunksForTokenBudget(
  chunks: FileEntry[][],
  ruleText: string,
  filterPattern: string | undefined,
  tokenBudget: number,
): FileEntry[][] {
  const refined: FileEntry[][] = [];

  for (const chunk of chunks) {
    const queue: FileEntry[][] = [chunk];

    while (queue.length > 0) {
      const current = queue.shift()!;
      const datasetText = formatPatchDataset(
        { patchInfo: {}, files: current, summary: { totalFiles: 0, filesAdded: 0, filesModified: 0, filesDeleted: 0, totalAdditions: 0, totalDeletions: 0 } },
        filterPattern,
      );
      const approx = estimatePromptTokens(ruleText, datasetText);

      if (approx > tokenBudget && current.length > 1) {
        const mid = Math.max(1, Math.floor(current.length / 2));
        // Prepend both halves (right first, then left in front) so left is
        // processed next -- mirrors Python deque.appendleft ordering.
        queue.unshift(current.slice(0, mid), current.slice(mid));
        continue;
      }

      refined.push(current);
    }
  }

  return refined;
}

// ── Patch formatting ─────────────────────────────────────────────────────────

/**
 * Render a parsed patch as human-readable text suitable for LLM consumption.
 * Optionally filters to files matching `filterPattern` (regex).
 */
export function formatPatchDataset(
  parsedPatch: ParsedPatch,
  filterPattern?: string,
): string {
  const lines: string[] = [];
  const files = parsedPatch.files ?? [];

  if (files.length === 0) {
    lines.push("No files with additions were detected in this patch.");
    return lines.join("\n");
  }

  let filteredFiles = files;
  if (filterPattern) {
    try {
      const patternRe = new RegExp(filterPattern);
      filteredFiles = files.filter((f) => f.path && patternRe.test(f.path));
    } catch {
      filteredFiles = files;
    }
  }

  if (filteredFiles.length === 0) {
    lines.push(
      `No files matching pattern '${filterPattern}' were found in this patch.`,
    );
    return lines.join("\n");
  }

  for (const fileEntry of filteredFiles) {
    const path = fileEntry.path;
    const changeType = fileEntry.changeType ?? "modified";
    const language = fileEntry.language ?? "unknown";
    const chunkIndex = fileEntry._chunkIndex;
    const chunkTotal = fileEntry._chunkTotal;

    let chunkSuffix = "";
    if (
      chunkIndex !== undefined &&
      chunkTotal !== undefined &&
      chunkTotal > 1
    ) {
      chunkSuffix = ` (segment ${chunkIndex + 1}/${chunkTotal})`;
    }

    lines.push(`FILE: ${path}${chunkSuffix}`);
    lines.push(`  Change type: ${changeType}`);
    lines.push(`  Language: ${language}`);

    const hunks = fileEntry.hunks ?? [];
    const totalAdded = hunks.reduce(
      (s, h) => s + (h.addedLines?.length ?? 0),
      0,
    );
    const totalRemoved = hunks.reduce(
      (s, h) => s + (h.removedLines?.length ?? 0),
      0,
    );

    lines.push(`  Total added lines: ${totalAdded}`);
    lines.push(`  Total removed lines: ${totalRemoved}`);

    if (hunks.length === 0) {
      lines.push("  HUNK: (none)");
      lines.push("    CONTEXT:\n      (none)");
      lines.push("    ADDED LINES:\n      (none)");
      lines.push("    REMOVED LINES:\n      (none)");
      lines.push("");
      continue;
    }

    for (const hunk of hunks) {
      const header = (hunk.header ?? "").trim();
      lines.push(`  HUNK: ${header || "(no header)"}`);

      // Context lines
      const context = hunk.contextLines ?? [];
      lines.push("    CONTEXT:");
      if (context.length > 0) {
        for (const ctx of context) {
          const ln = ctx.lineNumber;
          const c = ctx.content ?? "";
          if (typeof ln === "number") {
            lines.push(`      ${String(ln).padStart(4)} | ${c}`);
          } else {
            lines.push(`      | ${c}`);
          }
        }
      } else {
        lines.push("      (none)");
      }

      // Added lines
      const added = hunk.addedLines ?? [];
      lines.push("    ADDED LINES:");
      if (added.length > 0) {
        for (const e of added) {
          const ln = e.lineNumber;
          const c = e.content ?? "";
          if (typeof ln === "number") {
            lines.push(`      ${String(ln).padStart(4)} + ${c}`);
          } else {
            lines.push(`      + ${c}`);
          }
        }
      } else {
        lines.push("      (none)");
      }

      // Removed lines
      const removed = hunk.removedLines ?? [];
      lines.push("    REMOVED LINES:");
      if (removed.length > 0) {
        for (const e of removed) {
          const ln = e.lineNumber;
          const c = e.content ?? "";
          if (typeof ln === "number") {
            lines.push(`      ${String(ln).padStart(4)} - ${c}`);
          } else {
            lines.push(`      - ${c}`);
          }
        }
      } else {
        lines.push("      (none)");
      }

      lines.push("");
    }

    lines.push("");
  }

  return lines.join("\n").trimEnd();
}
