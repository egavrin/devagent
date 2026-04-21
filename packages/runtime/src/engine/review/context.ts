/**
 * Context enrichment -- reads surrounding source code for changed hunks
 * to provide the reviewer with additional context.
 * Ported from the Python SourceContextProvider and ContextOrchestrator.
 */

import { readFileSync, statSync } from "node:fs";
import { resolve } from "node:path";

import type { FileEntry, Hunk } from "../../tools/builtins/patch-parser.js";

// ── ContextItem ─────────────────────────────────────────────────────────────

export interface ContextItem {
  kind: string;
  title: string;
  path: string | null;
  span: [number, number] | null;
  body: string;
}

function contextItemLineCount(item: ContextItem): number {
  return item.body ? item.body.split("\n").length : 0;
}

// ── ContextProvider protocol ────────────────────────────────────────────────

export interface ContextProvider {
  buildItems(workspaceRoot: string, fileEntry: FileEntry): ContextItem[];
}

// ── SourceContextProvider ───────────────────────────────────────────────────

interface CacheEntry {
  mtime: number;
  size: number;
  lines: string[];
}

export class SourceContextProvider implements ContextProvider {
  private readonly padLines: number;
  private readonly maxLinesPerItem: number;
  private readonly cache = new Map<string, CacheEntry>();

  constructor(options?: { padLines?: number; maxLinesPerItem?: number }) {
    this.padLines = options?.padLines ?? 20;
    this.maxLinesPerItem = options?.maxLinesPerItem ?? 160;
  }
  buildItems(workspaceRoot: string, fileEntry: FileEntry): ContextItem[] {
    const relPath = fileEntry.path;
    if (typeof relPath !== "string") return [];

    const absPath = resolve(workspaceRoot, relPath);
    const lines = this.getFileLines(absPath);
    if (!lines || lines.length === 0) return [];

    const ranges = this.collectRanges(fileEntry.hunks ?? []);
    if (ranges.length === 0) return [];

    const merged = SourceContextProvider.mergeRanges(ranges, lines.length);
    const items: ContextItem[] = [];
    let totalLines = 0;

    for (const [start, end] of merged) {
      const snippet = buildSourceContextSnippet(lines, start, end);
      if (!snippet) continue;

      totalLines += snippet.lineCount;
      if (totalLines > this.maxLinesPerItem && items.length > 0) break;

      items.push({
        kind: "code",
        title: `${relPath}:${snippet.start}-${snippet.end}`,
        path: relPath,
        span: [snippet.start, snippet.end],
        body: snippet.body,
      });
    }

    return items;
  }

  private getFileLines(absPath: string): string[] | null {
    let stat: { mtimeMs: number; size: number };
    try {
      const s = statSync(absPath);
      stat = { mtimeMs: s.mtimeMs, size: s.size };
    } catch {
      return null;
    }

    const cached = this.cache.get(absPath);
    if (cached && cached.mtime === stat.mtimeMs && cached.size === stat.size) {
      return cached.lines;
    }

    try {
      const content = readFileSync(absPath, "utf-8");
      const lines = content.split("\n");
      this.cache.set(absPath, { mtime: stat.mtimeMs, size: stat.size, lines });
      return lines;
    } catch {
      return null;
    }
  }

  private collectRanges(hunks: Hunk[]): Array<[number, number]> {
    const ranges: Array<[number, number]> = [];

    for (const hunk of hunks) {
      let begin: number;
      let end: number;

      if (typeof hunk.newStart === "number" && typeof hunk.newCount === "number" && hunk.newCount > 0) {
        begin = hunk.newStart;
        end = hunk.newStart + Math.max(hunk.newCount - 1, 0);
      } else if (typeof hunk.oldStart === "number" && typeof hunk.oldCount === "number" && hunk.oldCount > 0) {
        begin = hunk.oldStart;
        end = hunk.oldStart + Math.max(hunk.oldCount - 1, 0);
      } else {
        continue;
      }

      ranges.push([begin - this.padLines, end + this.padLines]);
    }

    return ranges;
  }

  static mergeRanges(
    ranges: Array<[number, number]>,
    maxLine: number,
  ): Array<[number, number]> {
    if (ranges.length === 0) return [];

    const ordered = [...ranges].sort((a, b) => a[0] - b[0]);
    const merged: Array<[number, number]> = [];
    let [currentStart, currentEnd] = ordered[0]!;

    for (let i = 1; i < ordered.length; i++) {
      const [start, end] = ordered[i]!;
      if (start <= currentEnd + 1) {
        currentEnd = Math.max(currentEnd, end);
      } else {
        merged.push([Math.max(1, currentStart), Math.min(maxLine, currentEnd)]);
        currentStart = start;
        currentEnd = end;
      }
    }

    merged.push([Math.max(1, currentStart), Math.min(maxLine, currentEnd)]);
    return merged;
  }
}

function buildSourceContextSnippet(
  lines: ReadonlyArray<string>,
  start: number,
  end: number,
): { readonly start: number; readonly end: number; readonly lineCount: number; readonly body: string } | null {
  const clampedStart = Math.max(1, start);
  const clampedEnd = Math.min(lines.length, end);
  const snippetLines: string[] = [];

  for (let idx = clampedStart; idx <= clampedEnd; idx++) {
    snippetLines.push(`${String(idx).padStart(5)}: ${lines[idx - 1]}`);
  }

  return snippetLines.length === 0
    ? null
    : {
      start: clampedStart,
      end: clampedEnd,
      lineCount: snippetLines.length,
      body: snippetLines.join("\n"),
    };
}

// ── ContextOrchestrator ─────────────────────────────────────────────────────

export class ContextOrchestrator {
  private readonly providers: ContextProvider[];
  private readonly maxTotalLines: number;

  constructor(
    providers: ContextProvider[],
    options?: { maxTotalLines?: number },
  ) {
    this.providers = [...providers];
    this.maxTotalLines = options?.maxTotalLines ?? 320;
  }
  buildSection(workspaceRoot: string, fileEntries: FileEntry[]): string {
    const collected = this.collectContextItems(workspaceRoot, fileEntries);
    const trimmed = trimContextItems(collected, this.maxTotalLines);
    return trimmed.length > 0 ? formatContextItems(trimmed) : "";
  }

  private collectContextItems(workspaceRoot: string, fileEntries: FileEntry[]): ContextItem[] {
    const collected: ContextItem[] = [];
    const seenKeys = new Set<string>();
    for (const entry of fileEntries) {
      for (const provider of this.providers) {
        addProviderItems(collected, seenKeys, provider, workspaceRoot, entry);
      }
    }
    return collected;
  }
}

function addProviderItems(
  collected: ContextItem[],
  seenKeys: Set<string>,
  provider: ContextProvider,
  workspaceRoot: string,
  entry: FileEntry,
): void {
  let items: ContextItem[];
  try {
    items = provider.buildItems(workspaceRoot, entry);
  } catch {
    return;
  }
  for (const item of items) {
    const key = `${item.kind}|${item.path}|${item.span?.[0]},${item.span?.[1]}|${item.body}`;
    if (seenKeys.has(key)) continue;
    seenKeys.add(key);
    collected.push(item);
  }
}

function trimContextItems(items: ReadonlyArray<ContextItem>, maxTotalLines: number): ContextItem[] {
  const trimmed: ContextItem[] = [];
  let totalLines = 0;
  for (const item of items) {
    const lines = contextItemLineCount(item);
    if (totalLines + lines > maxTotalLines && trimmed.length > 0) {
      continue;
    }
    trimmed.push(item);
    totalLines += lines;
  }
  return trimmed;
}

// ── Formatting ──────────────────────────────────────────────────────────────

function formatContextItems(items: ContextItem[]): string {
  const lines: string[] = ["Context:"];

  for (let idx = 0; idx < items.length; idx++) {
    const item = items[idx]!;
    lines.push(`#${idx + 1} ${item.kind.toUpperCase()}: ${item.title}`);
    if (item.body) {
      lines.push(item.body.trimEnd());
    }
    lines.push("");
  }

  return lines.join("\n").trim();
}
