/**
 * PatchParser -- unified diff parser.
 * Ported from the Python PatchParser implementation.
 */

import { posix } from "node:path";

// ── Interfaces ──────────────────────────────────────────────────────────────

export interface PatchInfo {
  commit?: string;
  author?: string;
  date?: string;
  message?: string;
}

export interface AddedLine {
  lineNumber: number;
  content: string;
  indentation: string;
}

export interface RemovedLine {
  lineNumber: number;
  content: string;
}

export interface ContextLine {
  lineNumber: number;
  content: string;
}

export interface Hunk {
  header: string;
  oldStart: number;
  oldCount: number;
  newStart: number;
  newCount: number;
  addedLines: AddedLine[];
  removedLines: RemovedLine[];
  contextLines?: ContextLine[];
}

export interface FileStats {
  additions: number;
  deletions: number;
  totalHunks: number;
}

export type ChangeType = "added" | "modified" | "deleted" | "renamed";

export interface FileEntry {
  path: string;
  changeType: ChangeType;
  oldPath: string | null;
  language: string;
  hunks: Hunk[];
  stats: FileStats;
  _chunkIndex?: number;
  _chunkTotal?: number;
}

export interface PatchSummary {
  totalFiles: number;
  filesAdded: number;
  filesModified: number;
  filesDeleted: number;
  totalAdditions: number;
  totalDeletions: number;
}

export interface ParsedPatch {
  patchInfo: PatchInfo;
  files: FileEntry[];
  summary: PatchSummary;
}

// ── Regex patterns ──────────────────────────────────────────────────────────

const DIFF_HEADER = /^diff --git a\/(.*) b\/(.*)$/;
const FILE_HEADER_OLD = /^--- (?:a\/)?(.*)$/;
const FILE_HEADER_NEW = /^\+\+\+ (?:b\/)?(.*)$/;
const HUNK_HEADER = /^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$/;
const COMMIT_HEADER = /^From ([0-9a-f]{40}) /;

interface DiffHeaderPaths {
  readonly oldPath: string;
  readonly newPath: string;
}

interface ChangeInfo {
  readonly changeType: ChangeType;
  readonly actualPath: string;
  readonly oldPath: string | null;
}

interface HunkCursor {
  newLineNum: number;
  oldLineNum: number;
}

// ── Helpers ─────────────────────────────────────────────────────────────────

function leadingWhitespace(s: string): string {
  const match = /^(\s*)/.exec(s);
  return match?.[1] ?? "";
}

const EXT_MAP: Record<string, string> = {
  ".py": "python",
  ".js": "javascript",
  ".ts": "typescript",
  ".java": "java",
  ".cpp": "cpp",
  ".c": "c",
  ".h": "c",
  ".hpp": "cpp",
  ".go": "go",
  ".rs": "rust",
  ".rb": "ruby",
  ".sh": "shell",
  ".md": "markdown",
};

function detectLanguage(filePath: string): string {
  const ext = posix.extname(filePath).toLowerCase();
  const mapped = EXT_MAP[ext];
  if (mapped !== undefined) {
    return mapped;
  }
  return ext.length > 1 ? ext.slice(1) : "unknown";
}

function computeSummary(files: FileEntry[]): PatchSummary {
  const summary: PatchSummary = {
    totalFiles: files.length,
    filesAdded: 0,
    filesModified: 0,
    filesDeleted: 0,
    totalAdditions: 0,
    totalDeletions: 0,
  };
  for (const f of files) {
    if (f.changeType === "added") {
      summary.filesAdded += 1;
    } else if (f.changeType === "deleted") {
      summary.filesDeleted += 1;
    } else {
      summary.filesModified += 1;
    }
    summary.totalAdditions += f.stats.additions;
    summary.totalDeletions += f.stats.deletions;
  }
  return summary;
}

function getChangeInfo(
  paths: DiffHeaderPaths,
  oldFile: string | null,
  newFile: string | null,
): ChangeInfo {
  if (oldFile === "/dev/null") {
    return { changeType: "added", actualPath: paths.newPath, oldPath: null };
  }
  if (newFile === "/dev/null") {
    return { changeType: "deleted", actualPath: paths.oldPath, oldPath: null };
  }
  if (paths.oldPath !== paths.newPath) {
    return { changeType: "renamed", actualPath: paths.newPath, oldPath: paths.oldPath };
  }
  return { changeType: "modified", actualPath: paths.newPath, oldPath: null };
}

function parseHunkHeader(line: string): {
  readonly oldStart: number;
  readonly oldCount: number;
  readonly newStart: number;
  readonly newCount: number;
  readonly headerContext: string;
} | null {
  const match = HUNK_HEADER.exec(line);
  if (!match) return null;
  return {
    oldStart: parseInt(match[1]!, 10),
    oldCount: match[2] !== undefined ? parseInt(match[2], 10) : 1,
    newStart: parseInt(match[3]!, 10),
    newCount: match[4] !== undefined ? parseInt(match[4], 10) : 1,
    headerContext: (match[5] ?? "").trim(),
  };
}

function isHunkTerminator(line: string): boolean {
  return line.startsWith("@@") ||
    line.startsWith("diff --git") ||
    (line.startsWith("---") && FILE_HEADER_OLD.test(line)) ||
    (line.startsWith("+++") && FILE_HEADER_NEW.test(line));
}

function consumeHunkLine(
  line: string,
  cursor: HunkCursor,
  target: {
    readonly addedLines: AddedLine[];
    readonly removedLines: RemovedLine[];
    readonly contextLines: ContextLine[];
    readonly includeContext: boolean;
  },
): boolean {
  if (line.startsWith("+") && !line.startsWith("+++")) {
    addHunkAddedLine(line, cursor, target.addedLines);
    return true;
  }
  if (line.startsWith("-") && !line.startsWith("---")) {
    addHunkRemovedLine(line, cursor, target.removedLines);
    return true;
  }
  if (line.startsWith(" ") || line === "") {
    addHunkContextLine(line, cursor, target.contextLines, target.includeContext);
    return true;
  }
  return line.startsWith("\\");
}

function addHunkAddedLine(
  line: string,
  cursor: HunkCursor,
  addedLines: AddedLine[],
): void {
  const content = line.slice(1);
  addedLines.push({
    lineNumber: cursor.newLineNum,
    content,
    indentation: leadingWhitespace(content),
  });
  cursor.newLineNum += 1;
}

function addHunkRemovedLine(
  line: string,
  cursor: HunkCursor,
  removedLines: RemovedLine[],
): void {
  removedLines.push({
    lineNumber: cursor.oldLineNum,
    content: line.slice(1),
  });
  cursor.oldLineNum += 1;
}

function addHunkContextLine(
  line: string,
  cursor: HunkCursor,
  contextLines: ContextLine[],
  includeContext: boolean,
): void {
  if (includeContext) {
    contextLines.push({
      lineNumber: cursor.newLineNum,
      content: line.length > 0 ? line.slice(1) : "",
    });
  }
  cursor.newLineNum += 1;
  cursor.oldLineNum += 1;
}

// ── PatchParser ─────────────────────────────────────────────────────────────

export class PatchParser {
  private readonly lines: string[];
  private readonly includeContext: boolean;
  private currentLine: number;

  constructor(patchContent: string, includeContext = false) {
    this.lines = patchContent.split("\n");
    this.includeContext = includeContext;
    this.currentLine = 0;
  }

  parse(filterPattern?: string | null): ParsedPatch {
    const patchInfo = this.extractCommitInfo();
    const files: FileEntry[] = [];

    while (this.currentLine < this.lines.length) {
      const fileEntry = this.parseFile(filterPattern ?? null);
      if (fileEntry) {
        files.push(fileEntry);
      }
    }

    return {
      patchInfo,
      files,
      summary: computeSummary(files),
    };
  }

  // ── Commit info ─────────────────────────────────────────────────────────

  private extractCommitInfo(): PatchInfo {
    const info: PatchInfo = {};
    const limit = Math.min(20, this.lines.length);

    for (let i = 0; i < limit; i++) {
      const line = this.lines[i]!;
      const commitMatch = COMMIT_HEADER.exec(line);
      if (commitMatch) {
        info.commit = commitMatch[1];
      } else if (line.startsWith("From: ")) {
        info.author = line.slice(6).trim();
      } else if (line.startsWith("Date: ")) {
        info.date = line.slice(6).trim();
      } else if (line.startsWith("Subject: ")) {
        info.message = line.slice(9).trim();
      }
    }

    return info;
  }

  // ── File parsing ────────────────────────────────────────────────────────
  private parseFile(filterPattern: string | null): FileEntry | null {
    const diffHeader = this.findNextDiffHeader();
    if (!diffHeader) return null;
    const oldFile = this.findFileHeader(FILE_HEADER_OLD);
    const newFile = this.findFileHeader(FILE_HEADER_NEW);
    const change = getChangeInfo(diffHeader, oldFile, newFile);
    if (!this.matchesFilter(change.actualPath, filterPattern)) return null;
    const { hunks, additions, deletions } = this.parseFileHunks();

    return {
      path: change.actualPath,
      changeType: change.changeType,
      oldPath: change.oldPath,
      language: detectLanguage(change.actualPath),
      hunks,
      stats: {
        additions,
        deletions,
        totalHunks: hunks.length,
      },
    };
  }

  private findNextDiffHeader(): DiffHeaderPaths | null {
    while (this.currentLine < this.lines.length) {
      const match = DIFF_HEADER.exec(this.lines[this.currentLine]!);
      this.currentLine += 1;
      if (match) return { oldPath: match[1]!, newPath: match[2]! };
    }
    return null;
  }

  private findFileHeader(pattern: RegExp): string | null {
    while (this.currentLine < this.lines.length) {
      const line = this.lines[this.currentLine]!;
      const match = pattern.exec(line);
      if (match) {
        this.currentLine += 1;
        return match[1]!;
      }
      if (DIFF_HEADER.test(line)) break;
      this.currentLine += 1;
    }
    return null;
  }

  private matchesFilter(path: string, filterPattern: string | null): boolean {
    if (filterPattern === null || new RegExp(filterPattern).test(path)) return true;
    this.skipToNextFile();
    return false;
  }

  private skipToNextFile(): void {
    while (this.currentLine < this.lines.length && !DIFF_HEADER.test(this.lines[this.currentLine]!)) {
      this.currentLine += 1;
    }
  }

  private parseFileHunks(): { readonly hunks: Hunk[]; readonly additions: number; readonly deletions: number } {
    const hunks: Hunk[] = [];
    let additions = 0;
    let deletions = 0;

    while (this.currentLine < this.lines.length && !DIFF_HEADER.test(this.lines[this.currentLine]!)) {
      const hunk = this.parseHunk();
      if (!hunk) {
        this.currentLine += 1;
        continue;
      }
      hunks.push(hunk);
      additions += hunk.addedLines.length;
      deletions += hunk.removedLines.length;
    }
    return { hunks, additions, deletions };
  }

  // ── Hunk parsing ────────────────────────────────────────────────────────
  private parseHunk(): Hunk | null {
    if (this.currentLine >= this.lines.length) {
      return null;
    }

    const parsedHeader = parseHunkHeader(this.lines[this.currentLine]!);
    if (!parsedHeader) return null;
    this.currentLine += 1;

    const addedLines: AddedLine[] = [];
    const removedLines: RemovedLine[] = [];
    const contextLines: ContextLine[] = [];
    const cursor: HunkCursor = {
      newLineNum: parsedHeader.newStart,
      oldLineNum: parsedHeader.oldStart,
    };

    while (this.currentLine < this.lines.length) {
      const hunkLine = this.lines[this.currentLine]!;
      if (isHunkTerminator(hunkLine)) break;
      if (!consumeHunkLine(hunkLine, cursor, {
        addedLines,
        removedLines,
        contextLines,
        includeContext: this.includeContext,
      })) break;
      this.currentLine += 1;
    }

    const result: Hunk = {
      header: `@@ -${parsedHeader.oldStart},${parsedHeader.oldCount} +${parsedHeader.newStart},${parsedHeader.newCount} @@ ${parsedHeader.headerContext}`,
      oldStart: parsedHeader.oldStart,
      oldCount: parsedHeader.oldCount,
      newStart: parsedHeader.newStart,
      newCount: parsedHeader.newCount,
      addedLines,
      removedLines,
    };

    if (this.includeContext) {
      result.contextLines = contextLines;
    }

    return result;
  }
}
