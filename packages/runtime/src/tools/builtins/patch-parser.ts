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
    // Find next diff header
    let oldPath: string | undefined;
    let newPath: string | undefined;

    while (this.currentLine < this.lines.length) {
      const line = this.lines[this.currentLine]!;
      const match = DIFF_HEADER.exec(line);
      if (match) {
        oldPath = match[1];
        newPath = match[2];
        this.currentLine += 1;
        break;
      }
      this.currentLine += 1;
    }

    if (oldPath === undefined || newPath === undefined) {
      return null;
    }

    // Find old file header
    let oldFile: string | null = null;
    while (this.currentLine < this.lines.length) {
      const line = this.lines[this.currentLine]!;
      const match = FILE_HEADER_OLD.exec(line);
      if (match) {
        oldFile = match[1]!;
        this.currentLine += 1;
        break;
      }
      // If we hit the next diff header before finding file headers, bail
      if (DIFF_HEADER.test(line)) {
        break;
      }
      this.currentLine += 1;
    }

    // Find new file header
    let newFile: string | null = null;
    while (this.currentLine < this.lines.length) {
      const line = this.lines[this.currentLine]!;
      const match = FILE_HEADER_NEW.exec(line);
      if (match) {
        newFile = match[1]!;
        this.currentLine += 1;
        break;
      }
      if (DIFF_HEADER.test(line)) {
        break;
      }
      this.currentLine += 1;
    }

    // Determine change type
    let changeType: ChangeType = "modified";
    let actualPath: string;

    if (oldFile === "/dev/null") {
      changeType = "added";
      actualPath = newPath;
    } else if (newFile === "/dev/null") {
      changeType = "deleted";
      actualPath = oldPath;
    } else if (oldPath !== newPath) {
      changeType = "renamed";
      actualPath = newPath;
    } else {
      actualPath = newPath;
    }

    // Apply filter
    if (filterPattern !== null) {
      const regex = new RegExp(filterPattern);
      if (!regex.test(actualPath)) {
        // Skip remaining hunk lines for this file
        while (this.currentLine < this.lines.length) {
          if (DIFF_HEADER.test(this.lines[this.currentLine]!)) {
            break;
          }
          this.currentLine += 1;
        }
        return null;
      }
    }

    // Parse hunks
    const hunks: Hunk[] = [];
    let additions = 0;
    let deletions = 0;

    while (this.currentLine < this.lines.length) {
      const line = this.lines[this.currentLine]!;
      if (DIFF_HEADER.test(line)) {
        break;
      }
      const hunk = this.parseHunk();
      if (hunk) {
        hunks.push(hunk);
        additions += hunk.addedLines.length;
        deletions += hunk.removedLines.length;
      } else {
        this.currentLine += 1;
      }
    }

    return {
      path: actualPath,
      changeType,
      oldPath: changeType === "renamed" ? oldPath : null,
      language: detectLanguage(actualPath),
      hunks,
      stats: {
        additions,
        deletions,
        totalHunks: hunks.length,
      },
    };
  }

  // ── Hunk parsing ────────────────────────────────────────────────────────

  private parseHunk(): Hunk | null {
    if (this.currentLine >= this.lines.length) {
      return null;
    }

    const line = this.lines[this.currentLine]!;
    const match = HUNK_HEADER.exec(line);
    if (!match) {
      return null;
    }

    const oldStart = parseInt(match[1]!, 10);
    const oldCount = match[2] !== undefined ? parseInt(match[2], 10) : 1;
    const newStart = parseInt(match[3]!, 10);
    const newCount = match[4] !== undefined ? parseInt(match[4], 10) : 1;
    const headerContext = (match[5] ?? "").trim();
    this.currentLine += 1;

    const addedLines: AddedLine[] = [];
    const removedLines: RemovedLine[] = [];
    const contextLines: ContextLine[] = [];
    let newLineNum = newStart;
    let oldLineNum = oldStart;

    while (this.currentLine < this.lines.length) {
      const hunkLine = this.lines[this.currentLine]!;

      // Check for termination conditions
      if (hunkLine.startsWith("@@") || hunkLine.startsWith("diff --git")) {
        break;
      }
      if (hunkLine.startsWith("---") && FILE_HEADER_OLD.test(hunkLine)) {
        break;
      }
      if (hunkLine.startsWith("+++") && FILE_HEADER_NEW.test(hunkLine)) {
        break;
      }

      if (hunkLine.startsWith("+") && !hunkLine.startsWith("+++")) {
        const content = hunkLine.slice(1);
        addedLines.push({
          lineNumber: newLineNum,
          content,
          indentation: leadingWhitespace(content),
        });
        newLineNum += 1;
      } else if (hunkLine.startsWith("-") && !hunkLine.startsWith("---")) {
        removedLines.push({
          lineNumber: oldLineNum,
          content: hunkLine.slice(1),
        });
        oldLineNum += 1;
      } else if (hunkLine.startsWith(" ") || hunkLine === "") {
        if (this.includeContext) {
          const content = hunkLine.length > 0 ? hunkLine.slice(1) : "";
          contextLines.push({
            lineNumber: newLineNum,
            content,
          });
        }
        newLineNum += 1;
        oldLineNum += 1;
      } else if (hunkLine.startsWith("\\")) {
        // "\ No newline at end of file" -- skip
      } else {
        break;
      }
      this.currentLine += 1;
    }

    const result: Hunk = {
      header: `@@ -${oldStart},${oldCount} +${newStart},${newCount} @@ ${headerContext}`,
      oldStart,
      oldCount,
      newStart,
      newCount,
      addedLines,
      removedLines,
    };

    if (this.includeContext) {
      result.contextLines = contextLines;
    }

    return result;
  }
}
