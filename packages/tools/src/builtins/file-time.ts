/**
 * FileTime — Track when files were read/written to enforce pre-read discipline.
 *
 * Inspired by OpenCode's file/time.ts: tools that modify files MUST read them
 * first. This prevents stale mental models and wasted iterations.
 *
 * Module-level singleton (reset per session via FileTime.reset()).
 */

import { statSync } from "node:fs";

/** Per-file read timestamps. */
const readTimes = new Map<string, Date>();

/** Per-file write timestamps (set after successful writes via this module). */
const writeTimes = new Map<string, Date>();

export const FileTime = {
  /**
   * Record that a file was read. Called by read_file.
   */
  recordRead(filePath: string): void {
    readTimes.set(filePath, new Date());
  },

  /**
   * Record that a file was written by a tool. Called after write_file / replace_in_file.
   * This updates the "last known" timestamp so subsequent edits don't trigger
   * the "modified externally" error for our own writes.
   */
  recordWrite(filePath: string): void {
    writeTimes.set(filePath, new Date());
  },

  /**
   * Check if a file was read before editing it.
   * Throws if the file was never read or was modified externally since last read.
   */
  assert(filePath: string): void {
    const readTime = readTimes.get(filePath);
    if (!readTime) {
      throw new Error(
        `You must read file ${filePath} before editing it. Use read_file first.`,
      );
    }

    // Check if the file was modified externally since our last read/write
    try {
      const stats = statSync(filePath);
      const lastKnown = writeTimes.get(filePath) ?? readTime;

      // Allow 1 second tolerance for filesystem timestamp granularity
      if (stats.mtimeMs > lastKnown.getTime() + 1000) {
        throw new Error(
          `File ${filePath} has been modified since it was last read. Read it again before editing.`,
        );
      }
    } catch (err) {
      // If stat fails (file deleted), let the tool handler deal with it
      if (err instanceof Error && err.message.includes("modified since")) {
        throw err;
      }
    }
  },

  /**
   * Check if a file was read (without throwing). Used by write_file for
   * existing files only — new files don't need pre-read.
   */
  wasRead(filePath: string): boolean {
    return readTimes.has(filePath);
  },

  /**
   * Reset all tracking. Called at the start of each session/run.
   */
  reset(): void {
    readTimes.clear();
    writeTimes.clear();
  },
};
