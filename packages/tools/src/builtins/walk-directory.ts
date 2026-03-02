/**
 * walk-directory — Shared recursive directory traversal generator.
 *
 * Provides a single source of truth for walking a directory tree while
 * skipping common non-project directories (node_modules, .git, dist, .cache).
 */

import { readdirSync, type Stats, statSync } from "node:fs";
import { join, relative } from "node:path";

/** Directories that are always skipped during traversal. */
export const IGNORED_DIRS: ReadonlyArray<string> = [
  "node_modules",
  ".git",
  "dist",
  ".cache",
];

const ignoredSet = new Set<string>(IGNORED_DIRS);

export interface WalkEntry {
  /** Absolute path to the file. */
  readonly fullPath: string;
  /** Path relative to `repoRoot`. */
  readonly relativePath: string;
  /** `fs.Stats` for the entry (from `statSync`). */
  readonly stat: Stats;
}

export interface WalkOptions {
  /** Stop yielding after this many results. */
  maxResults?: number;
}

/**
 * Recursively walk `dir`, yielding every *file* entry (not directories).
 *
 * - Skips directories listed in {@link IGNORED_DIRS}.
 * - Silently skips entries/directories that cannot be read (permission errors, etc.).
 * - Respects an optional `maxResults` cap.
 *
 * @param dir       The directory to start walking (absolute path).
 * @param repoRoot  The repository root used to compute `relativePath`.
 * @param opts      Optional configuration.
 */
export function* walkDirectory(
  dir: string,
  repoRoot: string,
  opts?: WalkOptions,
): Generator<WalkEntry> {
  const maxResults = opts?.maxResults ?? Infinity;
  let yielded = 0;

  function* walk(current: string): Generator<WalkEntry> {
    if (yielded >= maxResults) return;

    let entries: string[];
    try {
      entries = readdirSync(current);
    } catch {
      return; // Skip directories we can't read
    }

    for (const entry of entries) {
      if (yielded >= maxResults) return;

      if (ignoredSet.has(entry)) continue;

      const fullPath = join(current, entry);
      let stat: Stats;
      try {
        stat = statSync(fullPath);
      } catch {
        continue;
      }

      if (stat.isDirectory()) {
        yield* walk(fullPath);
      } else {
        const relativePath = relative(repoRoot, fullPath);
        yielded++;
        yield { fullPath, relativePath, stat };
      }
    }
  }

  yield* walk(dir);
}
