/**
 * Bridge module for SQLite.
 *
 * Fallback chain:
 *   1. bun:sqlite (Bun runtime — built-in, fastest)
 *   2. better-sqlite3 (Node.js — native addon, npm dependency)
 *   3. Stub that throws a clear error
 */

let _Database: unknown;
let _available = false;

try {
  // Build specifier at runtime so bundlers cannot statically match it
  const _specifier = ["bun", "sqlite"].join(":");
  const _mod = await import(_specifier);
  _Database = _mod.Database;
  _available = true;
} catch {
  try {
    // Node.js fallback: better-sqlite3 has a near-identical sync API
    const _mod = await import("better-sqlite3");
    _Database = _mod.default;
    _available = true;
  } catch {
    // No SQLite available — provide a stub that throws on instantiation
    _Database = class StubDatabase {
      constructor() {
        throw new Error(
          "No SQLite library available. " +
            "Install better-sqlite3 (npm i better-sqlite3) or use Bun runtime.",
        );
      }
    };
  }
}

/** Re-exported Database constructor (bun:sqlite, better-sqlite3, or stub). */
export const Database = _Database as typeof import("bun:sqlite").Database;

/** Instance type so callers can use `Database` in type position. */
export type Database = InstanceType<typeof Database>;

/** `true` when a real SQLite library loaded successfully. */
export const BUN_SQLITE_AVAILABLE: boolean = _available;
