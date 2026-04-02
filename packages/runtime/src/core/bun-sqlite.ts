/**
 * Bridge module for SQLite.
 *
 * Fallback chain:
 *   1. bun:sqlite (Bun runtime — built-in, fastest)
 *   2. better-sqlite3 (Node.js — native addon, npm dependency)
 *   3. Stub that throws a clear error
 */

import { createRequire } from "node:module";

const require = createRequire(import.meta.url);

let _Database: unknown;
let _available = false;

try {
  if (typeof Bun !== "undefined") {
    // Resolve through require() so the constructor is available synchronously.
    const _mod = require("bun:sqlite") as typeof import("bun:sqlite");
    _Database = _mod.Database;
    _available = true;
  } else {
    const _mod = require("better-sqlite3") as { default?: unknown } | unknown;
    _Database = (_mod as { default?: unknown }).default ?? _mod;
    _available = true;
  }
} catch {
  try {
    // Bun can still fall back to better-sqlite3 if bun:sqlite is unavailable.
    const _mod = require("better-sqlite3") as { default?: unknown } | unknown;
    _Database = (_mod as { default?: unknown }).default ?? _mod;
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
