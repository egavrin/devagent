/**
 * Bridge module for bun:sqlite.
 *
 * Vitest uses Vite's module resolver which cannot handle the `bun:` protocol.
 * This shim tries a dynamic import with a runtime-computed specifier; if that
 * fails (e.g. on Linux CI where Vite's ESM loader rejects bun: even for
 * dynamic imports), it falls back to a stub class.  Tests that need real
 * sqlite use `BUN_SQLITE_AVAILABLE` to skip gracefully.
 */

let _Database: unknown;
let _available = false;

try {
  // Build specifier at runtime so Vite cannot statically match it
  const _specifier = ["bun", "sqlite"].join(":");
  const _mod = await import(_specifier);
  _Database = _mod.Database;
  _available = true;
} catch {
  // bun:sqlite not loadable in this environment — provide a stub that throws
  // a clear error if anyone tries to instantiate it.
  _Database = class StubDatabase {
    constructor() {
      throw new Error(
        "bun:sqlite is not available in this environment. " +
          "Tests requiring SQLite should be skipped with BUN_SQLITE_AVAILABLE.",
      );
    }
  };
}

/** Re-exported Database constructor (real bun:sqlite or stub). */
export const Database = _Database as typeof import("bun:sqlite").Database;

/** Instance type so callers can use `Database` in type position. */
export type Database = InstanceType<typeof Database>;

/** `true` when the real bun:sqlite module loaded successfully. */
export const BUN_SQLITE_AVAILABLE: boolean = _available;
