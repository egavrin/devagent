/**
 * Canonical language-to-extension mapping shared across the codebase.
 *
 * Every package that needs to resolve file extensions for a language ID
 * should import from here instead of maintaining its own copy.
 */

/** All language IDs that have a known extension mapping. */
export type KnownLanguageId =
  | "typescript"
  | "javascript"
  | "python"
  | "c"
  | "cpp"
  | "rust"
  | "shellscript"
  | "arkts";

/**
 * Maps language identifiers to their recognised file extensions.
 * Keys are VS Code / LSP-style language IDs (lowercase).
 *
 * Typed with exact keys so that `LANGUAGE_EXTENSIONS.typescript` is
 * `readonly string[]` (not `readonly string[] | undefined`), while
 * still allowing dynamic string-key lookups via the index signature.
 */
export const LANGUAGE_EXTENSIONS: {
  readonly [K in KnownLanguageId]: readonly string[];
} & { readonly [key: string]: readonly string[] | undefined } = {
  typescript: [".ts", ".tsx", ".mts", ".cts"],
  javascript: [".js", ".jsx", ".mjs", ".cjs"],
  python: [".py", ".pyi"],
  c: [".c", ".h"],
  cpp: [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh"],
  rust: [".rs"],
  shellscript: [".sh", ".bash", ".zsh"],
  arkts: [".ets"],
};

/**
 * Detect a language ID from a file extension.
 * Returns `null` when the extension is not recognised.
 */
export function detectLanguageFromExtension(ext: string): string | null {
  const lower = ext.toLowerCase();
  for (const [langId, exts] of Object.entries(LANGUAGE_EXTENSIONS)) {
    if (exts?.includes(lower)) return langId;
  }
  return null;
}
