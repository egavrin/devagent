/**
 * Tool summary formatters — pure/stateless functions that produce compact
 * summaries of tool results for session state persistence.
 *
 * Extracted from task-loop.ts to keep the core loop focused on orchestration.
 */

import { SUMMARY_MAX_CHARS } from "./session-state.js";

// ─── Types ──────────────────────────────────────────────────

/**
 * Minimal tool call shape needed by formatting functions.
 * Compatible with the PendingToolCall interface in task-loop.ts.
 */
interface ToolCallInfo {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
}

// ─── Truncation ─────────────────────────────────────────────

/** Truncate text to SUMMARY_MAX_CHARS with trailing "..." when exceeded. */
function truncateToSummary(text: string): string {
  return text.length <= SUMMARY_MAX_CHARS
    ? text
    : text.slice(0, SUMMARY_MAX_CHARS - 3) + "...";
}

// ─── Top-level Router ───────────────────────────────────────

/**
 * Format a structured summary for a tool result.
 * For replace_in_file, includes search->replace details and count.
 * For other tools, falls back to truncated output.
 */
export function formatToolSummary(
  toolCall: ToolCallInfo,
  originalOutput: string,
): string {

  if (toolCall.name === "replace_in_file") {
    const search = toolCall.arguments["search"] as string | undefined;
    const replace = toolCall.arguments["replace"] as string | undefined;
    const replacements = toolCall.arguments["replacements"] as unknown[] | undefined;
    // Extract count from output like "Replaced 4 occurrence(s)" or "Applied 3 replacement(s)"
    const countMatch = originalOutput.match(/(\d+)\s+(?:replacement|occurrence)/);
    const count = countMatch ? countMatch[1] : "?";

    if (replacements && Array.isArray(replacements)) {
      return `batch: ${replacements.length} pairs (${count} total replacements)`;
    }

    if (search && replace) {
      // Truncate search/replace to keep summary compact
      const s = search.length > 40 ? search.slice(0, 37) + "..." : search;
      const r = replace.length > 40 ? replace.slice(0, 37) + "..." : replace;
      return `'${s}' → '${r}' (${count} occurrences)`;
    }
  }

  if (toolCall.name === "write_file") {
    const path = toolCall.arguments["path"] as string | undefined;
    if (path) return `Wrote ${path}`;
  }

  // Readonly tools: compact summaries to avoid bloating session state
  if (toolCall.name === "read_file") {
    const lines = originalOutput.split("\n");
    const lineCount = lines.length;
    const startLine = toolCall.arguments["start_line"] as number | undefined;
    const endLine = toolCall.arguments["end_line"] as number | undefined;
    const rangeHint = startLine !== undefined || endLine !== undefined
      ? ` (lines ${startLine ?? 1}-${endLine ?? "end"})`
      : "";
    const digest = extractStructuralDigest(originalOutput, 1000);
    // Include content context: first and last non-blank lines for orientation
    const contentSnippets = extractContentSnippets(lines, 500);
    const parts = [`Read ${lineCount} lines${rangeHint}`];
    if (digest) parts.push(digest);
    if (contentSnippets) parts.push(contentSnippets);
    return parts.join(": ");
  }

  if (toolCall.name === "search_files") {
    return formatSearchFilesSummary(toolCall, originalOutput);
  }

  if (toolCall.name === "find_files") {
    return formatFindFilesSummary(toolCall, originalOutput);
  }

  if (toolCall.name === "git_diff") {
    return summarizeDiff(originalOutput);
  }

  if (toolCall.name === "git_status") {
    return formatGitStatusSummary(originalOutput);
  }

  if (toolCall.name === "run_command") {
    return formatRunCommandSummary(toolCall, originalOutput);
  }

  if (toolCall.name === "diagnostics") {
    return formatDiagnosticsSummary(originalOutput);
  }

  if (toolCall.name === "symbols") {
    return formatSymbolsSummary(originalOutput);
  }

  // Default: truncated original output (no DoubleCheck noise)
  return originalOutput.slice(0, SUMMARY_MAX_CHARS);
}

// ─── Per-Tool Formatters ────────────────────────────────────

/**
 * Format search_files summary preserving pattern, file paths, and match lines.
 */
export function formatSearchFilesSummary(
  toolCall: ToolCallInfo,
  originalOutput: string,
): string {
  const pattern = toolCall.arguments["pattern"] as string | undefined;
  const lines = originalOutput.split("\n");
  const nonEmpty = lines.filter((l) => l.trim());

  // Extract header line (e.g., "Found 15 matches for ...")
  const headerMatch = originalOutput.match(/^(Found \d+ match[^\n]*)/);
  const header = headerMatch
    ? headerMatch[1]
    : `${nonEmpty.length} matches for "${pattern ?? "?"}"`;

  // Collect file paths and match lines
  const fileLines: string[] = [];
  const matchLines: string[] = [];
  for (const line of nonEmpty) {
    const trimmed = line.trim();
    if (trimmed.startsWith("Found ")) continue;
    // Match lines typically start with whitespace + line number
    if (/^\s+\d+:/.test(line)) {
      matchLines.push(trimmed);
    } else if (trimmed.length > 0 && !trimmed.startsWith("---")) {
      fileLines.push(trimmed);
    }
  }

  // Match content lines first (higher semantic value), then file list.
  // When truncation at SUMMARY_MAX_CHARS occurs, match content is preserved
  // over file paths — preventing loss of semantic context after compaction.
  const parts = [header];
  for (const ml of matchLines) {
    parts.push(ml);
  }
  if (fileLines.length > 0) {
    parts.push(`Files: ${fileLines.join(", ")}`);
  }

  return truncateToSummary(parts.join("\n"));
}

/**
 * Format find_files summary preserving glob pattern and file paths.
 */
function formatFindFilesSummary(
  toolCall: ToolCallInfo,
  originalOutput: string,
): string {
  const pattern = toolCall.arguments["pattern"] as string | undefined;
  const lines = originalOutput.split("\n").filter((l) => l.trim());

  // Extract header (e.g., "Found 12 files matching ...")
  const headerMatch = originalOutput.match(/^(Found \d+ file[^\n]*)/);
  const filePaths = lines.filter((l) => !l.startsWith("Found "));
  const header = headerMatch
    ? headerMatch[1]
    : `${filePaths.length} files matching "${pattern ?? "?"}"`;

  const parts = [header, ...filePaths];
  return truncateToSummary(parts.join("\n"));
}

/**
 * Format run_command summary with head+tail and test output extraction.
 */
function formatRunCommandSummary(
  toolCall: ToolCallInfo,
  originalOutput: string,
): string {
  const cmd = toolCall.arguments["command"] as string | undefined;

  // Special case: git diff
  if (cmd && /\bgit\s+diff\b/.test(cmd)) {
    return summarizeDiff(originalOutput);
  }

  // Special case: test/typecheck/lint commands
  if (cmd && /\b(?:test|vitest|jest|mocha|pytest|typecheck|tsc|lint|eslint|biome)\b/.test(cmd)) {
    const testSummary = summarizeTestOutput(originalOutput);
    if (testSummary) {
      const prefix = `$ ${cmd}\n`;
      return truncateToSummary(prefix + testSummary);
    }
  }

  // General case: head+tail with command prefix
  const lines = originalOutput.split("\n");
  const prefix = cmd ? `$ ${cmd}\n` : "";

  if (lines.length <= 10) {
    // Short output: keep it all
    return truncateToSummary(prefix + originalOutput);
  }

  // Head (first 5 lines) + tail (last 3 lines)
  const head = lines.slice(0, 5).join("\n");
  const tail = lines.slice(-3).join("\n");
  const omitted = lines.length - 8;
  return truncateToSummary(`${prefix}${head}\n[... ${omitted} lines omitted ...]\n${tail}`);
}

/**
 * Format git_status output into grouped status summary.
 * Groups files by status code (M, A, D, ?, etc.) for compact display.
 */
function formatGitStatusSummary(output: string): string {
  const lines = output.split("\n").filter((l) => l.trim());
  const groups = new Map<string, string[]>();

  for (const line of lines) {
    const trimmed = line.trim();
    // git status --porcelain format: XY filename
    const match = trimmed.match(/^([MADRCU?! ]{1,2})\s+(.+)$/);
    if (match) {
      const statusCode = match[1]!.trim() || "M";
      const fileName = match[2]!.split("/").pop() ?? match[2]!;
      const key = statusCode.startsWith("?") ? "?" : statusCode.charAt(0);
      const existing = groups.get(key) ?? [];
      existing.push(fileName);
      groups.set(key, existing);
    }
  }

  if (groups.size === 0) {
    return `${lines.length} entries`;
  }

  const parts = [`${lines.length} entries`];
  for (const [status, files] of groups) {
    parts.push(`[${status}] ${files.join(", ")}`);
  }

  return truncateToSummary(parts.join("\n"));
}

/**
 * Format diagnostics output with severity counts and diagnostic lines.
 * Prioritizes errors over warnings.
 */
function formatDiagnosticsSummary(output: string): string {
  const lines = output.split("\n").filter((l) => l.trim());
  let errorCount = 0;
  let warningCount = 0;
  const errorLines: string[] = [];
  const warningLines: string[] = [];

  for (const line of lines) {
    const trimmed = line.trim();
    if (/\berror\b/i.test(trimmed)) {
      errorCount++;
      errorLines.push(trimmed);
    } else if (/\bwarning\b/i.test(trimmed)) {
      warningCount++;
      warningLines.push(trimmed);
    }
  }

  const total = errorCount + warningCount;
  if (total === 0) {
    return output.slice(0, SUMMARY_MAX_CHARS);
  }

  const countParts: string[] = [];
  if (errorCount > 0) countParts.push(`${errorCount} errors`);
  if (warningCount > 0) countParts.push(`${warningCount} warnings`);
  const header = `${total} diagnostics (${countParts.join(", ")})`;

  // Include all diagnostic lines, errors first
  const allDiagLines = [...errorLines, ...warningLines];
  const parts = [header, ...allDiagLines];
  return truncateToSummary(parts.join("\n"));
}

/**
 * Format symbols output preserving the symbol list.
 * Symbols are compact (~50 chars each), so 2000 chars fits ~35 symbols.
 */
function formatSymbolsSummary(output: string): string {
  const lines = output.split("\n").filter((l) => l.trim());
  const header = `${lines.length} symbols`;
  const parts = [header, ...lines];
  return truncateToSummary(parts.join("\n"));
}

/**
 * Extract structured summary from test/typecheck/lint output.
 * Returns pass/fail counts, error lines, and failing test names.
 * Returns null if the output doesn't look like test/typecheck output.
 */
export function summarizeTestOutput(output: string): string | null {
  const parts: string[] = [];
  let isTestOutput = false;

  // Vitest/Jest pass/fail counts
  const testCountMatch = output.match(/Tests?:\s*(.+total)/i);
  if (testCountMatch) {
    parts.push(testCountMatch[0]);
    isTestOutput = true;
  }

  // Bun test counts
  const bunTestMatch = output.match(/(\d+)\s+pass(?:ed)?.*?(\d+)\s+fail/i);
  if (!testCountMatch && bunTestMatch) {
    parts.push(`${bunTestMatch[1]} passed, ${bunTestMatch[2]} failed`);
    isTestOutput = true;
  }

  // TypeScript error count
  const tsErrorMatch = output.match(/Found (\d+) errors? in (\d+) files?\./);
  if (tsErrorMatch) {
    parts.push(`${tsErrorMatch[1]} errors in ${tsErrorMatch[2]} files`);
    isTestOutput = true;
  }

  // Individual TS errors (e.g., "error TS2322:")
  const tsErrors: string[] = [];
  for (const line of output.split("\n")) {
    const trimmed = line.trim();
    if (/error TS\d+/.test(trimmed)) {
      tsErrors.push(trimmed);
    }
  }
  if (tsErrors.length > 0) {
    isTestOutput = true;
    for (const e of tsErrors.slice(0, 10)) {
      parts.push(e);
    }
    if (tsErrors.length > 10) {
      parts.push(`... +${tsErrors.length - 10} more errors`);
    }
  }

  // Failing test files (FAIL lines)
  const failLines: string[] = [];
  for (const line of output.split("\n")) {
    const trimmed = line.trim();
    if (/^FAIL\s/.test(trimmed)) {
      failLines.push(trimmed);
    }
  }
  if (failLines.length > 0) {
    isTestOutput = true;
    for (const f of failLines.slice(0, 5)) {
      parts.push(f);
    }
  }

  // Error assertion lines (bullet test name)
  const assertionLines: string[] = [];
  for (const line of output.split("\n")) {
    const trimmed = line.trim();
    if (trimmed.startsWith("\u25CF") || trimmed.startsWith("\u2715") || trimmed.startsWith("\u00D7")) {
      assertionLines.push(trimmed);
    }
  }
  if (assertionLines.length > 0) {
    for (const a of assertionLines.slice(0, 5)) {
      parts.push(a);
    }
  }

  if (!isTestOutput) return null;

  return truncateToSummary(parts.join("\n"));
}

// ─── Structural Digest ───────────────────────────────────────

const STRUCTURAL_LINE_PATTERNS: ReadonlyArray<RegExp> = [
  /^(?:export\s+)?(?:default\s+)?(?:async\s+)?function\s+[A-Za-z_]\w*\s*\(/,
  /^(?:export\s+)?(?:abstract\s+)?class\s+[A-Za-z_]\w*/,
  /^(?:export\s+)?interface\s+[A-Za-z_]\w*/,
  /^(?:export\s+)?type\s+[A-Za-z_]\w*\s*=/,
  /^(?:export\s+)?enum\s+[A-Za-z_]\w*/,
  /^(?:export\s+)?const\s+[A-Za-z_]\w*\s*=\s*(?:async\s*)?(?:\(|<)/,
  /^def\s+[A-Za-z_]\w*\s*\(/,
  /^class\s+[A-Za-z_]\w*(?:\(|:|\s|$)/,
  /^(?:pub\s+)?fn\s+[A-Za-z_]\w*\s*\(/,
  /^(?:pub\s+)?struct\s+[A-Za-z_]\w*/,
  /^impl\s+[A-Za-z_]\w*/,
  /^func\s+[A-Za-z_]\w*\s*\(/,
];

/**
 * Extract a compact structural digest from source text.
 * Captures top-level declaration-like lines so summaries retain
 * semantic anchors after compaction.
 */
export function extractStructuralDigest(
  source: string,
  maxChars: number = 500,
): string {
  if (maxChars <= 0) return "";

  const declarations: string[] = [];
  const seen = new Set<string>();

  for (const line of source.split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;
    if (
      trimmed.startsWith("//")
      || trimmed.startsWith("#")
      || trimmed.startsWith("/*")
      || trimmed.startsWith("*")
    ) {
      continue;
    }
    if (!STRUCTURAL_LINE_PATTERNS.some((pattern) => pattern.test(trimmed))) continue;

    const normalized = trimmed
      .replace(/\s*\{\s*$/, "")
      .replace(/\s+/g, " ");
    const snippet = normalized.length > 80
      ? `${normalized.slice(0, 77)}...`
      : normalized;
    if (seen.has(snippet)) continue;
    seen.add(snippet);
    declarations.push(snippet);
    if (declarations.length >= 20) break;
  }

  if (declarations.length === 0) return "";

  const joined = declarations.join("; ");
  if (joined.length <= maxChars) return joined;
  if (maxChars <= 3) return joined.slice(0, maxChars);
  return `${joined.slice(0, maxChars - 3)}...`;
}

/**
 * Extract a few meaningful non-blank lines from source content for summary context.
 * Captures first 2 and last 2 non-blank, non-comment lines so the LLM
 * retains orientation cues (variable assignments, key logic) beyond just declarations.
 */
function extractContentSnippets(lines: string[], maxChars: number): string {
  const meaningful: string[] = [];
  for (const line of lines) {
    const trimmed = line.replace(/^\d+\t/, "").trim(); // strip line number prefix
    if (!trimmed) continue;
    if (trimmed.startsWith("//") || trimmed.startsWith("#") || trimmed.startsWith("/*") || trimmed.startsWith("*")) continue;
    if (trimmed === "{" || trimmed === "}" || trimmed === ");") continue;
    meaningful.push(trimmed);
  }
  if (meaningful.length === 0) return "";

  const snippets: string[] = [];
  const firstFour = meaningful.slice(0, 4);
  const lastFour = meaningful.length > 8 ? meaningful.slice(-4) : [];
  for (const s of [...firstFour, ...lastFour]) {
    const truncated = s.length > 100 ? s.slice(0, 97) + "..." : s;
    snippets.push(truncated);
  }

  const joined = `[${snippets.join(" | ")}]`;
  return joined.length <= maxChars ? joined : joined.slice(0, maxChars - 3) + "...";
}

// ─── Diff Summary ────────────────────────────────────────────

/**
 * Extract a compact semantic summary from unified diff output.
 * Parses hunk headers to identify modified functions/methods and
 * counts additions/deletions. Returns a human-readable summary
 * that captures WHAT changed, not just how many lines.
 */
export function summarizeDiff(diffOutput: string): string {
  // Check for diffstat first (e.g., "3 files changed, 10 insertions(+), 5 deletions(-)")
  const statMatch = diffOutput.match(/(\d+)\s+files?\s+changed/);
  if (statMatch) {
    const insertions = diffOutput.match(/(\d+)\s+insertions?\(\+\)/);
    const deletions = diffOutput.match(/(\d+)\s+deletions?\(-\)/);
    const parts = [`${statMatch[1]} files changed`];
    if (insertions) parts.push(`+${insertions[1]}`);
    if (deletions) parts.push(`-${deletions[1]}`);
    return parts.join(", ");
  }

  // Parse unified diff: extract hunk headers and count changes
  const lines = diffOutput.split("\n");
  const hunks: Array<{ context: string; added: number; removed: number }> = [];
  let currentHunk: { context: string; added: number; removed: number } | null = null;

  for (const line of lines) {
    // Hunk header: @@ -a,b +c,d @@ optional context
    const hunkMatch = line.match(/^@@\s+[^@]+@@\s*(.*)$/);
    if (hunkMatch) {
      if (currentHunk) hunks.push(currentHunk);
      currentHunk = {
        context: hunkMatch[1]?.trim() ?? "",
        added: 0,
        removed: 0,
      };
      continue;
    }

    if (currentHunk) {
      if (line.startsWith("+") && !line.startsWith("+++")) {
        currentHunk.added++;
      } else if (line.startsWith("-") && !line.startsWith("---")) {
        currentHunk.removed++;
      }
    }
  }
  if (currentHunk) hunks.push(currentHunk);

  if (hunks.length === 0) {
    const lineCount = lines.length;
    return `diff: ${lineCount} lines`;
  }

  // Build summary from hunks — take up to 4 most significant
  const sorted = [...hunks].sort((a, b) =>
    (b.added + b.removed) - (a.added + a.removed),
  );
  const top = sorted.slice(0, 4);
  const totalAdded = hunks.reduce((s, h) => s + h.added, 0);
  const totalRemoved = hunks.reduce((s, h) => s + h.removed, 0);

  const parts: string[] = [];
  for (const h of top) {
    if (h.context) {
      parts.push(`${h.context} +${h.added}/-${h.removed}`);
    }
  }

  if (parts.length > 0) {
    const extra = hunks.length > 4 ? ` (+${hunks.length - 4} more)` : "";
    return `${parts.join("; ")}${extra} [total +${totalAdded}/-${totalRemoved}]`;
  }

  return `${hunks.length} hunks, +${totalAdded}/-${totalRemoved}`;
}
