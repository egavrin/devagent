/**
 * Shared constants, types, and helpers for TUI components.
 */

// ─── Spinner Constants ──────────────────────────────────────

export const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export const SPINNER_VERBS = [
  "Thinking", "Analyzing", "Considering", "Reasoning", "Processing",
  "Evaluating", "Examining", "Working", "Reflecting", "Assessing",
  "Deliberating", "Exploring", "Weighing", "Pondering", "Reviewing",
];

// ─── Log Entry Types ────────────────────────────────────────

export type LogEntryType = "tool" | "tool-group" | "reasoning" | "thinking-duration" | "plan" | "error" | "info" | "compaction" | "final-output" | "turn-summary";

export interface LogEntry {
  readonly id: string;
  readonly type: LogEntryType;
  readonly data: unknown;
}

// ─── Tool Helpers ───────────────────────────────────────────

export function extractToolPreview(toolName: string, output: string): string | undefined {
  if (!output || output.length < 10) return undefined;
  if (toolName === "search_files") {
    const match = output.match(/^(\d+) match/);
    if (match) return output.split("\n")[0]!.slice(0, 80);
  }
  if (toolName === "run_command") {
    const lines = output.split("\n").filter((l) => l.trim() && !l.startsWith("Exit code:"));
    if (lines.length > 0) return lines[0]!.trim().slice(0, 80);
  }
  if (toolName === "find_files") {
    const lines = output.split("\n").filter((l) => l.trim());
    if (lines.length > 0) return `${lines.length} file(s) found`;
  }
  return undefined;
}

export function extractEditDiff(toolName: string, output: string): string | undefined {
  if (toolName !== "replace_in_file" && toolName !== "write_file") return undefined;
  if (!output || output.length < 20) return undefined;
  return output.split("\n").filter((l) => l.trim()).slice(0, 8).join("\n");
}

export function summarizeParams(name: string, params: Record<string, unknown>): string {
  const path = params["path"] as string | undefined;
  if (path) return path;
  const command = params["command"] as string | undefined;
  if (command) return command.slice(0, 80);
  const pattern = params["pattern"] as string | undefined;
  if (pattern) return `"${pattern}"`;
  return "";
}

// ─── Text Helpers ───────────────────────────────────────────

export function cleanTime(text: string): string {
  return text.replace(/(\d+)\.\d+s/g, "$1s");
}

export function tokenProgressBar(used: number, max: number): string {
  const pct = Math.round((used / max) * 100);
  const width = 20;
  const filled = Math.round((pct / 100) * width);
  const bar = "▰".repeat(filled) + "▱".repeat(width - filled);
  const usedK = used >= 1000 ? `${Math.round(used / 1000)}k` : String(used);
  const maxK = max >= 1000 ? `${Math.round(max / 1000)}k` : String(max);
  return `${bar} ${usedK}/${maxK} (${pct}%)`;
}
