/**
 * CLI output formatting — colors, spinner, tool call display.
 * Zero external dependencies. Respects NO_COLOR env var.
 * All output goes to stderr (stdout reserved for LLM content).
 */

import type {
  VerbosityConfig,
  SubagentStartEvent,
  SubagentUpdateEvent,
  SubagentEndEvent,
  SubagentErrorEvent,
  DelegatedWorkSummary,
  ToolFileChangePreview,
} from "@devagent/runtime";
import { formatDuration } from "@devagent/runtime";
import type {
  TranscriptPart,
  PresentedToolEvent,
  PresentedToolGroup,
} from "./transcript-presenter.js";
import type { PresentedTurn } from "./transcript-composer.js";
import {
  buildHighlightedFileEdit,
  getPresentedDiffGutterWidth,
  takeVisibleHighlightedDiffItems,
} from "./file-edit-presentation.js";

// ─── Color Helpers ──────────────────────────────────────────

function wrap(code: string, s: string): string {
  return isColorEnabled() ? `\x1b[${code}m${s}\x1b[0m` : s;
}

export function dim(s: string): string { return wrap("90", s); }
export function red(s: string): string { return wrap("31", s); }
export function green(s: string): string { return wrap("32", s); }
export function yellow(s: string): string { return wrap("33", s); }
export function cyan(s: string): string { return wrap("36", s); }
export function bold(s: string): string { return wrap("1", s); }
function dimBold(s: string): string { return isColorEnabled() ? `\x1b[90;1m${s}\x1b[0m` : s; }

function isColorEnabled(): boolean {
  return !process.env["NO_COLOR"];
}

// ─── Categorical Verbosity ──────────────────────────────────

export function isCategoryEnabled(cat: string, config: VerbosityConfig): boolean {
  if (config.categories.size > 0) {
    return config.categories.has(cat);
  }
  return config.base === "verbose";
}

export function buildVerbosityConfig(
  base: "quiet" | "normal" | "verbose",
  verboseCategories: string | undefined,
): VerbosityConfig {
  const categories = new Set<string>();

  if (verboseCategories) {
    for (const c of verboseCategories.split(",")) {
      const trimmed = c.trim();
      if (trimmed.length > 0) categories.add(trimmed);
    }
  }

  const envDebug = process.env["DEVAGENT_DEBUG"];
  if (envDebug) {
    for (const c of envDebug.split(",")) {
      const trimmed = c.trim();
      if (trimmed.length > 0) categories.add(trimmed);
    }
  }

  return { base, categories };
}

// ─── Diff Preview ──────────────────────────────────────────

export function formatFileEditPreview(
  fileEdits: ReadonlyArray<ToolFileChangePreview> | undefined,
  hiddenCount: number = 0,
  maxDiffLines: number = 8,
): string | null {
  if (!fileEdits || fileEdits.length === 0) return null;

  const blocks = fileEdits.map((fileEdit) => {
    const highlighted = buildHighlightedFileEdit(fileEdit);
    const visible = takeVisibleHighlightedDiffItems(highlighted.hunks, maxDiffLines);
    const gutterWidth = getPresentedDiffGutterWidth(highlighted.hunks);
    const overflow = visible.hiddenLines;
    const heading = `    ${dim(fileEdit.path)} ${green(`+${fileEdit.additions}`)} ${red(`-${fileEdit.deletions}`)}${fileEdit.truncated ? ` ${dim("(truncated)")}` : ""}`;
    const summary = `    ${dim(highlighted.summary)}`;
    const body = visible.items
      .map((item) => {
        if (item.type === "separator") {
          return `    ${dim("...")}`;
        }
        return `    ${formatStructuredDiffLine(item.line, gutterWidth)}`;
      })
      .join("\n");
    const footer = overflow > 0 ? `\n${dim(`    ... +${overflow} more diff lines`)}` : "";
    return `${heading}\n${summary}${body ? `\n${body}` : ""}${footer}`;
  });

  if (hiddenCount > 0) {
    blocks.push(dim(`    ... +${hiddenCount} more files`));
  }

  return blocks.join("\n");
}

export function formatTranscriptPart(
  part: TranscriptPart,
  options?: { readonly maxDiffLines?: number },
): string | null {
  switch (part.kind) {
    case "tool":
      return formatPresentedToolEvent(part.event);
    case "tool-group":
      return formatPresentedToolGroup(part.event);
    case "file-edit":
      return formatFileEditPreview([part.data.fileEdit], 0, options?.maxDiffLines);
    case "file-edit-overflow":
      return dim(`    ... +${part.data.hiddenCount} more files`);
    case "command-result":
      return formatPresentedCommandResult(part.data);
    case "validation-result":
      return formatPresentedValidationResult(part.data);
    case "diagnostic-list":
      return formatPresentedDiagnosticList(part.data);
    case "status":
      return formatPresentedStatus(part.data.title, part.data.lines);
    case "progress":
      return dim(`[${part.data.title.toLowerCase()}] ${part.data.detail ?? "Working…"}`);
    case "approval":
      return yellow(`[approval] ${part.data.toolName}: ${part.data.details}`);
    case "error":
      return formatError(part.data.message);
    case "user":
      return `> ${part.data.text}`;
    case "info":
      return formatPresentedStatus(part.data.title, part.data.lines);
    case "reasoning":
      return dim(`ℹ ${part.data.text}`);
    case "final-output":
    case "turn-summary":
    case "plan":
      return null;
    default:
      return null;
  }
}

function formatPresentedToolEvent(event: PresentedToolEvent): string {
  const detail = event.summary ? ` ${dim(event.summary)}` : "";
  if (event.status === "running") {
    return `${dim("  ")}${cyan("● ")}${bold(event.name)}${detail}`;
  }
  const icon = event.status === "success" ? green("✓ ") : red("✗ ");
  const duration = event.durationMs !== undefined ? dim(` (${event.durationMs}ms)`) : "";
  const error = event.error ? dim(`: ${event.error.split("\n")[0]}`) : "";
  const preview = event.preview ? `\n${dim("    → ")}${cyan(event.preview)}` : "";
  return `${dim("  ")}${icon}${dim(event.name)}${duration}${error}${preview}`;
}

function formatPresentedToolGroup(event: PresentedToolGroup): string {
  const files = event.summaries.slice(0, 3).map((summary) => summary.split("/").pop() ?? summary);
  if (event.status === "running") {
    return `${dim("  ")}${cyan("● ")}${bold(event.name)}${dim(` ${event.count} calls (${event.summaries.slice(0, 3).join(", ")}${event.count > 3 ? ` +${event.count - 3} more` : ""})`)}`;
  }
  return `${dim("  ")}${event.status === "success" ? green("✓ ") : red("✗ ")}${dim(event.name)}${dim(` ×${event.count}`)}${files.length > 0 ? ` ${cyan(files.join(", "))}` : ""}${event.totalDurationMs !== undefined ? dim(` (${event.totalDurationMs}ms)`) : ""}`;
}

function formatPresentedStatus(title: string, lines: ReadonlyArray<string>): string {
  if (lines.length === 0) return dim(`[${title}]`);
  if (lines.length === 1) return dim(`[${title}] ${lines[0]}`);
  return [dim(`[${title}]`), ...lines.map((line) => dim(`  ${line}`))].join("\n");
}

function formatPresentedCommandResult(
  data: Extract<TranscriptPart, { readonly kind: "command-result" }>["data"],
): string {
  const tone = data.status === "success" ? green : data.status === "warning" ? yellow : red;
  const lines = [
    `${dim("[command]")} ${bold(data.command)}`,
    `${dim("  cwd")} ${data.cwd}`,
    `${dim("  status")} ${tone(data.statusLine)}`,
  ];
  if (data.stdoutPreview) {
    if (data.stdoutPreview.includes("\n")) {
      lines.push(`${dim("  stdout")}`);
      lines.push(...data.stdoutPreview.split("\n").map((line) => `${dim("    ")}${line}`));
      if (data.stdoutTruncated) {
        lines.push(dim("    …"));
      }
    } else {
      lines.push(`${dim("  stdout")} ${data.stdoutPreview}${data.stdoutTruncated ? dim(" …") : ""}`);
    }
  }
  if (data.stderrPreview) {
    if (data.stderrPreview.includes("\n")) {
      lines.push(`${dim("  stderr")}`);
      lines.push(...data.stderrPreview.split("\n").map((line) => `${dim("    ")}${tone(line)}`));
      if (data.stderrTruncated) {
        lines.push(dim("    …"));
      }
    } else {
      lines.push(`${dim("  stderr")} ${tone(data.stderrPreview)}${data.stderrTruncated ? dim(" …") : ""}`);
    }
  }
  return lines.join("\n");
}

function formatPresentedValidationResult(
  data: Extract<TranscriptPart, { readonly kind: "validation-result" }>["data"],
): string {
  const head = `${dim("[validation]")} ${data.passed ? green(data.summary) : yellow(data.summary)}`;
  const rest: string[] = [];
  if (data.testSummaryLine) {
    rest.push(`${dim("  tests")} ${data.testSummaryLine}`);
  }
  if (data.testOutputPreview) {
    if (data.testOutputPreview.includes("\n")) {
      rest.push(`${dim("  output")}`);
      rest.push(...data.testOutputPreview.split("\n").map((line) => `${dim("    ")}${line}`));
    } else {
      rest.push(`${dim("  output")} ${data.testOutputPreview}`);
    }
  }
  return [head, ...rest].join("\n");
}

function formatPresentedDiagnosticList(
  data: Extract<TranscriptPart, { readonly kind: "diagnostic-list" }>["data"],
): string {
  const lines = [`${yellow(`[${data.title}]`)}`];
  for (const diagnostic of data.diagnostics) {
    lines.push(`${dim("  •")} ${diagnostic}`);
  }
  if (data.hiddenCount > 0) {
    lines.push(dim(`  ... +${data.hiddenCount} more diagnostics`));
  }
  return lines.join("\n");
}

function formatStructuredDiffLine(
  line: {
    readonly type: "context" | "add" | "delete";
    readonly oldLine: number | null;
    readonly newLine: number | null;
    readonly text: string;
    readonly renderedText?: string;
    readonly syntaxHighlighted?: boolean;
  },
  gutterWidth: number,
): string {
  const oldLine = formatLineNumber(line.oldLine, gutterWidth);
  const newLine = formatLineNumber(line.newLine, gutterWidth);
  const marker = line.type === "add" ? green("+") : line.type === "delete" ? red("-") : dim(" ");
  const renderedText = line.renderedText ?? line.text;
  const text = line.text.length > 0 ? renderedText : dim("<blank>");
  const code = line.syntaxHighlighted
    ? text
    : line.type === "add"
      ? green(text)
      : line.type === "delete"
        ? red(text)
        : text;
  return `${dim(oldLine)} ${dim(newLine)} ${marker} ${code}`;
}

function formatLineNumber(line: number | null, width: number): string {
  return line === null ? "".padStart(width, " ") : String(line).padStart(width, " ");
}

// ─── Terminal Title & Bell ──────────────────────────────────

const isTTY = process.stderr.isTTY ?? false;

/** Set the terminal title (tab name). No-op if not a TTY. */
export function setTerminalTitle(title: string): void {
  if (!isTTY) return;
  process.stderr.write(`\x1b]0;${title}\x07`);
}

/** Ring the terminal bell. No-op if not a TTY. */
export function terminalBell(): void {
  if (!isTTY) return;
  process.stderr.write("\x07");
}

// ─── Spinner ────────────────────────────────────────────────

// Shared with tui/shared.ts — keep in sync or import when ESM allows
const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

const SPINNER_VERBS = [
  "Thinking", "Analyzing", "Considering", "Reasoning", "Processing",
  "Evaluating", "Examining", "Working", "Reflecting", "Assessing",
  "Deliberating", "Exploring", "Weighing", "Pondering", "Reviewing",
];

export class Spinner {
  private frameIndex = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private message = "";
  private startedAt = 0;
  /** Optional metrics suffix appended after the message + elapsed time. */
  private suffix = "";

  start(message?: string): void {
    this.stop();
    this.message = message ?? SPINNER_VERBS[Math.floor(Math.random() * SPINNER_VERBS.length)]! + "…";
    this.frameIndex = 0;
    this.startedAt = Date.now();
    this.render();
    this.timer = setInterval(() => this.render(), 80);
  }

  update(message: string): void {
    this.message = message;
  }

  /** Update the metrics suffix shown after the spinner message. */
  updateSuffix(suffix: string): void {
    this.suffix = suffix;
  }

  stop(finalMessage?: string): void {
    if (!this.timer) return; // Not running — nothing to clear
    clearInterval(this.timer);
    this.timer = null;
    // Clear the spinner line
    process.stderr.write("\x1b[2K\r");
    if (finalMessage) {
      process.stderr.write(finalMessage + "\n");
    }
  }

  /** Write a line to stderr without colliding with an active spinner. */
  log(line: string): void {
    if (this.timer) {
      // Clear spinner, write the line, then re-render
      process.stderr.write(`\x1b[2K\r${line}\n`);
      this.render();
    } else {
      process.stderr.write(line + "\n");
    }
  }

  get active(): boolean {
    return this.timer !== null;
  }

  private render(): void {
    const frame = SPINNER_FRAMES[this.frameIndex % SPINNER_FRAMES.length]!;
    this.frameIndex++;
    const elapsed = Date.now() - this.startedAt;
    const elapsedStr = elapsed >= 1000 ? ` ${dim(`(${(elapsed / 1000).toFixed(1)}s)`)}` : "";
    const suffixStr = this.suffix ? ` ${dim(this.suffix)}` : "";
    process.stderr.write(`\x1b[2K\r${cyan(frame)} ${dim(this.message)}${elapsedStr}${suffixStr}`);
  }
}

// ─── Tool Call Formatting ───────────────────────────────────

/**
 * Extract a human-readable summary of tool parameters.
 * Shows the most relevant info per tool type.
 */
export function summarizeToolParams(name: string, params: Record<string, unknown>): string {
  switch (name) {
    case "read_file": {
      const path = params["path"] as string ?? "";
      const start = params["start_line"] as number | undefined;
      const end = params["end_line"] as number | undefined;
      if (start !== undefined && end !== undefined) {
        return `${path}:${start}-${end}`;
      }
      if (start !== undefined) {
        return `${path}:${start}+`;
      }
      return path;
    }

    case "write_file": {
      const path = params["path"] as string ?? "";
      const content = params["content"] as string ?? "";
      return `${path} (${content.length} bytes)`;
    }

    case "replace_in_file": {
      const rifPath = (params["path"] as string) ?? "";
      const search = params["search"] as string | undefined;
      if (search) {
        const firstLine = search.split("\n")[0] ?? "";
        return `${rifPath} "${truncate(firstLine.trim(), 30)}"`;
      }
      return rifPath;
    }

    case "search_files": {
      const pattern = params["pattern"] as string ?? "";
      const scope = params["path"] as string | undefined;
      const filePattern = params["file_pattern"] as string | undefined;
      let s = `"${truncate(pattern, 30)}"`;
      if (filePattern) s += ` in ${filePattern}`;
      else if (scope && scope !== ".") s += ` in ${scope}`;
      return s;
    }

    case "find_files":
      return (params["pattern"] as string) ?? "";

    case "run_command":
      return truncate((params["command"] as string) ?? "", 60);

    case "git_status":
      return "";

    case "git_diff": {
      const parts: string[] = [];
      if (params["staged"]) parts.push("--staged");
      if (params["ref"]) parts.push(params["ref"] as string);
      if (params["path"]) parts.push(params["path"] as string);
      return parts.join(" ");
    }

    case "git_commit":
      return truncate((params["message"] as string) ?? "", 50);

    case "update_plan":
      return "";

    default:
      // For unknown tools, show the first string parameter.
      for (const value of Object.values(params)) {
        if (typeof value === "string" && value.length > 0) {
          return truncate(value, 40);
        }
      }
      return "";
  }
}

interface ToolVerbPair {
  readonly present: string;
  readonly past: string;
}

const TOOL_VERBS: Record<string, ToolVerbPair> = {
  read_file:       { present: "Reading",          past: "Read" },
  write_file:      { present: "Writing",          past: "Wrote" },
  replace_in_file: { present: "Editing",          past: "Edited" },
  search_files:    { present: "Searching",        past: "Searched" },
  find_files:      { present: "Finding",          past: "Found" },
  run_command:     { present: "Running",          past: "Ran" },
  git_status:      { present: "Checking status",  past: "Checked status" },
  git_diff:        { present: "Diffing",          past: "Diffed" },
  git_commit:      { present: "Committing",       past: "Committed" },
  update_plan:     { present: "Updating plan",    past: "Updated plan" },
};

function toolVerb(name: string): ToolVerbPair {
  return TOOL_VERBS[name] ?? { present: `Calling ${name}`, past: `Called ${name}` };
}

function buildCounter(counterLabel: string | undefined, iteration: number, maxIter: number): string {
  if (counterLabel) return dim(`[${counterLabel}:${iteration}]`);
  return maxIter > 0 ? dim(`[${iteration}/${maxIter}]`) : dim(`[${iteration}]`);
}

export function formatToolStart(
  name: string,
  params: Record<string, unknown>,
  iteration: number,
  maxIter: number,
  gauge?: string,
  counterLabel?: string,
): string {
  const counter = buildCounter(counterLabel, iteration, maxIter);
  const summary = summarizeToolParams(name, params);
  const verb = toolVerb(name);
  const detail = summary ? ` ${dim(summary)}` : "";
  const gaugeSuffix = gauge ? ` ${gauge}` : "";
  return `${counter} ${dimBold("↳")} ${bold(verb.present)}${detail}${gaugeSuffix}`;
}

export function formatToolEnd(
  name: string,
  success: boolean,
  durationMs: number,
  error?: string,
  params?: Record<string, unknown>,
): string {
  const duration = dim(`(${formatDuration(durationMs)})`);
  const verb = toolVerb(name);
  const summary = params ? summarizeToolParams(name, params) : "";
  const detail = summary ? ` ${dim(summary)}` : "";
  if (success) {
    return `  ${green("✓")} ${dim(verb.past)}${detail} ${duration}`;
  }
  const errMsg = error ? `: ${truncate(error, 80)}` : "";
  return `  ${red("✗")} ${dim(verb.past)}${detail} ${duration}${red(errMsg)}`;
}

// ─── Tool Grouping ──────────────────────────────────────────

const FILE_TOOLS = new Set(["read_file", "write_file", "replace_in_file", "find_files"]);

function groupNoun(name: string, count: number): string {
  if (FILE_TOOLS.has(name)) return `${count} file${count > 1 ? "s" : ""}`;
  return `${count} call${count > 1 ? "s" : ""}`;
}

export function formatToolGroupStart(
  name: string,
  count: number,
  paramSummaries: ReadonlyArray<string>,
  iteration: number,
  maxIter: number,
  gauge?: string,
  counterLabel?: string,
): string {
  const counter = buildCounter(counterLabel, iteration, maxIter);
  const verb = toolVerb(name);
  const noun = groupNoun(name, count);
  const displayed = paramSummaries.slice(0, 3);
  const extra = paramSummaries.length > 3 ? `, +${paramSummaries.length - 3} more` : "";
  const detail = displayed.length > 0
    ? ` ${dim(`(${displayed.join(", ")}${extra})`)}`
    : "";
  const gaugeSuffix = gauge ? ` ${gauge}` : "";
  return `${counter} ${dimBold("↳")} ${bold(`${verb.present} ${noun}`)}${detail}${gaugeSuffix}`;
}

export function formatToolGroupEnd(
  name: string,
  count: number,
  success: boolean,
  totalDurationMs: number,
  error?: string,
): string {
  const duration = dim(`(${formatDuration(totalDurationMs)} total)`);
  const verb = toolVerb(name);
  const noun = groupNoun(name, count);
  if (success) {
    return `  ${green("✓")} ${dim(`${verb.past} ${noun}`)} ${duration}`;
  }
  const errMsg = error ? `: ${truncate(error, 80)}` : "";
  return `  ${red("✗")} ${dim(`${verb.past} ${noun}`)} ${duration}${red(errMsg)}`;
}

// ─── Plan Rendering ─────────────────────────────────────────

export function formatPlan(
  steps: ReadonlyArray<{ description: string; status: string; lastTransitionIteration?: number }>,
  showIteration = false,
): string {
  const lines = steps.map((s) => {
    const iterSuffix = showIteration && s.lastTransitionIteration
      ? dim(` @iter ${s.lastTransitionIteration}`)
      : "";
    switch (s.status) {
      case "completed":
        return `  ${green("[x]")} ${dim(s.description)}${iterSuffix}`;
      case "in_progress":
        return `  ${yellow("[>]")} ${s.description}${iterSuffix}`;
      default:
        return `  ${dim("[ ]")} ${dim(s.description)}${iterSuffix}`;
    }
  });
  return lines.join("\n");
}

// ─── Per-Turn Summary ───────────────────────────────────────

interface TurnStats {
  readonly toolCallCount: number;
  readonly iterationCount: number;
  readonly inputTokens?: number;
  readonly costDelta: number;
  readonly elapsedMs: number;
  readonly filesChanged?: number;
  readonly validationFailed?: boolean;
  readonly status?: "running" | "completed" | "error" | "budget_exceeded";
}

export function formatTurnSummary(stats: TurnStats): string {
  const parts: string[] = [];
  parts.push(`${stats.iterationCount} ${stats.iterationCount === 1 ? "iteration" : "iterations"}`);
  if (stats.toolCallCount > 0) {
    parts.push(`${stats.toolCallCount} ${stats.toolCallCount === 1 ? "tool call" : "tool calls"}`);
  }
  if ((stats.filesChanged ?? 0) > 0) {
    const filesChanged = stats.filesChanged ?? 0;
    parts.push(`${filesChanged} ${filesChanged === 1 ? "file" : "files"} changed`);
  }
  if (stats.validationFailed) {
    parts.push("validation failed");
  }
  if ((stats.inputTokens ?? 0) > 0) {
    const kIn = Math.round((stats.inputTokens ?? 0) / 1000);
    parts.push(`${kIn}k input tokens`);
  }
  if (stats.costDelta > 0) {
    parts.push(`$${stats.costDelta.toFixed(4)}`);
  }
  parts.push(formatDuration(stats.elapsedMs));
  const status = stats.status ?? "completed";
  const icon = status === "error" ? red("✗") : status === "budget_exceeded" ? yellow("!") : green("✓");
  const label = status === "error" ? "Finished with errors" : status === "budget_exceeded" ? "Budget exhausted" : "Done";
  return `${icon} ${bold(label)} ${dim(`(${parts.join(", ")})`)}`;
}

export function formatTurnStart(userText: string): string {
  return `${dim("╭─")} ${cyan("turn")} ${bold(userText)}`;
}

export function formatTurnEnd(turn: PresentedTurn): string {
  return `${dim("╰─")} ${formatTurnSummary({
    iterationCount: turn.metrics.iterations,
    toolCallCount: turn.metrics.toolCalls,
    filesChanged: turn.metrics.filesChanged,
    validationFailed: turn.metrics.validationFailed,
    costDelta: turn.metrics.cost,
    elapsedMs: turn.metrics.elapsedMs,
    status: turn.status,
  })}`;
}

export function formatError(message: string): string {
  return `${red("✗")} ${red(message)}`;
}

// ─── Session Summary ────────────────────────────────────────

interface SessionSummaryData {
  readonly sessionId: string;
  readonly totalIterations: number;
  readonly totalToolCalls: number;
  readonly toolUsage: ReadonlyMap<string, number>;
  readonly filesChanged: ReadonlyArray<string>;
  readonly planSteps?: ReadonlyArray<{ description: string; status: string }>;
  readonly totalCost: number;
  readonly totalInputTokens: number;
  readonly totalOutputTokens: number;
  readonly elapsedMs: number;
  readonly completionReason: string;
  readonly delegatedWork?: DelegatedWorkSummary;
}

export function formatSessionSummary(data: SessionSummaryData): string {
  const lines: string[] = [];
  lines.push("");
  lines.push(bold("Session Summary"));
  lines.push(dim("─".repeat(50)));

  lines.push(`  Session:      ${dim(data.sessionId)}`);
  lines.push(`  Duration:     ${formatDuration(data.elapsedMs)}`);
  lines.push(`  Iterations:   ${data.totalIterations}`);
  lines.push(`  Tool calls:   ${data.totalToolCalls}`);

  // Tool usage sorted by count descending
  if (data.toolUsage.size > 0) {
    const sorted = [...data.toolUsage.entries()].sort((a, b) => b[1] - a[1]);
    lines.push("  Tool usage:");
    for (const [name, count] of sorted) {
      lines.push(`    ${name}: ${count}`);
    }
  }

  // Files changed (max 15)
  if (data.filesChanged.length > 0) {
    const displayFiles = data.filesChanged.slice(0, 15);
    lines.push(`  Files changed (${data.filesChanged.length}):`);
    for (const f of displayFiles) {
      lines.push(`    ${f}`);
    }
    if (data.filesChanged.length > 15) {
      lines.push(dim(`    ... (+${data.filesChanged.length - 15} more)`));
    }
  }

  // Plan progress
  if (data.planSteps && data.planSteps.length > 0) {
    const completed = data.planSteps.filter((s) => s.status === "completed").length;
    lines.push(`  Plan:         ${completed}/${data.planSteps.length} completed`);
  }

  // Cost
  if (data.totalCost > 0) {
    lines.push(`  Cost:         $${data.totalCost.toFixed(4)}`);
  }

  // Tokens
  if (data.totalInputTokens > 0 || data.totalOutputTokens > 0) {
    const kIn = Math.round(data.totalInputTokens / 1000);
    const kOut = Math.round(data.totalOutputTokens / 1000);
    lines.push(`  Tokens:       ${kIn}k in / ${kOut}k out`);
  }

  if (data.delegatedWork && data.delegatedWork.childCount > 0) {
    lines.push(`  Subagents:    ${data.delegatedWork.childCount}`);
    const byTypeEntries = Object.entries(data.delegatedWork.byType);
    if (byTypeEntries.length > 0) {
      lines.push(`  By type:      ${byTypeEntries.map(([type, count]) => `${type}=${count}`).join(", ")}`);
    }
    if (data.delegatedWork.lanes.length > 0) {
      lines.push(`  Lanes:        ${data.delegatedWork.lanes.join(", ")}`);
    }
    lines.push(`  Delegated:    ${formatDuration(data.delegatedWork.totalDelegatedDurationMs)} total`);
    lines.push(`  Parallel:     ${data.delegatedWork.parallelBatchCount} batch(es), max ${data.delegatedWork.maxParallelChildren} child(ren)`);
  }

  lines.push(dim("─".repeat(50)));
  return lines.join("\n");
}

export function formatSubagentBatchLaunch(
  agentType: string,
  batchSize: number,
): string {
  return `${dim("  ↳")} ${bold(`Launching ${batchSize} ${agentType} subagents in parallel`)}`;
}

export function formatSubagentStart(event: SubagentStartEvent): string {
  const lane = event.laneLabel ? ` ${dim(event.laneLabel)}` : "";
  const modelBits = [event.model, event.reasoningEffort].filter(Boolean).join(", ");
  const model = modelBits ? ` ${dim(`(${modelBits})`)}` : "";
  return `${dim("  ↳")} ${bold(`Subagent ${event.agentId}`)} ${dim(event.agentType)}${lane}${model}`;
}

export function formatSubagentError(event: SubagentErrorEvent): string {
  return `  ${red("✗")} ${dim(`Subagent ${event.agentId} failed`)} ${dim(`(${formatDuration(event.durationMs)})`)}${red(`: ${truncate(event.error, 80)}`)}`;
}

export function summarizeSubagentUpdate(event: SubagentUpdateEvent): string {
  if (event.summary && event.summary.trim().length > 0) {
    return event.summary;
  }
  if (event.milestone === "iteration:start") {
    return event.iteration ? `Starting iteration ${event.iteration}` : "Starting iteration";
  }
  if (event.milestone === "tool:before") {
    return event.toolName ? `Running ${event.toolName}` : "Running tool";
  }
  if (event.toolName) {
    return event.toolSuccess === false ? `Failed ${event.toolName}` : `Completed ${event.toolName}`;
  }
  return "Updated progress";
}

// ─── Error Enrichment ───────────────────────────────────────

interface RecentToolResult {
  readonly name: string;
  readonly success: boolean;
  readonly durationMs: number;
}

interface EnrichedErrorContext {
  readonly message: string;
  readonly recentTools: ReadonlyArray<RecentToolResult>;
  readonly suggestion: string | null;
}

export function inferErrorSuggestion(
  message: string,
  recentTools: ReadonlyArray<RecentToolResult>,
): string | null {
  const lower = message.toLowerCase();

  if (lower.includes("rate limit") || lower.includes("429") || lower.includes("rate_limit")) {
    return "Rate limited — wait a moment or switch to a different provider.";
  }
  if (lower.includes("context length") || lower.includes("token limit") || lower.includes("maximum context")) {
    return "Context limit reached — use /clear or reduce output length.";
  }
  if (lower.includes("timeout") || lower.includes("etimedout")) {
    return "Request timed out — check network connectivity.";
  }

  // Check for repeated same-tool failures
  const failedTools = recentTools.filter((t) => !t.success);
  if (failedTools.length >= 2) {
    const lastName = failedTools[failedTools.length - 1]?.name;
    const repeatedFailures = failedTools.filter((t) => t.name === lastName);
    if (repeatedFailures.length >= 2) {
      return `Tool "${lastName}" has failed repeatedly. Try a different approach.`;
    }
  }

  return null;
}

export function formatEnrichedError(ctx: EnrichedErrorContext): string {
  const lines: string[] = [];
  lines.push(`${red("✗")} ${red(ctx.message)}`);

  if (ctx.recentTools.length > 0) {
    lines.push(dim("  Recent tool chain:"));
    for (const t of ctx.recentTools) {
      const icon = t.success ? green("✓") : red("✗");
      const dur = formatDuration(t.durationMs);
      lines.push(`    ${icon} ${t.name} (${dur})`);
    }
  }

  if (ctx.suggestion) {
    lines.push(`  ${yellow("Suggestion:")} ${ctx.suggestion}`);
  }

  return lines.join("\n");
}

// ─── Compaction Result ──────────────────────────────────────

interface CompactionResultData {
  readonly tokensBefore: number;
  readonly estimatedTokens: number;
  readonly removedCount: number;
  readonly prunedCount?: number;
}

export function formatCompactionResult(data: CompactionResultData): string {
  const kAfter = Math.round(data.estimatedTokens / 1000);
  const parts: string[] = [];

  // Before → after with percentage
  if (data.tokensBefore > 0) {
    const kBefore = Math.round(data.tokensBefore / 1000);
    const reduction = data.tokensBefore - data.estimatedTokens;
    if (reduction > 0) {
      const pct = Math.round((reduction / data.tokensBefore) * 100);
      parts.push(`${kBefore}k → ${kAfter}k tokens (${pct}% reduction)`);
    } else {
      parts.push(`${kBefore}k → ${kAfter}k tokens (no reduction)`);
    }
  } else {
    parts.push(`~${kAfter}k tokens remaining`);
  }

  // Activity details
  const details: string[] = [];
  if (data.removedCount > 0) {
    details.push(`removed ${data.removedCount} msgs`);
  }
  if ((data.prunedCount ?? 0) > 0) {
    details.push(`pruned ${data.prunedCount} outputs`);
  }
  if (details.length > 0) {
    parts.push(details.join(", "));
  }

  return dim(`[context] Compacted: ${parts.join(", ")}`);
}

// ─── Context Gauge ──────────────────────────────────────────

export function formatContextGauge(estimatedTokens: number, maxTokens: number): string {
  if (maxTokens <= 0 || estimatedTokens <= 0) return "";
  const kEstimated = Math.round(estimatedTokens / 1000);
  const kMax = Math.round(maxTokens / 1000);
  const ratio = estimatedTokens / maxTokens;
  const label = `[tokens: ${kEstimated}k/${kMax}k]`;
  if (ratio > 0.8) return red(label);
  if (ratio > 0.6) return yellow(label);
  return dim(label);
}

// ─── Reasoning Display ──────────────────────────────────────

export function formatReasoning(text: string): string | null {
  const collapsed = text.replace(/\s+/g, " ").trim();
  if (collapsed.length === 0) return null;

  // Take first sentence if it ends within 120 chars
  const sentenceEnd = collapsed.search(/[.!?]\s/);
  let display: string;
  if (sentenceEnd > 0 && sentenceEnd < 120) {
    display = collapsed.substring(0, sentenceEnd + 1);
  } else {
    display = truncate(collapsed, 120);
  }

  return `  ${dim("ℹ")} ${dim(display)}`;
}

interface SubagentPanelData {
  readonly agentId: string;
  readonly agentType: string;
  readonly laneLabel?: string | null;
  readonly model: string;
  readonly reasoningEffort?: string;
  readonly status: "running" | "completed" | "error";
  readonly currentIteration: number;
  readonly startedAtMs: number;
  readonly durationMs?: number;
  readonly currentActivity: string;
  readonly recentActivity: ReadonlyArray<string>;
  readonly quality?: {
    readonly score: number;
    readonly completeness: string;
  };
}

export class SubagentPanelRenderer {
  private static readonly REDRAW_DEBOUNCE_MS = 100;
  private readonly enabled: boolean;
  private panels: ReadonlyArray<SubagentPanelData> = [];
  private renderedLineCount = 0;
  private hidden = false;
  private redrawTimer: ReturnType<typeof setTimeout> | null = null;

  constructor(enabled: boolean) {
    this.enabled = enabled && !!process.stderr.isTTY;
  }

  get active(): boolean {
    return this.enabled;
  }

  setPanels(panels: ReadonlyArray<SubagentPanelData>): void {
    this.panels = panels;
    if (!this.enabled || this.hidden) return;
    if (panels.length === 0) {
      this.clearPendingRedraw();
      this.redraw();
      return;
    }
    this.scheduleRedraw();
  }

  suspend(): void {
    if (!this.enabled || this.hidden) return;
    this.clearPendingRedraw();
    this.clearRendered();
    this.hidden = true;
  }

  resume(): void {
    if (!this.enabled) return;
    this.hidden = false;
  }

  cleanup(): void {
    if (!this.enabled) return;
    this.clearPendingRedraw();
    this.clearRendered();
    this.hidden = false;
    this.panels = [];
  }

  formatPanels(panels: ReadonlyArray<SubagentPanelData>, now = Date.now()): string[] {
    const lines: string[] = [];
    for (const panel of panels) {
      const lane = panel.laneLabel ? ` ${dim(panel.laneLabel)}` : "";
      const modelBits = [panel.model, panel.reasoningEffort].filter(Boolean).join(", ");
      const model = modelBits ? ` ${dim(`(${modelBits})`)}` : "";
      lines.push(`${dim("  ↳")} ${bold(`Subagent ${panel.agentId}`)} ${dim(panel.agentType)}${lane}${model}`);

      const elapsedMs = panel.status === "running"
        ? Math.max(0, now - panel.startedAtMs)
        : (panel.durationMs ?? Math.max(0, now - panel.startedAtMs));
      const statusParts: string[] = [];
      if (panel.status === "running") {
        statusParts.push(yellow("running"));
      } else if (panel.status === "completed") {
        statusParts.push(green("completed"));
      } else {
        statusParts.push(red("failed"));
      }
      if (panel.currentIteration > 0) {
        statusParts.push(dim(`iter ${panel.currentIteration}`));
      }
      statusParts.push(dim(formatDuration(elapsedMs)));
      if (panel.quality) {
        statusParts.push(dim(`score ${panel.quality.score.toFixed(2)}`));
        statusParts.push(dim(panel.quality.completeness));
      }
      lines.push(`    ${statusParts.join("  ")}`);

      const recent = panel.recentActivity.length > 0
        ? `Recent: ${panel.recentActivity.slice(0, 2).join(" • ")}`
        : panel.currentActivity;
      lines.push(`    ${truncate(recent, 120)}`);
    }
    return lines;
  }

  private redraw(): void {
    this.clearPendingRedraw();
    if (this.hidden) return;
    this.clearRendered();
    const lines = this.formatPanels(this.panels);
    if (lines.length === 0) return;
    process.stderr.write(lines.join("\n") + "\n");
    this.renderedLineCount = lines.length;
  }

  private scheduleRedraw(): void {
    if (this.redrawTimer) return;
    this.redrawTimer = setTimeout(() => {
      this.redrawTimer = null;
      this.redraw();
    }, SubagentPanelRenderer.REDRAW_DEBOUNCE_MS);
  }

  private clearPendingRedraw(): void {
    if (!this.redrawTimer) return;
    clearTimeout(this.redrawTimer);
    this.redrawTimer = null;
  }

  private clearRendered(): void {
    if (!this.enabled || this.renderedLineCount === 0) return;
    process.stderr.write(`\x1b[${this.renderedLineCount}F`);
    for (let i = 0; i < this.renderedLineCount; i++) {
      process.stderr.write("\x1b[2K");
      if (i < this.renderedLineCount - 1) {
        process.stderr.write("\x1b[1E");
      }
    }
    if (this.renderedLineCount > 1) {
      process.stderr.write(`\x1b[${this.renderedLineCount - 1}F`);
    }
    this.renderedLineCount = 0;
  }
}

// ─── Helpers ────────────────────────────────────────────────

export function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.substring(0, maxLen - 1) + "…";
}
