/**
 * CLI output formatting — colors, spinner, tool call display.
 * Zero external dependencies. Respects NO_COLOR env var.
 * All output goes to stderr (stdout reserved for LLM content).
 */

import { formatDuration } from "@devagent/runtime";

import {
  buildHighlightedFileEdit,
  getPresentedDiffGutterWidth,
  takeVisibleHighlightedDiffItems,
} from "./file-edit-presentation.js";
import { bold, cyan, dim, dimBold, green, red, truncate, yellow } from "./format-colors.js";
import type { PresentedTurn } from "./transcript-composer.js";
import type {
  TranscriptPart,
  PresentedToolEvent,
  PresentedToolGroup,
} from "./transcript-presenter.js";
import type {
  VerbosityConfig,
  ToolFileChangePreview,
} from "@devagent/runtime";
export { bold, cyan, dim, green, red, yellow } from "./format-colors.js";
export { SubagentPanelRenderer } from "./subagent-panel-renderer.js";

// ─── Color Helpers ──────────────────────────────────────────

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
  return TRANSCRIPT_PART_FORMATTERS[part.kind]?.(part, options) ?? null;
}

type TranscriptPartFormatter = (
  part: TranscriptPart,
  options?: { readonly maxDiffLines?: number },
) => string | null;

const TRANSCRIPT_PART_FORMATTERS: Partial<Record<TranscriptPart["kind"], TranscriptPartFormatter>> = {
  tool: (part) => part.kind === "tool" ? formatPresentedToolEvent(part.event) : null,
  "tool-group": (part) => part.kind === "tool-group" ? formatPresentedToolGroup(part.event) : null,
  "file-edit": formatFileEditTranscriptPart,
  "file-edit-overflow": (part) =>
    part.kind === "file-edit-overflow" ? dim(`    ... +${part.data.hiddenCount} more files`) : null,
  "command-result": (part) =>
    part.kind === "command-result" ? formatPresentedCommandResult(part.data) : null,
  "validation-result": (part) =>
    part.kind === "validation-result" ? formatPresentedValidationResult(part.data) : null,
  "diagnostic-list": (part) =>
    part.kind === "diagnostic-list" ? formatPresentedDiagnosticList(part.data) : null,
  status: (part) => part.kind === "status" ? formatPresentedStatus(part.data.title, part.data.lines) : null,
  progress: formatProgressTranscriptPart,
  approval: (part) =>
    part.kind === "approval" ? yellow(`[approval] ${part.data.toolName}: ${part.data.details}`) : null,
  error: (part) => part.kind === "error" ? formatError(part.data.message) : null,
  user: (part) => part.kind === "user" ? `> ${part.data.text}` : null,
  info: (part) => part.kind === "info" ? formatPresentedStatus(part.data.title, part.data.lines) : null,
  reasoning: (part) => part.kind === "reasoning" ? dim(`ℹ ${part.data.text}`) : null,
};

function formatFileEditTranscriptPart(
  part: TranscriptPart,
  options?: { readonly maxDiffLines?: number },
): string | null {
  return part.kind === "file-edit"
    ? formatFileEditPreview([part.data.fileEdit], 0, options?.maxDiffLines)
    : null;
}

function formatProgressTranscriptPart(part: TranscriptPart): string | null {
  return part.kind === "progress"
    ? dim(`[${part.data.title.toLowerCase()}] ${part.data.detail ?? "Working…"}`)
    : null;
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
  appendCommandPreview(lines, "stdout", data.stdoutPreview, data.stdoutTruncated, (line) => line);
  appendCommandPreview(lines, "stderr", data.stderrPreview, data.stderrTruncated, tone);
  return lines.join("\n");
}

function appendCommandPreview(
  lines: string[],
  label: "stdout" | "stderr",
  preview: string | undefined,
  truncated: boolean | undefined,
  formatLine: (line: string) => string,
): void {
  if (!preview) {
    return;
  }
  if (!preview.includes("\n")) {
    lines.push(`${dim(`  ${label}`)} ${formatLine(preview)}${truncated ? dim(" …") : ""}`);
    return;
  }
  lines.push(`${dim(`  ${label}`)}`);
  lines.push(...preview.split("\n").map((line) => `${dim("    ")}${formatLine(line)}`));
  if (truncated) {
    lines.push(dim("    …"));
  }
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
type ToolParamFormatter = (params: Record<string, unknown>) => string;

const TOOL_PARAM_FORMATTERS: Record<string, ToolParamFormatter> = {
  read_file: summarizeReadFileParams,
  write_file: summarizeWriteFileParams,
  replace_in_file: summarizeReplaceInFileParams,
  search_files: summarizeSearchFilesParams,
  find_files: (params) => (params["pattern"] as string) ?? "",
  run_command: (params) => truncate((params["command"] as string) ?? "", 60),
  git_status: () => "",
  git_diff: summarizeGitDiffParams,
  git_commit: (params) => truncate((params["message"] as string) ?? "", 50),
  update_plan: () => "",
};

export function summarizeToolParams(name: string, params: Record<string, unknown>): string {
  return (TOOL_PARAM_FORMATTERS[name] ?? summarizeUnknownToolParams)(params);
}

function summarizeReadFileParams(params: Record<string, unknown>): string {
  const path = params["path"] as string ?? "";
  const start = params["start_line"] as number | undefined;
  const end = params["end_line"] as number | undefined;
  if (start !== undefined && end !== undefined) {
    return `${path}:${start}-${end}`;
  }
  return start !== undefined ? `${path}:${start}+` : path;
}

function summarizeWriteFileParams(params: Record<string, unknown>): string {
  const path = params["path"] as string ?? "";
  const content = params["content"] as string ?? "";
  return `${path} (${content.length} bytes)`;
}

function summarizeReplaceInFileParams(params: Record<string, unknown>): string {
  const path = (params["path"] as string) ?? "";
  const search = params["search"] as string | undefined;
  const firstLine = search?.split("\n")[0]?.trim();
  return firstLine ? `${path} "${truncate(firstLine, 30)}"` : path;
}

function summarizeSearchFilesParams(params: Record<string, unknown>): string {
  const pattern = params["pattern"] as string ?? "";
  const scope = params["path"] as string | undefined;
  const filePattern = params["file_pattern"] as string | undefined;
  const suffix = filePattern ? ` in ${filePattern}` : scope && scope !== "." ? ` in ${scope}` : "";
  return `"${truncate(pattern, 30)}"${suffix}`;
}

function summarizeGitDiffParams(params: Record<string, unknown>): string {
  return [
    params["staged"] ? "--staged" : "",
    (params["ref"] as string | undefined) ?? "",
    (params["path"] as string | undefined) ?? "",
  ].filter(Boolean).join(" ");
}

function summarizeUnknownToolParams(params: Record<string, unknown>): string {
  const value = Object.values(params).find((entry) => typeof entry === "string" && entry.length > 0);
  return typeof value === "string" ? truncate(value, 40) : "";
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
interface ToolStartFormatOptions {
  readonly name: string;
  readonly params: Record<string, unknown>;
  readonly iteration: number;
  readonly maxIter: number;
  readonly gauge?: string;
  readonly counterLabel?: string;
}

export function formatToolStart(options: ToolStartFormatOptions): string {
  const counter = buildCounter(options.counterLabel, options.iteration, options.maxIter);
  const summary = summarizeToolParams(options.name, options.params);
  const verb = toolVerb(options.name);
  const detail = summary ? ` ${dim(summary)}` : "";
  const gaugeSuffix = options.gauge ? ` ${options.gauge}` : "";
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
interface ToolGroupStartFormatOptions {
  readonly name: string;
  readonly count: number;
  readonly paramSummaries: ReadonlyArray<string>;
  readonly iteration: number;
  readonly maxIter: number;
  readonly gauge?: string;
  readonly counterLabel?: string;
}

export function formatToolGroupStart(options: ToolGroupStartFormatOptions): string {
  const counter = buildCounter(options.counterLabel, options.iteration, options.maxIter);
  const verb = toolVerb(options.name);
  const noun = groupNoun(options.name, options.count);
  const displayed = options.paramSummaries.slice(0, 3);
  const extra = options.paramSummaries.length > 3 ? `, +${options.paramSummaries.length - 3} more` : "";
  const detail = displayed.length > 0
    ? ` ${dim(`(${displayed.join(", ")}${extra})`)}`
    : "";
  const gaugeSuffix = options.gauge ? ` ${options.gauge}` : "";
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
  const parts = buildTurnSummaryParts(stats);
  const status = getTurnStatusFormat(stats.status ?? "completed");
  return `${status.icon} ${bold(status.label)} ${dim(`(${parts.join(", ")})`)}`;
}

function buildTurnSummaryParts(stats: TurnStats): string[] {
  return [
    `${stats.iterationCount} ${pluralize(stats.iterationCount, "iteration", "iterations")}`,
    formatPositiveCount(stats.toolCallCount, "tool call", "tool calls"),
    formatFilesChanged(stats.filesChanged ?? 0),
    stats.validationFailed ? "validation failed" : "",
    formatInputTokens(stats.inputTokens ?? 0),
    stats.costDelta > 0 ? `$${stats.costDelta.toFixed(4)}` : "",
    formatDuration(stats.elapsedMs),
  ].filter(Boolean);
}

function pluralize(count: number, singular: string, plural: string): string {
  return count === 1 ? singular : plural;
}

function formatPositiveCount(count: number, singular: string, plural: string): string {
  return count > 0 ? `${count} ${pluralize(count, singular, plural)}` : "";
}

function formatFilesChanged(filesChanged: number): string {
  return filesChanged > 0
    ? `${filesChanged} ${pluralize(filesChanged, "file", "files")} changed`
    : "";
}

function formatInputTokens(inputTokens: number): string {
  return inputTokens > 0 ? `${Math.round(inputTokens / 1000)}k input tokens` : "";
}

function getTurnStatusFormat(
  status: NonNullable<TurnStats["status"]>,
): { readonly icon: string; readonly label: string } {
  switch (status) {
    case "error":
      return { icon: red("✗"), label: "Finished with errors" };
    case "budget_exceeded":
      return { icon: yellow("!"), label: "Budget exhausted" };
    case "running":
    case "completed":
      return { icon: green("✓"), label: "Done" };
  }
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

export {
  formatCompactionResult,
  formatContextGauge,
  formatEnrichedError,
  formatReasoning,
  inferErrorSuggestion,
} from "./format-context.js";
export {
  formatSessionSummary,
  formatSubagentBatchLaunch,
  formatSubagentError,
  formatSubagentStart,
  summarizeSubagentUpdate,
} from "./format-session.js";
