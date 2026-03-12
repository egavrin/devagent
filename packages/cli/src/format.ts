/**
 * CLI output formatting — colors, spinner, tool call display.
 * Zero external dependencies. Respects NO_COLOR env var.
 * All output goes to stderr (stdout reserved for LLM content).
 */

import type { VerbosityConfig } from "@devagent/runtime";

// ─── Color Helpers ──────────────────────────────────────────

const useColor = !process.env["NO_COLOR"];

function wrap(code: string, s: string): string {
  return useColor ? `\x1b[${code}m${s}\x1b[0m` : s;
}

export function dim(s: string): string { return wrap("90", s); }
export function red(s: string): string { return wrap("31", s); }
export function green(s: string): string { return wrap("32", s); }
export function yellow(s: string): string { return wrap("33", s); }
export function cyan(s: string): string { return wrap("36", s); }
export function bold(s: string): string { return wrap("1", s); }
export function dimBold(s: string): string { return useColor ? `\x1b[90;1m${s}\x1b[0m` : s; }

// ─── Categorical Verbosity ──────────────────────────────────

export function isCategoryEnabled(cat: string, config: VerbosityConfig): boolean {
  if (config.categories.size > 0) {
    return config.categories.has(cat);
  }
  return config.base === "verbose";
}

export function debugLog(cat: string, msg: string, config: VerbosityConfig): void {
  if (!isCategoryEnabled(cat, config)) return;
  process.stderr.write(dim(`[${cat}] ${msg}`) + "\n");
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

// ─── Spinner ────────────────────────────────────────────────

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export const SPINNER_VERBS = [
  "Thinking", "Analyzing", "Considering", "Reasoning", "Processing",
  "Evaluating", "Examining", "Working", "Reflecting", "Assessing",
  "Deliberating", "Exploring", "Weighing", "Pondering", "Reviewing",
];

export class Spinner {
  private frameIndex = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private message = "";

  start(message?: string): void {
    this.stop();
    this.message = message ?? SPINNER_VERBS[Math.floor(Math.random() * SPINNER_VERBS.length)]! + "…";
    this.frameIndex = 0;
    this.render();
    this.timer = setInterval(() => this.render(), 80);
  }

  update(message: string): void {
    this.message = message;
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
    process.stderr.write(`\x1b[2K\r${cyan(frame)} ${dim(this.message)}`);
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

export interface ToolVerbPair {
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

export function toolVerb(name: string): ToolVerbPair {
  return TOOL_VERBS[name] ?? { present: `Calling ${name}`, past: `Called ${name}` };
}

export function formatToolStart(
  name: string,
  params: Record<string, unknown>,
  iteration: number,
  maxIter: number,
  gauge?: string,
): string {
  const counter = maxIter > 0 ? dim(`[${iteration}/${maxIter}]`) : dim(`[${iteration}]`);
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
): string {
  const counter = maxIter > 0 ? dim(`[${iteration}/${maxIter}]`) : dim(`[${iteration}]`);
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

// ─── Summary ────────────────────────────────────────────────

export function formatSummary(iterations: number, elapsedMs: number): string {
  return `${green("✓")} ${bold("Done")} ${dim(`(${iterations} iterations, ${formatDuration(elapsedMs)})`)}`;
}

// ─── Per-Turn Summary ───────────────────────────────────────

export interface TurnStats {
  readonly toolCallCount: number;
  readonly iterationCount: number;
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly costDelta: number;
  readonly elapsedMs: number;
}

export function formatTurnSummary(stats: TurnStats): string {
  const parts: string[] = [];
  parts.push(`${stats.iterationCount} iterations`);
  if (stats.toolCallCount > 0) {
    parts.push(`${stats.toolCallCount} tool calls`);
  }
  if (stats.inputTokens > 0) {
    const kIn = Math.round(stats.inputTokens / 1000);
    parts.push(`${kIn}k input tokens`);
  }
  if (stats.costDelta > 0) {
    parts.push(`$${stats.costDelta.toFixed(4)}`);
  }
  parts.push(formatDuration(stats.elapsedMs));
  return `${green("✓")} ${bold("Done")} ${dim(`(${parts.join(", ")})`)}`;
}

export function formatError(message: string): string {
  return `${red("✗")} ${red(message)}`;
}

// ─── Session Summary ────────────────────────────────────────

export interface SessionSummaryData {
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

  lines.push(dim("─".repeat(50)));
  return lines.join("\n");
}

// ─── Error Enrichment ───────────────────────────────────────

export interface RecentToolResult {
  readonly name: string;
  readonly success: boolean;
  readonly durationMs: number;
}

export interface EnrichedErrorContext {
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

// ─── Turn Separators ────────────────────────────────────────

export function formatTurnHeader(
  turnNumber: number,
  tokenInfo?: { estimated: number; max: number },
  costInfo?: { totalCost: number },
): string {
  const parts: string[] = [`Turn ${turnNumber}`];
  if (tokenInfo && tokenInfo.max > 0 && tokenInfo.estimated > 0) {
    const kEst = Math.round(tokenInfo.estimated / 1000);
    const kMax = Math.round(tokenInfo.max / 1000);
    parts.push(`tokens: ${kEst}k/${kMax}k`);
  }
  if (costInfo && costInfo.totalCost > 0) {
    parts.push(`cost: $${costInfo.totalCost.toFixed(4)}`);
  }
  const inner = parts.join(" | ");
  const padLen = Math.max(0, 50 - inner.length - 6);
  return dim(`── ${inner} ${"─".repeat(padLen)}`);
}

// ─── Compaction Result ──────────────────────────────────────

export interface CompactionResultData {
  readonly tokensBefore: number;
  readonly estimatedTokens: number;
  readonly removedCount: number;
  readonly prunedCount?: number;
  readonly tokensSaved?: number;
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

// ─── Status Bar ─────────────────────────────────────────────

export interface StatusBarData {
  readonly iteration?: number;
  readonly maxIter?: number;
  readonly tokens?: number;
  readonly maxTokens?: number;
  readonly cost?: number;
  readonly branch?: string;
}

export class StatusBar {
  private enabled: boolean;
  private data: StatusBarData = {};
  private rows = 0;
  private cols = 0;
  private initialized = false;

  constructor(enabled: boolean) {
    this.enabled = enabled && !!process.stderr.isTTY;
    if (this.enabled) {
      this.updateDimensions();
    }
  }

  get active(): boolean {
    return this.enabled;
  }

  private updateDimensions(): void {
    this.rows = process.stderr.rows ?? 24;
    this.cols = process.stderr.columns ?? 80;
  }

  /** Initialize scroll region, reserving bottom line. */
  init(): void {
    if (!this.enabled) return;
    this.updateDimensions();
    // Set scroll region to rows 1..(rows-1), leaving last row for status bar
    process.stderr.write(`\x1b[1;${this.rows - 1}r`);
    // Move cursor to top of scroll region
    process.stderr.write(`\x1b[${this.rows - 1};1H`);
    this.initialized = true;
    this.render();

    // Listen for terminal resize
    process.stderr.on("resize", () => {
      this.updateDimensions();
      // Reset scroll region to new size
      process.stderr.write(`\x1b[1;${this.rows - 1}r`);
      this.render();
    });
  }

  update(data: Partial<StatusBarData>): void {
    this.data = { ...this.data, ...data };
    if (this.initialized) this.render();
  }

  formatLine(data: StatusBarData): string {
    const parts: string[] = [];

    // Iteration
    if (data.iteration !== undefined) {
      if (data.maxIter && data.maxIter > 0) {
        parts.push(`iter ${data.iteration}/${data.maxIter}`);
      } else {
        parts.push(`iter ${data.iteration}`);
      }
    }

    // Tokens (color-coded)
    if (data.tokens && data.maxTokens && data.tokens > 0 && data.maxTokens > 0) {
      const kTokens = Math.round(data.tokens / 1000);
      const kMax = Math.round(data.maxTokens / 1000);
      const ratio = data.tokens / data.maxTokens;
      const tokenStr = `${kTokens}k/${kMax}k`;
      if (ratio > 0.8) {
        parts.push(red(tokenStr));
      } else if (ratio > 0.6) {
        parts.push(yellow(tokenStr));
      } else {
        parts.push(tokenStr);
      }
    }

    // Cost
    if (data.cost && data.cost > 0) {
      parts.push(`$${data.cost.toFixed(4)}`);
    }

    // Branch
    if (data.branch) {
      parts.push(data.branch);
    }

    return dim("[") + parts.join(dim(" | ")) + dim("]");
  }

  render(): void {
    if (!this.enabled || !this.initialized) return;
    this.updateDimensions();
    const line = this.formatLine(this.data);

    // Save cursor, move to last row, clear line, write status, restore cursor
    process.stderr.write(
      `\x1b7\x1b[${this.rows};1H\x1b[2K${line}\x1b8`,
    );
  }

  /** Restore full scroll region and clear status bar line. */
  cleanup(): void {
    if (!this.enabled || !this.initialized) return;
    // Reset scroll region to full terminal
    process.stderr.write(`\x1b[1;${this.rows}r`);
    // Clear the last line
    process.stderr.write(`\x1b[${this.rows};1H\x1b[2K`);
    // Move cursor back
    process.stderr.write(`\x1b[${this.rows - 1};1H`);
    this.initialized = false;
  }
}

// ─── Helpers ────────────────────────────────────────────────

export function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.substring(0, maxLen - 1) + "…";
}

export function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = Math.round((ms % 60000) / 1000);
  return `${mins}m ${secs}s`;
}
