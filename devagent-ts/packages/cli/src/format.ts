/**
 * CLI output formatting — colors, spinner, tool call display.
 * Zero external dependencies. Respects NO_COLOR env var.
 * All output goes to stderr (stdout reserved for LLM content).
 */

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

// ─── Spinner ────────────────────────────────────────────────

const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export class Spinner {
  private frameIndex = 0;
  private timer: ReturnType<typeof setInterval> | null = null;
  private message = "";

  start(message: string): void {
    this.stop();
    this.message = message;
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
function summarizeToolParams(name: string, params: Record<string, unknown>): string {
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
      // For unknown tools (MCP, plugins), show first string param
      for (const value of Object.values(params)) {
        if (typeof value === "string" && value.length > 0) {
          return truncate(value, 40);
        }
      }
      return "";
  }
}

export function formatToolStart(
  name: string,
  params: Record<string, unknown>,
  iteration: number,
  maxIter: number,
): string {
  const counter = maxIter > 0 ? dim(`[${iteration}/${maxIter}]`) : dim(`[${iteration}]`);
  const summary = summarizeToolParams(name, params);
  const detail = summary ? ` ${dim(summary)}` : "";
  return `${counter} ${dimBold("↳")} ${bold(name)}${detail}`;
}

export function formatToolEnd(
  name: string,
  success: boolean,
  durationMs: number,
  error?: string,
): string {
  const duration = dim(`(${formatDuration(durationMs)})`);
  if (success) {
    return `  ${green("✓")} ${dim(name)} ${duration}`;
  }
  const errMsg = error ? `: ${truncate(error, 80)}` : "";
  return `  ${red("✗")} ${dim(name)} ${duration}${red(errMsg)}`;
}

// ─── Plan Rendering ─────────────────────────────────────────

export function formatPlan(
  steps: ReadonlyArray<{ description: string; status: string }>,
): string {
  const lines = steps.map((s) => {
    switch (s.status) {
      case "completed":
        return `  ${green("[x]")} ${dim(s.description)}`;
      case "in_progress":
        return `  ${yellow("[>]")} ${s.description}`;
      default:
        return `  ${dim("[ ]")} ${dim(s.description)}`;
    }
  });
  return lines.join("\n");
}

// ─── Summary ────────────────────────────────────────────────

export function formatSummary(iterations: number, elapsedMs: number): string {
  return `${green("✓")} ${bold("Done")} ${dim(`(${iterations} iterations, ${formatDuration(elapsedMs)})`)}`;
}

export function formatError(message: string): string {
  return `${red("✗")} ${red(message)}`;
}

// ─── Helpers ────────────────────────────────────────────────

function truncate(s: string, maxLen: number): string {
  if (s.length <= maxLen) return s;
  return s.substring(0, maxLen - 1) + "…";
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
  const mins = Math.floor(ms / 60000);
  const secs = Math.round((ms % 60000) / 1000);
  return `${mins}m ${secs}s`;
}
