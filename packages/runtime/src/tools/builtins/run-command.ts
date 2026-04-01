/**
 * run_command — Execute a shell command with streaming output.
 * Category: workflow (requires approval in suggest/auto-edit modes).
 */

import { resolve } from "node:path";
import type { ToolSpec } from "../../core/index.js";
import { spawnAndCapture } from "./spawn-capture.js";

const DEFAULT_TIMEOUT_MS = 120_000; // 2 minutes
const MAX_COMMAND_TIMEOUT_MS = 600_000; // 10 minutes — hard cap for LLM-provided values
const MAX_OUTPUT_BYTES = 100_000;

function withTruncationMarker(
  streamName: "stdout" | "stderr",
  text: string,
  maxBytes: number,
): string {
  if (text.length < maxBytes) {
    return text;
  }

  const marker = `[output truncated: ${streamName} capped at ${maxBytes} bytes]`;
  return text.length > 0 ? `${text}\n${marker}` : marker;
}

export const runCommandTool: ToolSpec = {
  name: "run_command",
  description:
    "Execute a shell command. Returns stdout and stderr. Times out after 2 minutes by default. " +
    "Use for builds, tests, linting. For git operations, prefer dedicated tools (git_diff, git_status) — " +
    "they are context-optimized. Prefer targeted test commands over running the full suite. " +
    "Never run destructive git commands without user approval. " +
    "Use the env parameter to set environment variables instead of embedding them in the command string.",
  category: "workflow",
  paramSchema: {
    type: "object",
    properties: {
      command: { type: "string", description: "Shell command to execute" },
      cwd: {
        type: "string",
        description: "Working directory (relative to repo root, default: '.')",
      },
      timeout_ms: {
        type: "number",
        description: "Timeout in milliseconds (default: 120000, max: 600000)",
        maximum: 600_000,
      },
      env: {
        type: "string",
        description:
          'JSON object of environment variable overrides. Merged with the current process environment (these values take precedence). Example: {"DYLD_LIBRARY_PATH": "/usr/local/lib"}',
      },
    },
    required: ["command"],
  },
  errorGuidance: {
    common: "Read the earliest stderr line — it is usually the root cause. Fix the underlying code or config, then re-run.",
    patterns: [
      { match: "timed out", hint: "Command timed out. If searching a large repo, narrow scope: use --include/--exclude flags, limit depth with -maxdepth, or search specific subdirectories instead of the root. Prefer builtin search_files/find_files tools over shell grep/find when possible." },
      { match: "not found", hint: "Command not found. Check the project's package.json scripts or use the project's package manager." },
      { match: "Permission denied", hint: "Permission denied on some paths. The command may still have partial results. Try adding 2>/dev/null to suppress permission errors, or narrow the search scope." },
    ],
  },
  resultSchema: {
    type: "object",
    properties: {
      stdout: { type: "string" },
      stderr: { type: "string" },
      exit_code: { type: "number" },
    },
  },
  handler: async (params, context) => {
    const command = params["command"] as string;
    const cwd = resolve(
      context.repoRoot,
      (params["cwd"] as string | undefined) ?? ".",
    );
    const timeoutMs = Math.min(
      (params["timeout_ms"] as number | undefined) ?? DEFAULT_TIMEOUT_MS,
      MAX_COMMAND_TIMEOUT_MS,
    );
    let envOverrides: Record<string, string> = {};
    const envParam = params["env"];
    if (typeof envParam === "string" && envParam.length > 0) {
      try {
        envOverrides = JSON.parse(envParam) as Record<string, string>;
      } catch {
        return {
          success: false,
          output: "",
          error: `Invalid env JSON: ${envParam}`,
          artifacts: [],
        };
      }
    } else if (typeof envParam === "object" && envParam !== null) {
      // Also accept a direct object (e.g., from non-OpenAI providers)
      envOverrides = envParam as Record<string, string>;
    }

    const result = await spawnAndCapture("sh", ["-c", command], {
      cwd,
      timeout: timeoutMs,
      maxBytes: MAX_OUTPUT_BYTES,
      env: Object.keys(envOverrides).length > 0 ? envOverrides : undefined,
    });

    const safeStdout = withTruncationMarker("stdout", result.stdout, MAX_OUTPUT_BYTES);
    const safeStderr = withTruncationMarker("stderr", result.stderr, MAX_OUTPUT_BYTES);

    if (result.timedOut) {
      // Detect large-repo search patterns and provide targeted guidance
      const scopingHint = detectSearchScopingHint(command);
      return {
        success: false,
        output: safeStdout,
        error: `Command timed out after ${timeoutMs}ms\n${safeStderr}${scopingHint ? `\n[Hint] ${scopingHint}` : ""}`,
        artifacts: [],
      };
    }

    if (result.exitCode !== 0) {
      // Partial success heuristic: if stdout has substantial content and stderr
      // contains only warnings (permission denied, broken pipes), treat as success
      // with warnings. Common with `find` on restricted directories.
      const isPartialSuccess = safeStdout.length > 100 && isStderrWarningOnly(safeStderr);
      if (isPartialSuccess) {
        return {
          success: true,
          output: `${safeStdout}\n\n[Warning: exit code ${result.exitCode}. Some errors during execution:]\n${safeStderr}`,
          error: null,
          artifacts: [],
        };
      }

      return {
        success: false,
        output: `Exit code: ${result.exitCode}\nstdout: ${safeStdout}\nstderr: ${safeStderr}`,
        error: `Command exited with code ${result.exitCode}`,
        artifacts: [],
      };
    }

    return {
      success: true,
      output: safeStdout || "(no output)",
      error: null,
      artifacts: [],
    };
  },
};

// ─── Helpers ────────────────────────────────────────────────

/**
 * Detect if a timed-out command is a large-repo search and provide
 * specific scoping guidance.
 */
function detectSearchScopingHint(command: string): string | null {
  const isGrep = /\bgrep\s+-[rRl]*R/i.test(command) || /\bgrep\s.*-r\b/i.test(command);
  const isFind = /\bfind\b/.test(command) && !command.includes("-maxdepth");
  const isRg = /\brg\b/.test(command);

  if (isGrep) {
    return "grep -R on large repos is slow. Scope with --include='*.ext', search specific subdirectories, or use rg (ripgrep) which is much faster. Add 2>/dev/null to suppress permission errors.";
  }
  if (isFind) {
    return "find without -maxdepth traverses the entire tree. Add -maxdepth 3 to limit depth, or narrow the starting directory.";
  }
  if (isRg) {
    return "rg timed out. Use -g/--glob to filter file types, or search specific subdirectories instead of the repo root.";
  }
  return null;
}

/** Stderr patterns that indicate warnings, not fatal errors. */
const STDERR_WARNING_PATTERNS = [
  /permission denied/i,
  /operation not permitted/i,
  /no such file or directory/i,
  /broken pipe/i,
  /grep:.*binary file/i,
  /find:.*\bpermission\b/i,
  /warning:/i,
];

/**
 * Check if stderr contains only warning-like messages (not fatal errors).
 * Used to distinguish partial success (find with some permission errors)
 * from true failures (command not found, syntax errors).
 */
function isStderrWarningOnly(stderr: string): boolean {
  if (!stderr.trim()) return true;
  const lines = stderr.split("\n").filter((l) => l.trim().length > 0);
  return lines.every((line) =>
    STDERR_WARNING_PATTERNS.some((pattern) => pattern.test(line)),
  );
}
