/**
 * run_command — Execute a shell command with streaming output.
 * Category: workflow (approval depends on the active safety preset or legacy mode).
 */

import { resolve } from "node:path";

import { spawnAndCapture } from "./spawn-capture.js";
import type { ToolSpec , ToolCommandResultMetadata } from "../../core/index.js";

const DEFAULT_TIMEOUT_MS = 120_000; // 2 minutes
const MAX_COMMAND_TIMEOUT_MS = 600_000; // 10 minutes — hard cap for LLM-provided values
const MAX_OUTPUT_BYTES = 100_000;
const MAX_PREVIEW_CHARS = 1_200;
const MAX_PREVIEW_LINES = 12;

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

function buildPreview(
  text: string,
): {
  readonly preview: string;
  readonly truncated: boolean;
} {
  if (!text) {
    return { preview: "", truncated: false };
  }

  const lines = text.split("\n");
  const clippedByLines = lines.length > MAX_PREVIEW_LINES;
  const visibleLines = clippedByLines ? lines.slice(0, MAX_PREVIEW_LINES) : lines;
  let preview = visibleLines.join("\n");
  const clippedByChars = preview.length > MAX_PREVIEW_CHARS;
  if (clippedByChars) {
    preview = preview.slice(0, MAX_PREVIEW_CHARS);
  }
  if (clippedByLines || clippedByChars) {
    preview = `${preview}\n[preview truncated]`;
  }
  return {
    preview,
    truncated: clippedByLines || clippedByChars,
  };
}
function buildCommandResultMetadata(
  input: {
    readonly command: string;
    readonly cwd: string;
    readonly exitCode: number | null;
    readonly stdout: string;
    readonly stderr: string;
    readonly timedOut?: boolean;
    readonly warningOnly?: boolean;
    readonly stdoutWasCapped?: boolean;
    readonly stderrWasCapped?: boolean;
  },
): ToolCommandResultMetadata {
  const stdoutPreview = buildPreview(input.stdout);
  const stderrPreview = buildPreview(input.stderr);
  return {
    command: input.command,
    cwd: input.cwd,
    exitCode: input.exitCode,
    timedOut: input.timedOut === true,
    warningOnly: input.warningOnly === true,
    stdoutPreview: stdoutPreview.preview,
    stderrPreview: stderrPreview.preview,
    stdoutTruncated: (input.stdoutWasCapped ?? false) || stdoutPreview.truncated,
    stderrTruncated: (input.stderrWasCapped ?? false) || stderrPreview.truncated,
  };
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
    const request = parseRunCommandRequest(params, context.repoRoot);
    if ("error" in request) return request.error;

    const result = await spawnAndCapture("sh", ["-c", request.command], {
      cwd: request.cwd,
      timeout: request.timeoutMs,
      maxBytes: MAX_OUTPUT_BYTES,
      env: Object.keys(request.envOverrides).length > 0 ? request.envOverrides : undefined,
    });

    const safeStdout = withTruncationMarker("stdout", result.stdout, MAX_OUTPUT_BYTES);
    const safeStderr = withTruncationMarker("stderr", result.stderr, MAX_OUTPUT_BYTES);
    const stdoutWasCapped = result.stdout.length >= MAX_OUTPUT_BYTES;
    const stderrWasCapped = result.stderr.length >= MAX_OUTPUT_BYTES;

    if (result.timedOut) {
      return buildTimedOutResult(request, {
        safeStdout,
        safeStderr,
        stdoutWasCapped,
        stderrWasCapped,
      });
    }

    if (result.exitCode !== 0) {
      const isPartialSuccess = safeStdout.length > 100 && isStderrWarningOnly(safeStderr);
      return isPartialSuccess
        ? buildWarningOnlyResult(request, result.exitCode, {
          safeStdout,
          safeStderr,
          stdoutWasCapped,
          stderrWasCapped,
        })
        : buildExitFailureResult(request, result.exitCode, {
          safeStdout,
          safeStderr,
          stdoutWasCapped,
          stderrWasCapped,
        });
    }

    return buildSuccessResult(request, result.exitCode, {
      safeStdout,
      safeStderr,
      stdoutWasCapped,
      stderrWasCapped,
    });
  },
};

// ─── Helpers ────────────────────────────────────────────────

interface RunCommandRequest {
  readonly command: string;
  readonly cwdParam: string;
  readonly cwd: string;
  readonly timeoutMs: number;
  readonly envOverrides: Record<string, string>;
}

interface CommandOutputState {
  readonly safeStdout: string;
  readonly safeStderr: string;
  readonly stdoutWasCapped: boolean;
  readonly stderrWasCapped: boolean;
}

function parseRunCommandRequest(
  params: Record<string, unknown>,
  repoRoot: string,
): RunCommandRequest | { readonly error: ReturnType<typeof invalidEnvResult> } {
  const command = params["command"] as string;
  const cwdParam = (params["cwd"] as string | undefined) ?? ".";
  const envResult = parseEnvOverrides(params["env"]);
  if ("error" in envResult) {
    return { error: invalidEnvResult(command, cwdParam, envResult.error) };
  }
  return {
    command,
    cwdParam,
    cwd: resolve(repoRoot, cwdParam),
    timeoutMs: Math.min(
      (params["timeout_ms"] as number | undefined) ?? DEFAULT_TIMEOUT_MS,
      MAX_COMMAND_TIMEOUT_MS,
    ),
    envOverrides: envResult.value,
  };
}

function parseEnvOverrides(envParam: unknown): { readonly value: Record<string, string> } | { readonly error: string } {
  if (typeof envParam === "string" && envParam.length > 0) return parseStringEnv(envParam);
  if (typeof envParam === "object" && envParam !== null) {
    return { value: envParam as Record<string, string> };
  }
  return { value: {} };
}

function parseStringEnv(envParam: string): { readonly value: Record<string, string> } | { readonly error: string } {
  try {
    const parsedEnv = JSON.parse(envParam) as unknown;
    if (parsedEnv == null) return { value: {} };
    if (typeof parsedEnv === "object" && !Array.isArray(parsedEnv)) {
      return { value: parsedEnv as Record<string, string> };
    }
  } catch {
    // Fall through to the shared error result below.
  }
  return { error: `Invalid env JSON: ${envParam}` };
}

function invalidEnvResult(command: string, cwd: string, error: string) {
  return {
    success: false,
    output: "",
    error,
    artifacts: [],
    metadata: {
      commandResult: commandMetadata({
        command,
        cwd,
        exitCode: null,
        stdout: "",
        stderr: "",
      }),
    },
  };
}

function commandMetadata(
  input: {
    readonly command: string;
    readonly cwd: string;
    readonly exitCode: number | null;
    readonly stdout: string;
    readonly stderr: string;
    readonly timedOut?: boolean;
    readonly warningOnly?: boolean;
    readonly stdoutWasCapped?: boolean;
    readonly stderrWasCapped?: boolean;
  },
): ToolCommandResultMetadata {
  return buildCommandResultMetadata({
    ...input,
  });
}

function buildTimedOutResult(
  request: RunCommandRequest,
  output: CommandOutputState,
) {
  const scopingHint = detectSearchScopingHint(request.command);
  return {
    success: false,
    output: output.safeStdout,
    error: `Command timed out after ${request.timeoutMs}ms\n${output.safeStderr}${scopingHint ? `\n[Hint] ${scopingHint}` : ""}`,
    artifacts: [],
    metadata: {
      commandResult: commandMetadata({
        command: request.command,
        cwd: request.cwdParam,
        exitCode: null,
        stdout: output.safeStdout,
        stderr: output.safeStderr,
        timedOut: true,
        stdoutWasCapped: output.stdoutWasCapped,
        stderrWasCapped: output.stderrWasCapped,
      }),
    },
  };
}

function buildWarningOnlyResult(
  request: RunCommandRequest,
  exitCode: number,
  output: CommandOutputState,
) {
  return {
    success: true,
    output: `${output.safeStdout}\n\n[Warning: exit code ${exitCode}. Some errors during execution:]\n${output.safeStderr}`,
    error: null,
    artifacts: [],
    metadata: {
      commandResult: commandMetadata({
        command: request.command,
        cwd: request.cwdParam,
        exitCode,
        stdout: output.safeStdout,
        stderr: output.safeStderr,
        warningOnly: true,
        stdoutWasCapped: output.stdoutWasCapped,
        stderrWasCapped: output.stderrWasCapped,
      }),
    },
  };
}

function buildExitFailureResult(
  request: RunCommandRequest,
  exitCode: number,
  output: CommandOutputState,
) {
  return {
    success: false,
    output: `Exit code: ${exitCode}\nstdout: ${output.safeStdout}\nstderr: ${output.safeStderr}`,
    error: `Command exited with code ${exitCode}`,
    artifacts: [],
    metadata: {
      commandResult: commandMetadata({
        command: request.command,
        cwd: request.cwdParam,
        exitCode,
        stdout: output.safeStdout,
        stderr: output.safeStderr,
        stdoutWasCapped: output.stdoutWasCapped,
        stderrWasCapped: output.stderrWasCapped,
      }),
    },
  };
}

function buildSuccessResult(
  request: RunCommandRequest,
  exitCode: number | null,
  output: CommandOutputState,
) {
  const resultOutput = output.safeStdout || "(no output)";
  return {
    success: true,
    output: resultOutput,
    error: null,
    artifacts: [],
    metadata: {
      commandResult: commandMetadata({
        command: request.command,
        cwd: request.cwdParam,
        exitCode,
        stdout: resultOutput,
        stderr: output.safeStderr,
        stdoutWasCapped: output.stdoutWasCapped,
        stderrWasCapped: output.stderrWasCapped,
      }),
    },
  };
}

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
