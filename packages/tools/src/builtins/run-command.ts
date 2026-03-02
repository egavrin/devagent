/**
 * run_command — Execute a shell command with streaming output.
 * Category: workflow (requires approval in suggest/auto-edit modes).
 */

import { resolve } from "node:path";
import type { ToolSpec } from "@devagent/core";
import { spawnAndCapture } from "./spawn-capture.js";

const DEFAULT_TIMEOUT_MS = 120_000; // 2 minutes
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
        description: "Timeout in milliseconds (default: 120000)",
      },
      env: {
        type: "string",
        description:
          'JSON object of environment variable overrides. Merged with the current process environment (these values take precedence). Example: {"DYLD_LIBRARY_PATH": "/usr/local/lib"}',
      },
    },
    required: ["command"],
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
    const timeoutMs =
      (params["timeout_ms"] as number | undefined) ?? DEFAULT_TIMEOUT_MS;
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
      return {
        success: false,
        output: safeStdout,
        error: `Command timed out after ${timeoutMs}ms\n${safeStderr}`,
        artifacts: [],
      };
    }

    if (result.exitCode !== 0) {
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
