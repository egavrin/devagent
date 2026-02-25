/**
 * run_command — Execute a shell command with streaming output.
 * Category: workflow (requires approval in suggest/auto-edit modes).
 */

import { spawn } from "node:child_process";
import { resolve } from "node:path";
import type { ToolSpec } from "@devagent/core";
import { ToolError } from "@devagent/core";

const DEFAULT_TIMEOUT_MS = 120_000; // 2 minutes
const MAX_OUTPUT_BYTES = 100_000;

export const runCommandTool: ToolSpec = {
  name: "run_command",
  description:
    "Execute a shell command. Returns stdout and stderr. Times out after 2 minutes by default. Use for builds, tests, linting. Prefer targeted test commands over running the full suite. Never run destructive git commands without user approval. Use the env parameter to set environment variables instead of embedding them in the command string.",
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

    return new Promise((resolvePromise) => {
      const child = spawn("sh", ["-c", command], {
        cwd,
        env: { ...process.env, ...envOverrides },
        stdio: ["ignore", "pipe", "pipe"],
      });

      let stdout = "";
      let stderr = "";
      let killed = false;

      const timer = setTimeout(() => {
        killed = true;
        child.kill("SIGTERM");
      }, timeoutMs);

      child.stdout.on("data", (data: Buffer) => {
        const chunk = data.toString();
        if (stdout.length < MAX_OUTPUT_BYTES) {
          stdout += chunk.substring(0, MAX_OUTPUT_BYTES - stdout.length);
        }
      });

      child.stderr.on("data", (data: Buffer) => {
        const chunk = data.toString();
        if (stderr.length < MAX_OUTPUT_BYTES) {
          stderr += chunk.substring(0, MAX_OUTPUT_BYTES - stderr.length);
        }
      });

      child.on("close", (code) => {
        clearTimeout(timer);

        if (killed) {
          resolvePromise({
            success: false,
            output: stdout,
            error: `Command timed out after ${timeoutMs}ms\n${stderr}`,
            artifacts: [],
          });
          return;
        }

        const exitCode = code ?? 1;
        resolvePromise({
          success: exitCode === 0,
          output: exitCode === 0
            ? stdout || "(no output)"
            : `Exit code: ${exitCode}\nstdout: ${stdout}\nstderr: ${stderr}`,
          error: exitCode === 0 ? null : `Command exited with code ${exitCode}`,
          artifacts: [],
        });
      });

      child.on("error", (err) => {
        clearTimeout(timer);
        resolvePromise({
          success: false,
          output: "",
          error: `Failed to spawn command: ${err.message}`,
          artifacts: [],
        });
      });
    });
  },
};
