/**
 * Shared subprocess spawning utility.
 *
 * Captures stdout and stderr with byte-size limits and an optional timeout.
 * Used by run_command, double-check-wiring, and any other code that needs
 * to run a child process and collect its output.
 */

import { spawn } from "node:child_process";

const DEFAULT_TIMEOUT_MS = 120_000; // 2 minutes
const DEFAULT_MAX_BYTES = 100_000;

/** Node.js setTimeout uses a 32-bit signed integer; values above this overflow to 1ms. */
const MAX_SAFE_TIMEOUT_MS = 2_147_483_647; // 2^31 - 1

export interface SpawnCaptureOptions {
  /** Working directory for the subprocess. */
  readonly cwd: string;
  /** Timeout in milliseconds. Defaults to 120 000 (2 minutes). */
  readonly timeout?: number;
  /** Maximum bytes to keep per stream. Defaults to 100 000. */
  readonly maxBytes?: number;
  /** Environment variable overrides (merged with process.env). */
  readonly env?: Record<string, string>;
}

export interface SpawnCaptureResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  /** True when the process was killed because it exceeded the timeout. */
  readonly timedOut: boolean;
}

/**
 * Spawn a subprocess and capture its stdout/stderr up to `maxBytes`.
 *
 * The promise always resolves (never rejects) — spawn errors are surfaced
 * through `exitCode: 1` and an error description in `stderr`.
 */
export function spawnAndCapture(
  command: string,
  args: string[],
  options: SpawnCaptureOptions,
): Promise<SpawnCaptureResult> {
  const { cwd, timeout: rawTimeout = DEFAULT_TIMEOUT_MS, maxBytes = DEFAULT_MAX_BYTES, env } = options;
  const timeout = Math.min(rawTimeout, MAX_SAFE_TIMEOUT_MS);

  return new Promise((resolve) => {
    const child = spawn(command, args, {
      cwd,
      env: env ? { ...process.env, ...env } : undefined,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";
    let killed = false;

    child.stdout.on("data", (data: Buffer) => {
      const chunk = data.toString();
      if (stdout.length < maxBytes) {
        stdout += chunk.substring(0, maxBytes - stdout.length);
      }
    });

    child.stderr.on("data", (data: Buffer) => {
      const chunk = data.toString();
      if (stderr.length < maxBytes) {
        stderr += chunk.substring(0, maxBytes - stderr.length);
      }
    });

    const timer = setTimeout(() => {
      killed = true;
      child.kill("SIGTERM");
      resolve({ exitCode: 1, stdout, stderr: `Timed out after ${timeout}ms`, timedOut: true });
    }, timeout);

    child.on("close", (code) => {
      clearTimeout(timer);
      if (killed) return; // already resolved via timeout
      resolve({ exitCode: code ?? 1, stdout, stderr, timedOut: false });
    });

    child.on("error", (err) => {
      clearTimeout(timer);
      if (killed) return;
      resolve({ exitCode: 1, stdout: "", stderr: err.message, timedOut: false });
    });
  });
}
