/**
 * ArkTS Linter — subprocess wrapper for ets2panda/linter (tslinter).
 *
 * Invokes the real AST-based linter as a child process with --arkts-2 --ide-interactive,
 * parses its JSON output, and returns structured problem reports.
 */

import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { resolve, join } from "node:path";

import type { TsLinterProblem } from "./rules.js";
import { parseTsLinterLine } from "./rules.js";

// ─── Types ──────────────────────────────────────────────────

interface ArkTSLinterOptions {
  /** Path to the ets2panda/linter directory (contains dist/tslinter.js). */
  readonly linterPath: string;
  /** Enable ArkTS 2.0 strict mode. Default: true. */
  readonly arkts2?: boolean;
  /** Enable autofix suggestions. Default: false. */
  readonly autofix?: boolean;
  /** Subprocess timeout in ms. Default: 60000. */
  readonly timeout?: number;
}

// ─── Linter ─────────────────────────────────────────────────

export class ArkTSLinter {
  /** Resolved path to the linter root directory (used as cwd for subprocess). */
  private readonly linterDir: string;
  private readonly tslinterPath: string;
  private readonly arkts2: boolean;
  private readonly autofix: boolean;
  private readonly timeout: number;

  constructor(options: ArkTSLinterOptions) {
    this.linterDir = resolve(options.linterPath);
    this.tslinterPath = resolve(options.linterPath, "dist", "tslinter.js");
    this.arkts2 = options.arkts2 ?? true;
    this.autofix = options.autofix ?? false;
    this.timeout = options.timeout ?? 60_000;

    if (!existsSync(this.tslinterPath)) {
      throw new Error(
        `tslinter.js not found at ${this.tslinterPath}. ` +
        `Build the linter: cd ${options.linterPath} && npm install && npm run build`
      );
    }
  }

  /**
   * Lint a single file. Returns problems found.
   * Only processes .ets files — returns [] for other extensions.
   */
  async lintFile(filePath: string): Promise<ReadonlyArray<TsLinterProblem>> {
    const absPath = resolve(filePath);
    if (!absPath.endsWith(".ets")) {
      return [];
    }

    const results = await this.runTsLinter([absPath]);
    return results.get(absPath) ?? [];
  }

  /**
   * Lint a project folder. Returns problems keyed by file path.
   */
  async lintFolder(folderPath: string): Promise<Map<string, ReadonlyArray<TsLinterProblem>>> {
    const absPath = resolve(folderPath);
    return this.runTsLinter(["-f", absPath]);
  }

  /**
   * Run the tslinter subprocess with the given arguments.
   * Returns a map from file path to problems.
   */
  private runTsLinter(extraArgs: ReadonlyArray<string>): Promise<Map<string, ReadonlyArray<TsLinterProblem>>> {
    return new Promise((resolveP, reject) => {
      const args = [this.tslinterPath, "--ide-interactive"];
      if (this.arkts2) args.push("--arkts-2");
      if (this.autofix) args.push("--autofix");
      args.push(...extraArgs);

      // Run from the linter directory so it can find tsconfig-sdk.json and rule-config.json
      const child = spawn("node", args, {
        cwd: this.linterDir,
        stdio: ["pipe", "pipe", "pipe"],
        timeout: this.timeout,
      });

      let stdout = "";
      let stderr = "";

      child.stdout.on("data", (chunk: Buffer) => {
        stdout += chunk.toString();
      });

      child.stderr.on("data", (chunk: Buffer) => {
        stderr += chunk.toString();
      });

      child.on("error", (err) => {
        reject(new Error(`tslinter process error: ${err.message}`));
      });

      child.on("close", (code) => {
        // tslinter exits with 0 on success and 1 when lint errors found (both valid)
        if (code !== null && code > 1) {
          const errText = (stderr || stdout).slice(0, 500);
          reject(new Error(`tslinter exited with code ${code}: ${errText}`));
          return;
        }

        const results = new Map<string, ReadonlyArray<TsLinterProblem>>();

        // Parse JSON lines from stdout
        const lines = stdout.split("\n");
        for (const line of lines) {
          const parsed = parseTsLinterLine(line);
          if (parsed && parsed.problems.length > 0) {
            results.set(parsed.filePath, parsed.problems);
          }
        }

        resolveP(results);
      });

      // Close stdin to signal we're not sending anything
      child.stdin.end();
    });
  }
}

/**
 * Check if the tslinter binary exists at the given linter path.
 */
export function isTsLinterAvailable(linterPath: string): boolean {
  return existsSync(join(linterPath, "dist", "tslinter.js"));
}
