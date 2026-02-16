/**
 * Double-check completion — validates after mutating tool executions.
 * Checks that no new errors were introduced by running diagnostics.
 * Configurable: off by default, can be enabled per-session.
 *
 * From Cline v3.58+: Before marking a task done, verify acceptance criteria.
 */

import type { EventBus } from "@devagent/core";

// ─── Types ──────────────────────────────────────────────────

export interface DoubleCheckOptions {
  /** Enable double-check validation (default: false) */
  readonly enabled: boolean;
  /** Run diagnostics after file writes (default: true when enabled) */
  readonly checkDiagnostics: boolean;
  /** Run configured test command after edits (default: false) */
  readonly runTests: boolean;
  /** Test command to run (e.g., "bun test") */
  readonly testCommand: string | null;
  /** Max time to wait for diagnostics in ms (default: 5000) */
  readonly diagnosticTimeout: number;
}

export interface DoubleCheckResult {
  readonly passed: boolean;
  readonly diagnosticErrors: ReadonlyArray<string>;
  readonly testOutput: string | null;
  readonly testPassed: boolean | null;
}

export type DiagnosticProvider = (filePath: string) => Promise<ReadonlyArray<{
  readonly message: string;
  readonly severity: string;
}>>;

export type TestRunner = (command: string) => Promise<{
  readonly success: boolean;
  readonly output: string;
}>;

// ─── Default Options ────────────────────────────────────────

export const DEFAULT_DOUBLE_CHECK_OPTIONS: DoubleCheckOptions = {
  enabled: false,
  checkDiagnostics: true,
  runTests: false,
  testCommand: null,
  diagnosticTimeout: 5_000,
};

// ─── Double Check ───────────────────────────────────────────

export class DoubleCheck {
  private readonly options: DoubleCheckOptions;
  private readonly bus: EventBus;
  private diagnosticProvider: DiagnosticProvider | null = null;
  private testRunner: TestRunner | null = null;

  constructor(options: DoubleCheckOptions, bus: EventBus) {
    this.options = options;
    this.bus = bus;
  }

  /**
   * Set the diagnostic provider (e.g., LSP client).
   */
  setDiagnosticProvider(provider: DiagnosticProvider): void {
    this.diagnosticProvider = provider;
  }

  /**
   * Set the test runner (e.g., shell command executor).
   */
  setTestRunner(runner: TestRunner): void {
    this.testRunner = runner;
  }

  /**
   * Check if double-check is enabled.
   */
  isEnabled(): boolean {
    return this.options.enabled;
  }

  /**
   * Run double-check validation on a file that was modified.
   * Returns structured results — the caller (TaskLoop) can feed
   * these back to the LLM for self-correction.
   */
  async check(modifiedFiles: ReadonlyArray<string>): Promise<DoubleCheckResult> {
    if (!this.options.enabled) {
      return {
        passed: true,
        diagnosticErrors: [],
        testOutput: null,
        testPassed: null,
      };
    }

    const diagnosticErrors: string[] = [];
    let testOutput: string | null = null;
    let testPassed: boolean | null = null;

    // Check diagnostics for modified files
    if (this.options.checkDiagnostics && this.diagnosticProvider) {
      for (const file of modifiedFiles) {
        try {
          const diagnostics = await this.diagnosticProvider(file);
          const errors = diagnostics.filter((d) => d.severity === "error");
          for (const err of errors) {
            diagnosticErrors.push(`${file}: ${err.message}`);
          }
        } catch (err) {
          const message = err instanceof Error ? err.message : String(err);
          this.bus.emit("error", {
            message: `Double-check diagnostics failed for ${file}: ${message}`,
            code: "DOUBLE_CHECK_DIAGNOSTIC_ERROR",
            fatal: false,
          });
        }
      }
    }

    // Run tests if configured
    if (this.options.runTests && this.options.testCommand && this.testRunner) {
      try {
        const result = await this.testRunner(this.options.testCommand);
        testOutput = result.output;
        testPassed = result.success;
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        testOutput = `Test execution failed: ${message}`;
        testPassed = false;
      }
    }

    const passed =
      diagnosticErrors.length === 0 &&
      (testPassed === null || testPassed);

    return {
      passed,
      diagnosticErrors,
      testOutput,
      testPassed,
    };
  }

  /**
   * Format double-check results as a string for LLM feedback.
   */
  formatResults(result: DoubleCheckResult): string {
    if (result.passed) {
      return "Double-check: All validations passed.";
    }

    const parts: string[] = [];

    if (result.diagnosticErrors.length > 0) {
      parts.push(
        `Diagnostic errors (${result.diagnosticErrors.length}):\n` +
          result.diagnosticErrors.map((e) => `  - ${e}`).join("\n"),
      );
    }

    if (result.testPassed === false) {
      parts.push(`Test failures:\n${result.testOutput ?? "  (no output)"}`);
    }

    return `Double-check: Validation FAILED.\n${parts.join("\n\n")}`;
  }
}
