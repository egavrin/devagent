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

export interface TestSummary {
  readonly framework: string;
  readonly passed: number;
  readonly failed: number;
  readonly failureMessages: ReadonlyArray<string>;
}

export interface DoubleCheckResult {
  readonly passed: boolean;
  readonly diagnosticErrors: ReadonlyArray<string>;
  readonly testOutput: string | null;
  readonly testPassed: boolean | null;
  /** Parsed test failures (if test output was parseable) */
  readonly testSummary?: TestSummary;
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

// ─── Test Output Parser ─────────────────────────────────────

/**
 * Parse test runner output to extract structured summary.
 * Supports: Jest, Vitest, pytest, cargo test, Go test.
 */
export function parseTestOutput(raw: string): TestSummary | null {
  if (!raw || raw.length < 5) return null;

  // Jest / Vitest: "Tests:  1 failed, 2 passed, 3 total"
  const jestMatch = raw.match(/Tests:\s+(\d+)\s+failed,\s+(\d+)\s+passed(?:,\s+(\d+)\s+total)?/);
  if (jestMatch) {
    return {
      framework: "jest",
      failed: parseInt(jestMatch[1]!, 10),
      passed: parseInt(jestMatch[2]!, 10),
      failureMessages: extractJestFailures(raw),
    };
  }

  // Vitest alternative: "Tests  1 failed | 2 passed (3)"
  const vitestMatch = raw.match(/Tests\s+(\d+)\s+failed\s+\|\s+(\d+)\s+passed/);
  if (vitestMatch) {
    return {
      framework: "vitest",
      failed: parseInt(vitestMatch[1]!, 10),
      passed: parseInt(vitestMatch[2]!, 10),
      failureMessages: extractJestFailures(raw),
    };
  }

  // pytest: "1 failed, 2 passed in 0.5s" or "2 passed in 0.5s"
  const pytestFailMatch = raw.match(/(\d+)\s+failed,\s+(\d+)\s+passed\s+in/);
  if (pytestFailMatch) {
    return {
      framework: "pytest",
      failed: parseInt(pytestFailMatch[1]!, 10),
      passed: parseInt(pytestFailMatch[2]!, 10),
      failureMessages: extractPytestFailures(raw),
    };
  }
  const pytestPassMatch = raw.match(/(\d+)\s+passed\s+in\s+[\d.]+s/);
  if (pytestPassMatch) {
    return {
      framework: "pytest",
      failed: 0,
      passed: parseInt(pytestPassMatch[1]!, 10),
      failureMessages: [],
    };
  }

  // cargo test: "test result: FAILED. 2 passed; 1 failed; 0 ignored"
  const cargoMatch = raw.match(/test result:\s+\w+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed/);
  if (cargoMatch) {
    return {
      framework: "cargo",
      passed: parseInt(cargoMatch[1]!, 10),
      failed: parseInt(cargoMatch[2]!, 10),
      failureMessages: extractCargoFailures(raw),
    };
  }

  // Go test: "FAIL" at end of output + "--- FAIL: TestName"
  const goFailMatch = raw.match(/^FAIL\s/m);
  if (goFailMatch) {
    const goFailTests = raw.match(/--- FAIL: (\S+)/g) ?? [];
    const goPassTests = raw.match(/--- PASS: (\S+)/g) ?? [];
    return {
      framework: "go",
      failed: goFailTests.length,
      passed: goPassTests.length,
      failureMessages: goFailTests.map((m) => m.replace("--- FAIL: ", "")),
    };
  }

  return null;
}

/** Extract failure messages from Jest/Vitest output. */
function extractJestFailures(raw: string): string[] {
  const failures: string[] = [];
  // Match "FAIL" test names: "  ● test name" or "✕ test name"
  const lines = raw.split("\n");
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]!;
    const match = line.match(/^\s+[●✕]\s+(.+)/);
    if (match) {
      // Capture the test name and up to 3 lines of error context
      let msg = match[1]!.trim();
      for (let j = 1; j <= 3 && i + j < lines.length; j++) {
        const ctx = lines[i + j]!.trim();
        if (ctx && !ctx.startsWith("●") && !ctx.startsWith("✕")) {
          msg += "\n    " + ctx;
        } else {
          break;
        }
      }
      failures.push(msg);
    }
  }
  return failures.slice(0, 10); // Cap at 10 failures
}

/** Extract failure messages from pytest output. */
function extractPytestFailures(raw: string): string[] {
  const failures: string[] = [];
  // Match "FAILED test_file.py::test_name" lines
  const lines = raw.split("\n");
  for (const line of lines) {
    const match = line.match(/FAILED\s+(.+)/);
    if (match) {
      failures.push(match[1]!.trim());
    }
  }
  return failures.slice(0, 10);
}

/** Extract failure messages from cargo test output. */
function extractCargoFailures(raw: string): string[] {
  const failures: string[] = [];
  // Match "test module::test_name ... FAILED" lines
  const lines = raw.split("\n");
  for (const line of lines) {
    const match = line.match(/test\s+(\S+)\s+\.\.\.\s+FAILED/);
    if (match) {
      failures.push(match[1]!);
    }
  }
  return failures.slice(0, 10);
}

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
          diagnosticErrors.push(`${file}: diagnostics provider failure: ${message}`);
          this.bus.emit("error", {
            message: `Double-check diagnostics failed for ${file}: ${message}`,
            code: "DOUBLE_CHECK_DIAGNOSTIC_ERROR",
            fatal: false,
          });
        }
      }
    }

    // Run tests if configured
    let testSummary: TestSummary | undefined;
    if (this.options.runTests && this.options.testCommand && this.testRunner) {
      try {
        const result = await this.testRunner(this.options.testCommand);
        testOutput = result.output;
        testPassed = result.success;
        // Parse structured summary from test output
        const parsed = parseTestOutput(result.output);
        if (parsed) {
          testSummary = parsed;
        }
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
      testSummary,
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
      if (result.testSummary) {
        const s = result.testSummary;
        const header = `Test failures (${s.framework}): ${s.failed} failed, ${s.passed} passed`;
        if (s.failureMessages.length > 0) {
          parts.push(
            `${header}\n` +
              s.failureMessages.map((m) => `  - ${m}`).join("\n"),
          );
        } else {
          parts.push(header);
        }
      } else {
        parts.push(`Test failures:\n${result.testOutput ?? "  (no output)"}`);
      }
    }

    return `Double-check: Validation FAILED.\n${parts.join("\n\n")}`;
  }
}
