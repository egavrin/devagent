/**
 * Double-check completion — validates after mutating tool executions.
 * Checks that no new errors were introduced by running diagnostics.
 * Configurable: off by default, can be enabled per-session.
 *
 * From Cline v3.58+: Before marking a task done, verify acceptance criteria.
 */

import type { EventBus, DoubleCheckConfig } from "../core/index.js";
import { extractErrorMessage } from "../core/index.js";

// ─── Types ──────────────────────────────────────────────────

/** Engine-resolved double-check options — all fields required.
 *  Extends DoubleCheckConfig from @devagent/runtime. */
export interface DoubleCheckOptions extends Required<DoubleCheckConfig> {}

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
  /** Number of pre-existing errors that were filtered out (for display) */
  readonly baselineFiltered?: number;
}

/** Per-file error counts captured before an edit, used to filter pre-existing errors. */
export type DiagnosticBaseline = Record<string, number>;

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

  return parseJestSummary(raw)
    ?? parseVitestSummary(raw)
    ?? parsePytestFailureSummary(raw)
    ?? parsePytestPassSummary(raw)
    ?? parseCargoSummary(raw)
    ?? parseGoSummary(raw);
}

function parseJestSummary(raw: string): TestSummary | null {
  const jestMatch = raw.match(/Tests:\s+(\d+)\s+failed,\s+(\d+)\s+passed(?:,\s+(\d+)\s+total)?/);
  if (!jestMatch) return null;
  return {
    framework: "jest",
    failed: parseInt(jestMatch[1]!, 10),
    passed: parseInt(jestMatch[2]!, 10),
    failureMessages: extractJestFailures(raw),
  };
}

function parseVitestSummary(raw: string): TestSummary | null {
  const vitestMatch = raw.match(/Tests\s+(\d+)\s+failed\s+\|\s+(\d+)\s+passed/);
  if (!vitestMatch) return null;
  return {
    framework: "vitest",
    failed: parseInt(vitestMatch[1]!, 10),
    passed: parseInt(vitestMatch[2]!, 10),
    failureMessages: extractJestFailures(raw),
  };
}

function parsePytestFailureSummary(raw: string): TestSummary | null {
  const pytestFailMatch = raw.match(/(\d+)\s+failed,\s+(\d+)\s+passed\s+in/);
  if (!pytestFailMatch) return null;
  return {
    framework: "pytest",
    failed: parseInt(pytestFailMatch[1]!, 10),
    passed: parseInt(pytestFailMatch[2]!, 10),
    failureMessages: extractPytestFailures(raw),
  };
}

function parsePytestPassSummary(raw: string): TestSummary | null {
  const pytestPassMatch = raw.match(/(\d+)\s+passed\s+in\s+[\d.]+s/);
  if (!pytestPassMatch) return null;
  return {
    framework: "pytest",
    failed: 0,
    passed: parseInt(pytestPassMatch[1]!, 10),
    failureMessages: [],
  };
}

function parseCargoSummary(raw: string): TestSummary | null {
  const cargoMatch = raw.match(/test result:\s+\w+\.\s+(\d+)\s+passed;\s+(\d+)\s+failed/);
  if (!cargoMatch) return null;
  return {
    framework: "cargo",
    passed: parseInt(cargoMatch[1]!, 10),
    failed: parseInt(cargoMatch[2]!, 10),
    failureMessages: extractCargoFailures(raw),
  };
}

function parseGoSummary(raw: string): TestSummary | null {
  const goFailMatch = raw.match(/^FAIL\s/m);
  if (!goFailMatch) return null;
  const goFailTests = raw.match(/--- FAIL: (\S+)/g) ?? [];
  const goPassTests = raw.match(/--- PASS: (\S+)/g) ?? [];
  return {
    framework: "go",
    failed: goFailTests.length,
    passed: goPassTests.length,
    failureMessages: goFailTests.map((m) => m.replace("--- FAIL: ", "")),
  };
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
   * Capture pre-edit diagnostic error counts for baseline comparison.
   * Call this BEFORE the edit, then pass the result to check() after.
   * Returns a map of file → error count.
   */
  async captureBaseline(
    files: ReadonlyArray<string>,
  ): Promise<DiagnosticBaseline> {
    const baseline: Record<string, number> = {};
    if (!this.diagnosticProvider) {
      for (const file of files) baseline[file] = 0;
      return baseline;
    }

    const entries = await Promise.all(
      files.map(async (file) => {
        try {
          const diagnostics = await this.diagnosticProvider!(file);
          return [file, diagnostics.filter((d) => d.severity === "error").length] as const;
        } catch {
          return [file, 0] as const;
        }
      }),
    );
    for (const [file, count] of entries) {
      baseline[file] = count;
    }
    return baseline;
  }

  /**
   * Run double-check validation on a file that was modified.
   * Returns structured results — the caller (TaskLoop) can feed
   * these back to the LLM for self-correction.
   *
   * If a baseline is provided, only errors exceeding the baseline count
   * are reported — pre-existing errors are filtered out.
   */
  async check(
    modifiedFiles: ReadonlyArray<string>,
    baseline?: DiagnosticBaseline,
  ): Promise<DoubleCheckResult> {
    if (!this.options.enabled) {
      return {
        passed: true,
        diagnosticErrors: [],
        testOutput: null,
        testPassed: null,
      };
    }

    const diagnostics = await this.runDiagnosticChecks(modifiedFiles, baseline);
    const test = await this.runConfiguredTests();

    const passed =
      diagnostics.errors.length === 0 &&
      (test.passed === null || test.passed);

    return {
      passed,
      diagnosticErrors: diagnostics.errors,
      testOutput: test.output,
      testPassed: test.passed,
      testSummary: test.summary,
      baselineFiltered: diagnostics.baselineFiltered > 0
        ? diagnostics.baselineFiltered
        : undefined,
    };
  }

  private async runDiagnosticChecks(
    modifiedFiles: ReadonlyArray<string>,
    baseline: DiagnosticBaseline | undefined,
  ): Promise<{ readonly errors: string[]; readonly baselineFiltered: number }> {
    if (!this.options.checkDiagnostics || !this.diagnosticProvider) {
      return { errors: [], baselineFiltered: 0 };
    }

    const errors: string[] = [];
    let baselineFiltered = 0;

    for (const file of modifiedFiles) {
      const result = await this.runDiagnosticCheckForFile(file, baseline);
      errors.push(...result.errors);
      baselineFiltered += result.baselineFiltered;
    }

    return { errors, baselineFiltered };
  }

  private async runDiagnosticCheckForFile(
    file: string,
    baseline: DiagnosticBaseline | undefined,
  ): Promise<{ readonly errors: string[]; readonly baselineFiltered: number }> {
    try {
      const diagnostics = await this.diagnosticProvider!(file);
      const errors = diagnostics.filter((d) => d.severity === "error");
      return filterDiagnosticErrors(file, errors, baseline);
    } catch (err) {
      const message = extractErrorMessage(err);
      this.bus.emit("error", {
        message: `Double-check diagnostics failed for ${file}: ${message}`,
        code: "DOUBLE_CHECK_DIAGNOSTIC_ERROR",
        fatal: false,
      });
      return {
        errors: [`${file}: diagnostics provider failure: ${message}`],
        baselineFiltered: 0,
      };
    }
  }

  private async runConfiguredTests(): Promise<{
    readonly output: string | null;
    readonly passed: boolean | null;
    readonly summary?: TestSummary;
  }> {
    if (!this.options.runTests || !this.options.testCommand || !this.testRunner) {
      return { output: null, passed: null };
    }

    try {
      const result = await this.testRunner(this.options.testCommand);
      return {
        output: result.output,
        passed: result.success,
        summary: parseTestOutput(result.output) ?? undefined,
      };
    } catch (err) {
      return {
        output: `Test execution failed: ${extractErrorMessage(err)}`,
        passed: false,
      };
    }
  }

  /**
   * Format double-check results as a string for LLM feedback.
   */
  formatResults(result: DoubleCheckResult): string {
    if (result.passed) {
      return formatPassedResult(result.baselineFiltered);
    }

    const parts = [
      formatDiagnosticFailures(result),
      formatTestFailures(result),
    ].filter((part): part is string => part !== null);

    return `Double-check: Validation FAILED.\n${parts.join("\n\n")}`;
  }
}

function filterDiagnosticErrors(
  file: string,
  errors: ReadonlyArray<{ readonly message: string; readonly severity: string }>,
  baseline: DiagnosticBaseline | undefined,
): { readonly errors: string[]; readonly baselineFiltered: number } {
  const preExistingCount = baseline?.[file] ?? 0;
  if (baseline && errors.length <= preExistingCount) {
    return { errors: [], baselineFiltered: errors.length };
  }

  const newErrors = baseline ? errors.slice(preExistingCount) : errors;
  return {
    errors: newErrors.map((err) => `${file}: ${err.message}`),
    baselineFiltered: baseline ? preExistingCount : 0,
  };
}

function formatPassedResult(baselineFiltered: number | undefined): string {
  if (baselineFiltered && baselineFiltered > 0) {
    return `Double-check: All validations passed (${baselineFiltered} pre-existing error(s) filtered).`;
  }
  return "Double-check: All validations passed.";
}

function formatDiagnosticFailures(result: DoubleCheckResult): string | null {
  if (result.diagnosticErrors.length === 0) return null;
  const header = formatDiagnosticFailureHeader(result);
  return `${header}:\n${result.diagnosticErrors.map((e) => `  - ${e}`).join("\n")}`;
}

function formatDiagnosticFailureHeader(result: DoubleCheckResult): string {
  const header = `Diagnostic errors (${result.diagnosticErrors.length})`;
  return result.baselineFiltered && result.baselineFiltered > 0
    ? `${header} — ${result.baselineFiltered} pre-existing error(s) filtered`
    : header;
}

function formatTestFailures(result: DoubleCheckResult): string | null {
  if (result.testPassed !== false) return null;
  if (!result.testSummary) {
    return `Test failures:\n${result.testOutput ?? "  (no output)"}`;
  }

  const summary = result.testSummary;
  const header = `Test failures (${summary.framework}): ${summary.failed} failed, ${summary.passed} passed`;
  return summary.failureMessages.length > 0
    ? `${header}\n${summary.failureMessages.map((m) => `  - ${m}`).join("\n")}`
    : header;
}
