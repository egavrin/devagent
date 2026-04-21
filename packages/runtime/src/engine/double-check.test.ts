import { describe, it, expect, beforeEach } from "vitest";

import { DoubleCheck, DEFAULT_DOUBLE_CHECK_OPTIONS, parseTestOutput } from "./double-check.js";
import type { DiagnosticProvider, TestRunner } from "./double-check.js";
import { EventBus } from "../core/index.js";
let bus: EventBus;

beforeEach(() => {
  bus = new EventBus();
});

it("passes when disabled", async () => {
  const dc = new DoubleCheck(DEFAULT_DOUBLE_CHECK_OPTIONS, bus);
  expect(dc.isEnabled()).toBe(false);

  const result = await dc.check(["file.ts"]);
  expect(result.passed).toBe(true);
  expect(result.diagnosticErrors.length).toBe(0);
});

it("passes when no errors found", async () => {
  const dc = new DoubleCheck(
    { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
    bus,
  );

  const provider: DiagnosticProvider = async () => [];
  dc.setDiagnosticProvider(provider);

  const result = await dc.check(["file.ts"]);
  expect(result.passed).toBe(true);
  expect(result.diagnosticErrors.length).toBe(0);
});

it("fails when diagnostic errors found", async () => {
  const dc = new DoubleCheck(
    { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
    bus,
  );

  const provider: DiagnosticProvider = async () => [
    { message: "Type 'string' is not assignable to type 'number'", severity: "error" },
  ];
  dc.setDiagnosticProvider(provider);

  const result = await dc.check(["file.ts"]);
  expect(result.passed).toBe(false);
  expect(result.diagnosticErrors.length).toBe(1);
  expect(result.diagnosticErrors[0]).toContain("not assignable");
});

it("ignores warnings", async () => {
  const dc = new DoubleCheck(
    { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
    bus,
  );

  const provider: DiagnosticProvider = async () => [
    { message: "Unused variable", severity: "warning" },
  ];
  dc.setDiagnosticProvider(provider);

  const result = await dc.check(["file.ts"]);
  expect(result.passed).toBe(true);
});

it("checks multiple files", async () => {
  const dc = new DoubleCheck(
    { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
    bus,
  );

  const provider: DiagnosticProvider = async (file) => {
    if (file === "bad.ts") {
      return [{ message: "Error in bad.ts", severity: "error" }];
    }
    return [];
  };
  dc.setDiagnosticProvider(provider);

  const result = await dc.check(["good.ts", "bad.ts"]);
  expect(result.passed).toBe(false);
  expect(result.diagnosticErrors.length).toBe(1);
  expect(result.diagnosticErrors[0]).toContain("bad.ts");
});

it("runs tests when configured", async () => {
  const dc = new DoubleCheck(
    {
      ...DEFAULT_DOUBLE_CHECK_OPTIONS,
      enabled: true,
      runTests: true,
      testCommand: "bun test",
    },
    bus,
  );

  const runner: TestRunner = async () => ({
    success: true,
    output: "All tests passed",
  });
  dc.setTestRunner(runner);

  const result = await dc.check([]);
  expect(result.passed).toBe(true);
  expect(result.testPassed).toBe(true);
  expect(result.testOutput).toBe("All tests passed");
});

it("fails when tests fail", async () => {
  const dc = new DoubleCheck(
    {
      ...DEFAULT_DOUBLE_CHECK_OPTIONS,
      enabled: true,
      runTests: true,
      testCommand: "bun test",
    },
    bus,
  );

  const runner: TestRunner = async () => ({
    success: false,
    output: "1 test failed",
  });
  dc.setTestRunner(runner);

  const result = await dc.check([]);
  expect(result.passed).toBe(false);
  expect(result.testPassed).toBe(false);
});

it("formats passing results", () => {
  const dc = new DoubleCheck(DEFAULT_DOUBLE_CHECK_OPTIONS, bus);
  const formatted = dc.formatResults({
    passed: true,
    diagnosticErrors: [],
    testOutput: null,
    testPassed: null,
  });
  expect(formatted).toContain("passed");
});

it("formats failing results with diagnostics and test output", () => {
  const dc = new DoubleCheck(DEFAULT_DOUBLE_CHECK_OPTIONS, bus);
  const formatted = dc.formatResults({
    passed: false,
    diagnosticErrors: ["file.ts: some error"],
    testOutput: "FAIL: test1",
    testPassed: false,
  });
  expect(formatted).toContain("FAILED");
  expect(formatted).toContain("some error");
  expect(formatted).toContain("FAIL: test1");
});

it("fails closed when diagnostic provider errors", async () => {
  const errors: string[] = [];
  bus.on("error", (e) => errors.push(e.message));

  const dc = new DoubleCheck(
    { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
    bus,
  );

  const provider: DiagnosticProvider = async () => {
    throw new Error("LSP crashed");
  };
  dc.setDiagnosticProvider(provider);

  const result = await dc.check(["file.ts"]);
  expect(result.passed).toBe(false);
  expect(result.diagnosticErrors.length).toBe(1);
  expect(result.diagnosticErrors[0]).toContain("LSP crashed");
  expect(errors.length).toBe(1);
  expect(errors[0]).toContain("LSP crashed");
});

it("includes testSummary when test output is parseable", async () => {
  const dc = new DoubleCheck(
    {
      ...DEFAULT_DOUBLE_CHECK_OPTIONS,
      enabled: true,
      runTests: true,
      testCommand: "bun test",
    },
    bus,
  );

  const runner: TestRunner = async () => ({
    success: false,
    output: "Tests:  2 failed, 10 passed, 12 total",
  });
  dc.setTestRunner(runner);

  const result = await dc.check([]);
  expect(result.testPassed).toBe(false);
  expect(result.testSummary).toBeDefined();
  expect(result.testSummary!.framework).toBe("jest");
  expect(result.testSummary!.failed).toBe(2);
  expect(result.testSummary!.passed).toBe(10);
});

it("formats results with testSummary", () => {
  const dc = new DoubleCheck(DEFAULT_DOUBLE_CHECK_OPTIONS, bus);
  const formatted = dc.formatResults({
    passed: false,
    diagnosticErrors: [],
    testOutput: "raw output",
    testPassed: false,
    testSummary: {
      framework: "jest",
      passed: 10,
      failed: 2,
      failureMessages: ["test one failed", "test two failed"],
    },
  });
  expect(formatted).toContain("jest");
  expect(formatted).toContain("2 failed");
  expect(formatted).toContain("10 passed");
  expect(formatted).toContain("test one failed");
  expect(formatted).not.toContain("raw output"); // Structured summary preferred
});


describe("DoubleCheck baseline filtering", () => {
  let bus: EventBus;

  beforeEach(() => {
    bus = new EventBus();
  });

  it("captureBaseline collects pre-edit error counts per file", async () => {
    const dc = new DoubleCheck(
      { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
      bus,
    );

    const provider: DiagnosticProvider = async (file) => {
      if (file === "dirty.ts") {
        return [
          { message: "pre-existing error 1", severity: "error" },
          { message: "pre-existing error 2", severity: "error" },
          { message: "a warning", severity: "warning" },
        ];
      }
      return [];
    };
    dc.setDiagnosticProvider(provider);

    const baseline = await dc.captureBaseline(["dirty.ts", "clean.ts"]);
    expect(baseline).toEqual({ "dirty.ts": 2, "clean.ts": 0 });
  });

  it("check with baseline filters pre-existing errors", async () => {
    const dc = new DoubleCheck(
      { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
      bus,
    );

    // File had 3 pre-existing errors before edit, still has same 3 after
    const provider: DiagnosticProvider = async () => [
      { message: "pre-existing error 1", severity: "error" },
      { message: "pre-existing error 2", severity: "error" },
      { message: "pre-existing error 3", severity: "error" },
    ];
    dc.setDiagnosticProvider(provider);

    const baseline = { "file.ts": 3 };
    const result = await dc.check(["file.ts"], baseline);
    // All 3 errors existed before the edit — should pass
    expect(result.passed).toBe(true);
    expect(result.diagnosticErrors.length).toBe(0);
  });

  it("check with baseline detects genuinely new errors", async () => {
    const dc = new DoubleCheck(
      { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
      bus,
    );

    // File had 2 errors before, now has 4 (2 new ones introduced)
    const provider: DiagnosticProvider = async () => [
      { message: "pre-existing error 1", severity: "error" },
      { message: "pre-existing error 2", severity: "error" },
      { message: "NEW error 3", severity: "error" },
      { message: "NEW error 4", severity: "error" },
    ];
    dc.setDiagnosticProvider(provider);

    const baseline = { "file.ts": 2 };
    const result = await dc.check(["file.ts"], baseline);
    // 2 errors are pre-existing, but 2 are new — should fail
    expect(result.passed).toBe(false);
    // Only report the 2 NEW errors, not the 2 pre-existing ones
    expect(result.diagnosticErrors.length).toBe(2);
  });

  it("check without baseline reports all errors (backward compatible)", async () => {
    const dc = new DoubleCheck(
      { ...DEFAULT_DOUBLE_CHECK_OPTIONS, enabled: true },
      bus,
    );

    const provider: DiagnosticProvider = async () => [
      { message: "error 1", severity: "error" },
      { message: "error 2", severity: "error" },
    ];
    dc.setDiagnosticProvider(provider);

    const result = await dc.check(["file.ts"]);
    // No baseline → report all errors (backward compatible)
    expect(result.passed).toBe(false);
    expect(result.diagnosticErrors.length).toBe(2);
  });

  it("formats results indicating pre-existing errors were filtered", () => {
    const dc = new DoubleCheck(DEFAULT_DOUBLE_CHECK_OPTIONS, bus);
    const formatted = dc.formatResults({
      passed: false,
      diagnosticErrors: ["file.ts: NEW error"],
      testOutput: null,
      testPassed: null,
      baselineFiltered: 3,
    });
    expect(formatted).toContain("FAILED");
    expect(formatted).toContain("NEW error");
    expect(formatted).toContain("3 pre-existing");
  });
});

describe("parseTestOutput", () => {
  it("returns null for empty input", () => {
    expect(parseTestOutput("")).toBeNull();
    expect(parseTestOutput("hi")).toBeNull();
  });

  it("parses Jest output", () => {
    const output = `
FAIL src/foo.test.ts
  ● test suite > should work
    Expected: 5
    Received: 3

Tests:  1 failed, 5 passed, 6 total
`;
    const result = parseTestOutput(output);
    expect(result).not.toBeNull();
    expect(result!.framework).toBe("jest");
    expect(result!.failed).toBe(1);
    expect(result!.passed).toBe(5);
    expect(result!.failureMessages.length).toBeGreaterThanOrEqual(1);
  });

  it("parses Vitest output", () => {
    const output = `
 ❯ src/foo.test.ts (3 tests | 1 failed)
   ✕ should compute correctly

Tests  1 failed | 2 passed (3)
`;
    const result = parseTestOutput(output);
    expect(result).not.toBeNull();
    expect(result!.framework).toBe("vitest");
    expect(result!.failed).toBe(1);
    expect(result!.passed).toBe(2);
  });

  it("parses pytest output", () => {
    const output = `
FAILED tests/test_algo.py::test_edge_case - AssertionError
FAILED tests/test_algo.py::test_overflow - TypeError
1 failed, 8 passed in 0.5s
`;
    const result = parseTestOutput(output);
    expect(result).not.toBeNull();
    expect(result!.framework).toBe("pytest");
    expect(result!.failed).toBe(1);
    expect(result!.passed).toBe(8);
    expect(result!.failureMessages.length).toBe(2);
  });

  it("parses pytest all-pass output", () => {
    const output = "10 passed in 1.2s";
    const result = parseTestOutput(output);
    expect(result).not.toBeNull();
    expect(result!.framework).toBe("pytest");
    expect(result!.failed).toBe(0);
    expect(result!.passed).toBe(10);
  });

  it("parses cargo test output", () => {
    const output = `
running 5 tests
test utils::test_parse ... ok
test utils::test_format ... ok
test utils::test_edge ... FAILED
test utils::test_overflow ... ok
test utils::test_basic ... ok

test result: FAILED. 4 passed; 1 failed; 0 ignored; 0 measured; 0 filtered out
`;
    const result = parseTestOutput(output);
    expect(result).not.toBeNull();
    expect(result!.framework).toBe("cargo");
    expect(result!.passed).toBe(4);
    expect(result!.failed).toBe(1);
    expect(result!.failureMessages).toContain("utils::test_edge");
  });

  it("parses Go test output", () => {
    const output = `
--- PASS: TestBasic (0.00s)
--- FAIL: TestEdge (0.01s)
    main_test.go:42: expected 5, got 3
FAIL
`;
    const result = parseTestOutput(output);
    expect(result).not.toBeNull();
    expect(result!.framework).toBe("go");
    expect(result!.passed).toBe(1);
    expect(result!.failed).toBe(1);
    expect(result!.failureMessages).toContain("TestEdge");
  });
});
