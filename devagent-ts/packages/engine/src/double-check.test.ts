import { describe, it, expect, beforeEach } from "vitest";
import { DoubleCheck, DEFAULT_DOUBLE_CHECK_OPTIONS } from "./double-check.js";
import type { DiagnosticProvider, TestRunner } from "./double-check.js";
import { EventBus } from "@devagent/core";

describe("DoubleCheck", () => {
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

  it("handles diagnostic provider errors gracefully", async () => {
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
    // Should not fail — just report the error via bus
    expect(result.diagnosticErrors.length).toBe(0);
    expect(errors.length).toBe(1);
    expect(errors[0]).toContain("LSP crashed");
  });
});
