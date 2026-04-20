/**
 * Tests for tslinter output types and parsing.
 */

import { describe, it, expect } from "vitest";

import { parseTsLinterLine, mapSeverity, ProblemSeverity } from "./rules.js";

describe("mapSeverity", () => {
  // ── Numeric severity (IDE-interactive mode) ──
  it("maps numeric 2 (ERROR) to error", () => {
    expect(mapSeverity(2)).toBe("error");
    expect(mapSeverity(ProblemSeverity.ERROR)).toBe("error");
  });

  it("maps numeric 1 (WARNING) to warning", () => {
    expect(mapSeverity(1)).toBe("warning");
    expect(mapSeverity(ProblemSeverity.WARNING)).toBe("warning");
  });

  it("maps numeric 0 to warning", () => {
    expect(mapSeverity(0)).toBe("warning");
  });

  it("maps numeric > 2 to error", () => {
    expect(mapSeverity(3)).toBe("error");
  });

  // ── String severity (backwards compatibility) ──
  it("maps string ERROR to error", () => {
    expect(mapSeverity("ERROR")).toBe("error");
  });

  it("maps string WARNING to warning", () => {
    expect(mapSeverity("WARNING")).toBe("warning");
  });

  it("handles lowercase strings", () => {
    expect(mapSeverity("error")).toBe("error");
    expect(mapSeverity("warning")).toBe("warning");
  });

  it("treats unknown string severity as warning", () => {
    expect(mapSeverity("INFO")).toBe("warning");
  });
});

describe("parseTsLinterLine", () => {
  it("parses a valid problem line with numeric severity", () => {
    // Matches actual IDE-interactive output format
    const line = JSON.stringify({
      filePath: "/tmp/test.ets",
      problems: [
        {
          line: 3,
          column: 9,
          endLine: 3,
          endColumn: 19,
          start: 43,
          end: 53,
          type: "Identifier",
          severity: 2,
          faultId: 78,
          problem: "GlobalThisError",
          suggest: "",
          rule: '"globalThis" is not supported (arkts-no-globalthis)',
          ruleTag: 137,
          autofixable: false,
        },
      ],
    });

    const result = parseTsLinterLine(line);
    expect(result).not.toBeNull();
    expect(result!.filePath).toBe("/tmp/test.ets");
    expect(result!.problems).toHaveLength(1);
    expect(result!.problems[0]!.rule).toBe('"globalThis" is not supported (arkts-no-globalthis)');
    expect(result!.problems[0]!.severity).toBe(2); // numeric ERROR
    expect(result!.problems[0]!.line).toBe(3);
    expect(result!.problems[0]!.column).toBe(9);
    expect(result!.problems[0]!.faultId).toBe(78);
    expect(result!.problems[0]!.type).toBe("Identifier");
    expect(result!.problems[0]!.ruleTag).toBe(137);
  });

  it("parses a line with multiple problems", () => {
    const line = JSON.stringify({
      filePath: "/tmp/test.ets",
      problems: [
        {
          line: 1, column: 1, endLine: 1, endColumn: 5,
          problem: "AnyType", suggest: "", rule: "arkts-no-any", severity: 2,
        },
        {
          line: 3, column: 1, endLine: 3, endColumn: 10,
          problem: "SymbolType", suggest: "", rule: "arkts-no-symbol", severity: 1,
        },
      ],
    });

    const result = parseTsLinterLine(line);
    expect(result).not.toBeNull();
    expect(result!.problems).toHaveLength(2);
    expect(result!.problems[0]!.severity).toBe(2); // ERROR
    expect(result!.problems[1]!.severity).toBe(1); // WARNING
  });

  it("parses a line with autofix info", () => {
    const line = JSON.stringify({
      filePath: "/tmp/test.ets",
      problems: [
        {
          line: 1, column: 1, endLine: 1, endColumn: 5,
          problem: "AnyType", suggest: "Replace with Object",
          rule: "arkts-no-any", severity: 2,
          autofixable: true,
          autofix: [{ replacementText: "Object", line: 1, column: 1, endLine: 1, endColumn: 5 }],
        },
      ],
    });

    const result = parseTsLinterLine(line);
    expect(result).not.toBeNull();
    expect(result!.problems[0]!.autofixable).toBe(true);
    expect(result!.problems[0]!.autofix).toHaveLength(1);
    expect(result!.problems[0]!.autofix![0]!.replacementText).toBe("Object");
  });

  it("returns null for empty line", () => {
    expect(parseTsLinterLine("")).toBeNull();
    expect(parseTsLinterLine("  ")).toBeNull();
  });

  it("returns null for control messages", () => {
    const control = '{"content":"report finish","messageType":1,"indictor":1}';
    expect(parseTsLinterLine(control)).toBeNull();
  });

  it("returns null for invalid JSON", () => {
    expect(parseTsLinterLine("not json")).toBeNull();
  });

  it("returns null for JSON without expected shape", () => {
    expect(parseTsLinterLine('{"foo": "bar"}')).toBeNull();
    expect(parseTsLinterLine('{"filePath": 42}')).toBeNull();
  });
});
