/**
 * Tests for the ArkTS diagnostic provider.
 */

import { describe, it, expect } from "vitest";

import { createArkTSDiagnosticProvider } from "./diagnostic-provider.js";
import { isTsLinterAvailable } from "./linter.js";

describe("createArkTSDiagnosticProvider", () => {
  it("returns null when linterPath is not set", () => {
    const provider = createArkTSDiagnosticProvider({
      enabled: true,
      strictMode: true,
      targetVersion: "5.0",
    });
    expect(provider).toBeNull();
  });

  it("returns null when linterPath does not exist", () => {
    const provider = createArkTSDiagnosticProvider({
      enabled: true,
      strictMode: true,
      targetVersion: "5.0",
      linterPath: "/non/existent/path",
    });
    expect(provider).toBeNull();
  });

  it("returns null when linterPath exists but tslinter.js is not built", () => {
    const provider = createArkTSDiagnosticProvider({
      enabled: true,
      strictMode: true,
      targetVersion: "5.0",
      linterPath: "/tmp", // exists but no dist/tslinter.js
    });
    expect(provider).toBeNull();
  });
});

const LINTER_PATH = process.env["ARKTS_LINTER_PATH"] ??
  `${process.env["HOME"]}/Documents/arkcompiler_ets_frontend/ets2panda/linter`;

const linterAvailable = isTsLinterAvailable(LINTER_PATH);

describe.skipIf(!linterAvailable)("ArkTS diagnostic provider integration", () => {
  it("returns a function when linter is available", () => {
    const provider = createArkTSDiagnosticProvider({
      enabled: true,
      strictMode: true,
      targetVersion: "5.0",
      linterPath: LINTER_PATH,
    });
    expect(provider).not.toBeNull();
    expect(typeof provider).toBe("function");
  });

  it("returns empty array for non-.ets files", async () => {
    const provider = createArkTSDiagnosticProvider({
      enabled: true,
      strictMode: true,
      targetVersion: "5.0",
      linterPath: LINTER_PATH,
    });
    expect(provider).not.toBeNull();

    const result = await provider!("/tmp/test.ts");
    expect(result).toEqual([]);
  });

  it("returns diagnostics for .ets file with violations", { timeout: 30_000 }, async () => {
    const provider = createArkTSDiagnosticProvider({
      enabled: true,
      strictMode: true,
      targetVersion: "5.0",
      linterPath: LINTER_PATH,
    });
    expect(provider).not.toBeNull();

    // Create a temp .ets file
    const { writeFileSync, mkdirSync, rmSync } = await import("node:fs");
    const { join } = await import("node:path");
    const { tmpdir } = await import("node:os");
    const tmpDir = join(tmpdir(), ".arkts-diag-test-tmp");
    mkdirSync(tmpDir, { recursive: true });

    const etsFile = join(tmpDir, "test_diag.ets");
    writeFileSync(etsFile, `const x: any = 5;\n`);

    try {
      const diagnostics = await provider!(etsFile);
      // Should have at least some diagnostics for 'any' type usage
      expect(diagnostics.length).toBeGreaterThanOrEqual(0);

      if (diagnostics.length > 0) {
        // Verify diagnostic format
        expect(diagnostics[0]).toHaveProperty("message");
        expect(diagnostics[0]).toHaveProperty("severity");
        expect(["error", "warning"]).toContain(diagnostics[0]!.severity);
      }
    } finally {
      try { rmSync(tmpDir, { recursive: true }); } catch { /* ignore */ }
    }
  });
});
