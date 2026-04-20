import { describe, it, expect, vi } from "vitest";

import { probeShellTools, formatProbeResults } from "./shell-probe.js";
import type { ShellProbeResult } from "./shell-probe.js";

describe("probeShellTools", () => {
  it("returns an array of probe results", () => {
    const results = probeShellTools();

    expect(Array.isArray(results)).toBe(true);
    expect(results.length).toBeGreaterThan(0);

    for (const r of results) {
      expect(r).toHaveProperty("tool");
      expect(r).toHaveProperty("available");
      expect(typeof r.tool).toBe("string");
      expect(typeof r.available).toBe("boolean");
      if (r.available) {
        expect(r.version).toBeDefined();
        expect(typeof r.version).toBe("string");
      }
    }
  });
});

describe("formatProbeResults", () => {
  it("formats available and missing tools correctly", () => {
    const results: ShellProbeResult[] = [
      { tool: "rg", available: true, version: "ripgrep 14.0.0" },
      { tool: "fd", available: true, version: "fd 9.0.0" },
      { tool: "jq", available: false },
    ];

    const formatted = formatProbeResults(results);
    expect(formatted).toContain("Available CLI tools: rg, fd");
    expect(formatted).toContain("Not installed: jq");
  });

  it("handles all available", () => {
    const results: ShellProbeResult[] = [
      { tool: "rg", available: true, version: "ripgrep 14.0.0" },
      { tool: "node", available: true, version: "v20.0.0" },
    ];

    const formatted = formatProbeResults(results);
    expect(formatted).toContain("Available CLI tools: rg, node");
    expect(formatted).not.toContain("Not installed");
  });

  it("handles all missing", () => {
    const results: ShellProbeResult[] = [
      { tool: "rg", available: false },
      { tool: "fd", available: false },
    ];

    const formatted = formatProbeResults(results);
    expect(formatted).not.toContain("Available CLI tools");
    expect(formatted).toContain("Not installed: rg, fd");
  });
});
