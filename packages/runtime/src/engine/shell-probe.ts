/**
 * Shell tool availability probe — detects which common CLI tools
 * are available in the execution environment at session start.
 *
 * Results are stored as SessionState env facts so the LLM
 * knows which tools to use (e.g., `rg` vs `grep`, `fd` vs `find`).
 */

import { execSync } from "node:child_process";

// ─── Types ──────────────────────────────────────────────────

export interface ShellProbeResult {
  readonly tool: string;
  readonly available: boolean;
  readonly version?: string;
}

// ─── Constants ──────────────────────────────────────────────

/** Tools to probe at session start. */
const PROBE_TOOLS: ReadonlyArray<{ name: string; versionFlag: string }> = [
  { name: "rg", versionFlag: "--version" },
  { name: "fd", versionFlag: "--version" },
  { name: "jq", versionFlag: "--version" },
  { name: "gh", versionFlag: "--version" },
  { name: "python3", versionFlag: "--version" },
  { name: "node", versionFlag: "--version" },
];

// ─── Probe ──────────────────────────────────────────────────

/**
 * Probe for available shell tools. Returns results for all probed tools.
 * Fast: uses `which` for existence check, version flag for version string.
 */
export function probeShellTools(): ReadonlyArray<ShellProbeResult> {
  const results: ShellProbeResult[] = [];

  for (const { name, versionFlag } of PROBE_TOOLS) {
    try {
      const versionOutput = execSync(`${name} ${versionFlag} 2>/dev/null`, {
        timeout: 2000,
        encoding: "utf-8",
      }).trim();
      const firstLine = versionOutput.split("\n")[0] ?? "";
      results.push({
        tool: name,
        available: true,
        version: firstLine.slice(0, 80),
      });
    } catch {
      results.push({ tool: name, available: false });
    }
  }

  return results;
}

/**
 * Format probe results as a compact env fact string for SessionState.
 */
export function formatProbeResults(results: ReadonlyArray<ShellProbeResult>): string {
  const available = results.filter((r) => r.available).map((r) => r.tool);
  const missing = results.filter((r) => !r.available).map((r) => r.tool);

  const parts: string[] = [];
  if (available.length > 0) {
    parts.push(`Available CLI tools: ${available.join(", ")}`);
  }
  if (missing.length > 0) {
    parts.push(`Not installed: ${missing.join(", ")} — use builtin tools or alternatives`);
  }
  return parts.join(". ");
}
