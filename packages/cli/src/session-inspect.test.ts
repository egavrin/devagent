/**
 * Tests for session timeline inspection.
 */

import { describe, it, expect } from "vitest";
import { buildTimeline, renderTimeline } from "./session-inspect.js";
import type { LogEntry } from "@devagent/core";

describe("buildTimeline", () => {
  it("builds timeline from sample log entries", () => {
    const entries: LogEntry[] = [
      { ts: 1000, event: "iteration:start", sessionId: "s1", data: { iteration: 1, maxIterations: 10, estimatedTokens: 0, maxContextTokens: 0 } },
      { ts: 1100, event: "tool:before", sessionId: "s1", data: { name: "search_files", params: { pattern: "auth" }, callId: "c1" } },
      { ts: 1345, event: "tool:after", sessionId: "s1", data: { name: "search_files", callId: "c1", durationMs: 245, result: { success: true, output: "", error: null, artifacts: [] } } },
      { ts: 1400, event: "tool:before", sessionId: "s1", data: { name: "read_file", params: { path: "src/auth.ts" }, callId: "c2" } },
      { ts: 1520, event: "tool:after", sessionId: "s1", data: { name: "read_file", callId: "c2", durationMs: 120, result: { success: true, output: "", error: null, artifacts: [] } } },
      { ts: 4500, event: "plan:updated", sessionId: "s1", data: { steps: [{ description: "a", status: "completed" }, { description: "b", status: "completed" }, { description: "c", status: "pending" }, { description: "d", status: "pending" }], explanation: null } },
      { ts: 4600, event: "iteration:start", sessionId: "s1", data: { iteration: 2, maxIterations: 10, estimatedTokens: 0, maxContextTokens: 0 } },
    ];

    const timeline = buildTimeline(entries);

    expect(timeline.length).toBeGreaterThanOrEqual(5);

    // Check iteration entries
    const iterations = timeline.filter((t) => t.type === "iteration");
    expect(iterations.length).toBe(2);
    expect(iterations[0]!.label).toContain("Iteration 1");

    // Check tool entries with durations
    const tools = timeline.filter((t) => t.type === "tool");
    expect(tools.length).toBe(2);
    expect(tools[0]!.durationMs).toBe(245);
    expect(tools[1]!.durationMs).toBe(120);

    // Check plan entry
    const plans = timeline.filter((t) => t.type === "plan");
    expect(plans.length).toBe(1);
    expect(plans[0]!.label).toContain("2/4 completed");
  });

  it("computes tool duration from before/after pairs", () => {
    const entries: LogEntry[] = [
      { ts: 5000, event: "tool:before", sessionId: "s1", data: { name: "run_command", params: { command: "test" }, callId: "c3" } },
      { ts: 8200, event: "tool:after", sessionId: "s1", data: { name: "run_command", callId: "c3", durationMs: 3200, result: { success: true, output: "", error: null, artifacts: [] } } },
    ];

    const timeline = buildTimeline(entries);
    const tool = timeline.find((t) => t.type === "tool");
    expect(tool).toBeDefined();
    expect(tool!.durationMs).toBe(3200);
  });

  it("handles context:compacted events", () => {
    const entries: LogEntry[] = [
      { ts: 5100, event: "context:compacted", sessionId: "s1", data: { tokensBefore: 45000, estimatedTokens: 32000, removedCount: 3 } },
    ];

    const timeline = buildTimeline(entries);
    expect(timeline.length).toBe(1);
    expect(timeline[0]!.type).toBe("compaction");
    expect(timeline[0]!.label).toContain("45k");
    expect(timeline[0]!.label).toContain("32k");
  });

  it("handles error events", () => {
    const entries: LogEntry[] = [
      { ts: 6000, event: "error", sessionId: "s1", data: { message: "Rate limit exceeded", code: "429", fatal: false } },
    ];

    const timeline = buildTimeline(entries);
    expect(timeline.length).toBe(1);
    expect(timeline[0]!.type).toBe("error");
    expect(timeline[0]!.label).toContain("Rate limit");
  });

  it("returns empty array for empty input", () => {
    expect(buildTimeline([])).toEqual([]);
  });
});

describe("renderTimeline", () => {
  it("renders timeline with relative timestamps", () => {
    const timeline = [
      { timestamp: 1000, type: "iteration" as const, label: "Iteration 1" },
      { timestamp: 1100, type: "tool" as const, label: "search_files", durationMs: 245 },
      { timestamp: 4600, type: "iteration" as const, label: "Iteration 2" },
    ];

    const output = renderTimeline(timeline);
    expect(output).toContain("+0.0s");
    expect(output).toContain("Iteration 1");
    expect(output).toContain("search_files");
    expect(output).toContain("245ms");
    expect(output).toContain("+3.6s");
    expect(output).toContain("Iteration 2");
  });

  it("returns message for empty timeline", () => {
    expect(renderTimeline([])).toContain("No events");
  });
});
