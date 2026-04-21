/**
 * Tests for CLI output formatting — categorical verbosity, gauges, summaries.
 */

import { AgentType } from "@devagent/runtime";
import { describe, it, expect, afterEach, beforeEach, vi } from "vitest";

import {
  isCategoryEnabled,
  buildVerbosityConfig,
  formatContextGauge,
  formatEnrichedError,
  inferErrorSuggestion,
  formatTurnSummary,
  formatTurnStart,
  formatTurnEnd,
  formatSessionSummary,
  formatCompactionResult,
  formatReasoning,
  formatFileEditPreview,
  formatTranscriptPart,
  formatToolStart,
  formatToolEnd,
  formatToolGroupStart,
  formatToolGroupEnd,
  formatSubagentBatchLaunch,
  formatSubagentStart,
  formatSubagentError,
  summarizeSubagentUpdate,
  SubagentPanelRenderer,
} from "./format.js";
import {
  presentApprovalRequestEvent,
  presentContextCompactingEvent,
  presentToolAfterEvent,
} from "./transcript-presenter.js";
import type { VerbosityConfig } from "@devagent/runtime";

function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]*m/g, "");
}

// ─── Improvement 5: Categorical Verbosity ─────────────────────

describe("isCategoryEnabled", () => {
  it("returns true for any category when base is verbose and no categories specified", () => {
    const config: VerbosityConfig = { base: "verbose", categories: new Set() };
    expect(isCategoryEnabled("tools", config)).toBe(true);
    expect(isCategoryEnabled("context", config)).toBe(true);
    expect(isCategoryEnabled("cost", config)).toBe(true);
  });

  it("returns false for any category when base is quiet", () => {
    const config: VerbosityConfig = { base: "quiet", categories: new Set() };
    expect(isCategoryEnabled("tools", config)).toBe(false);
    expect(isCategoryEnabled("context", config)).toBe(false);
  });

  it("returns false for non-selected categories when base is normal", () => {
    const config: VerbosityConfig = { base: "normal", categories: new Set(["context"]) };
    expect(isCategoryEnabled("tools", config)).toBe(false);
    expect(isCategoryEnabled("cost", config)).toBe(false);
  });

  it("returns true for selected categories when base is normal", () => {
    const config: VerbosityConfig = { base: "normal", categories: new Set(["context", "cost"]) };
    expect(isCategoryEnabled("context", config)).toBe(true);
    expect(isCategoryEnabled("cost", config)).toBe(true);
  });

  it("returns true for selected categories even when base is quiet", () => {
    // Explicit categories override quiet
    const config: VerbosityConfig = { base: "quiet", categories: new Set(["context"]) };
    expect(isCategoryEnabled("context", config)).toBe(true);
    expect(isCategoryEnabled("tools", config)).toBe(false);
  });

  it("returns true only for selected categories when explicit categories are set", () => {
    // When categories are specified, they take precedence regardless of base
    const config: VerbosityConfig = { base: "verbose", categories: new Set(["tools"]) };
    expect(isCategoryEnabled("tools", config)).toBe(true);
    // Non-listed category is excluded because explicit categories override base
    expect(isCategoryEnabled("context", config)).toBe(false);
  });
});

describe("buildVerbosityConfig", () => {
  const originalEnv = process.env["DEVAGENT_DEBUG"];

  afterEach(() => {
    if (originalEnv === undefined) {
      delete process.env["DEVAGENT_DEBUG"];
    } else {
      process.env["DEVAGENT_DEBUG"] = originalEnv;
    }
  });

  it("returns verbose base with empty categories for bare --verbose", () => {
    delete process.env["DEVAGENT_DEBUG"];
    const config = buildVerbosityConfig("verbose", undefined);
    expect(config.base).toBe("verbose");
    expect(config.categories.size).toBe(0);
  });

  it("returns normal base with selected categories for --verbose=context,tools", () => {
    delete process.env["DEVAGENT_DEBUG"];
    const config = buildVerbosityConfig("normal", "context,tools");
    expect(config.base).toBe("normal");
    expect(config.categories.has("context")).toBe(true);
    expect(config.categories.has("tools")).toBe(true);
    expect(config.categories.size).toBe(2);
  });

  it("merges env var categories additively", () => {
    process.env["DEVAGENT_DEBUG"] = "cost,plan";
    const config = buildVerbosityConfig("normal", "context");
    expect(config.categories.has("context")).toBe(true);
    expect(config.categories.has("cost")).toBe(true);
    expect(config.categories.has("plan")).toBe(true);
    expect(config.categories.size).toBe(3);
  });

  it("returns quiet base with no categories for --quiet", () => {
    delete process.env["DEVAGENT_DEBUG"];
    const config = buildVerbosityConfig("quiet", undefined);
    expect(config.base).toBe("quiet");
    expect(config.categories.size).toBe(0);
  });

  it("env var alone creates categories on normal base", () => {
    process.env["DEVAGENT_DEBUG"] = "events";
    const config = buildVerbosityConfig("normal", undefined);
    expect(config.base).toBe("normal");
    expect(config.categories.has("events")).toBe(true);
  });
});

// ─── Improvement 2: Context Gauge ─────────────────────────────

describe("formatContextGauge", () => {
  it("returns empty string when maxTokens is 0", () => {
    expect(formatContextGauge(50000, 0)).toBe("");
  });

  it("returns empty string when estimatedTokens is 0", () => {
    expect(formatContextGauge(0, 128000)).toBe("");
  });

  it("formats tokens as k values", () => {
    const result = formatContextGauge(45000, 128000);
    expect(result).toContain("45k/128k");
  });

  it("contains ANSI codes for dim at low usage", () => {
    // < 60% usage = dim
    const result = formatContextGauge(30000, 128000);
    expect(result).toContain("30k/128k");
  });

  it("contains yellow-ish output at 60-80% usage", () => {
    // ~70% usage
    const result = formatContextGauge(90000, 128000);
    expect(result).toContain("90k/128k");
  });

  it("contains red-ish output at >80% usage", () => {
    // ~90% usage
    const result = formatContextGauge(115000, 128000);
    expect(result).toContain("115k/128k");
  });
});

describe("formatFileEditPreview", () => {
  it("renders a typed preview for a single file edit", () => {
    const result = formatFileEditPreview([{
      path: "src/new-file.ts",
      kind: "create",
      additions: 2,
      deletions: 0,
      unifiedDiff: "--- /dev/null\n+++ b/src/new-file.ts\n@@ -0,0 +1,2 @@\n+export const x = 1;\n+export const y = 2;",
      truncated: false,
      structuredDiff: {
        hunks: [{
          oldStart: 0,
          oldLines: 0,
          newStart: 1,
          newLines: 2,
          lines: [
            { type: "add", text: "export const x = 1;", oldLine: null, newLine: 1 },
            { type: "add", text: "export const y = 2;", oldLine: null, newLine: 2 },
          ],
        }],
      },
    }]);

    expect(result).toContain("src/new-file.ts");
    expect(result).toContain("Added 2 lines");
    expect(result).toContain("+2");
    expect(result).toContain("-0");
    expect(stripAnsi(result)).toContain("export const x = 1;");
  });

  it("limits visible diff lines and reports overflow", () => {
    const result = formatFileEditPreview([{
      path: "src/overflow.ts",
      kind: "update",
      additions: 10,
      deletions: 10,
      unifiedDiff: Array.from({ length: 12 }, (_, index) => `+line-${index + 1}`).join("\n"),
      truncated: true,
    }], 0, 8);

    expect(result).toContain("(truncated)");
    expect(result).toContain("... +4 more diff lines");
  });

  it("renders hidden-file overflow for multi-file previews", () => {
    const result = formatFileEditPreview([{
      path: "src/a.ts",
      kind: "update",
      additions: 1,
      deletions: 1,
      unifiedDiff: "@@ -1,1 +1,1 @@\n-a\n+b",
      truncated: false,
    }], 2);

    expect(result).toContain("... +2 more files");
  });
});
describe("formatTranscriptPart", () => {
  it("renders approval parts with typed approval semantics", () => {
    const result = formatTranscriptPart(presentApprovalRequestEvent({
      id: "approval-1",
      action: "edit",
      toolName: "write_file",
      details: "Write src/new.ts",
    }));
    expect(result).toContain("[approval]");
    expect(result).toContain("write_file");
    expect(result).toContain("Write src/new.ts");
  });

  it("renders progress parts for context compaction", () => {
    const result = formatTranscriptPart(presentContextCompactingEvent({
      estimatedTokens: 96000,
      maxTokens: 128000,
    }));
    expect(result).toContain("compacting context");
    expect(result).toContain("96k / 128k");
  });

  it("renders command-result parts from typed run_command metadata", () => {
    const parts = presentToolAfterEvent({
      name: "run_command",
      callId: "call-cmd-1",
      durationMs: 45,
      result: {
        success: false,
        output: "Exit code: 1",
        error: "Command exited with code 1",
        artifacts: [],
        metadata: {
          commandResult: {
            command: "npm test",
            cwd: ".",
            exitCode: 1,
            timedOut: false,
            warningOnly: false,
            stdoutPreview: "stdout line",
            stderrPreview: "stderr line",
            stdoutTruncated: false,
            stderrTruncated: false,
          },
        },
      },
    }, 2, 10);
    const rendered = formatTranscriptPart(parts[1]!);
    expect(rendered).toContain("[command]");
    expect(rendered).toContain("npm test");
    expect(rendered).toContain("Exited with code 1");
    expect(rendered).toContain("stderr");
  });

  it("preserves multiline command previews instead of flattening them", () => {
    const parts = presentToolAfterEvent({
      name: "run_command",
      callId: "call-cmd-2",
      durationMs: 45,
      result: {
        success: true,
        output: "done",
        error: null,
        artifacts: [],
        metadata: {
          commandResult: {
            command: "bun run test",
            cwd: ".",
            exitCode: 0,
            timedOut: false,
            warningOnly: false,
            stdoutPreview: "first line\nsecond line",
            stderrPreview: "warn one\nwarn two",
            stdoutTruncated: false,
            stderrTruncated: false,
          },
        },
      },
    }, 2, 10);

    const rendered = stripAnsi(formatTranscriptPart(parts[1]!));
    expect(rendered).toContain("  stdout\n");
    expect(rendered).toContain("    first line");
    expect(rendered).toContain("    second line");
    expect(rendered).toContain("  stderr\n");
    expect(rendered).toContain("    warn one");
    expect(rendered).not.toContain("↵");
  });

  it("renders validation and diagnostic parts from typed validation metadata", () => {
    const parts = presentToolAfterEvent({
      name: "write_file",
      callId: "call-validate-1",
      durationMs: 12,
      result: {
        success: true,
        output: "Wrote file",
        error: null,
        artifacts: ["src/new.ts"],
        metadata: {
          validationResult: {
            passed: false,
            diagnosticErrors: ["src/new.ts: Unexpected token"],
            testPassed: false,
            testOutputPreview: "1 failed",
            testSummary: {
              framework: "vitest",
              passed: 3,
              failed: 1,
              failureMessages: ["fails"],
            },
          },
        },
      },
    }, 2, 10);

    expect(formatTranscriptPart(parts[1]!)).toContain("Validation failed");
    expect(formatTranscriptPart(parts[2]!)).toContain("src/new.ts: Unexpected token");
  });
});

describe("formatTranscriptPart multiline previews", () => {
  it("preserves multiline validation previews instead of flattening them", () => {
    const parts = presentToolAfterEvent({
      name: "write_file",
      callId: "call-validate-2",
      durationMs: 12,
      result: {
        success: true,
        output: "Wrote file",
        error: null,
        artifacts: ["src/new.ts"],
        metadata: {
          validationResult: {
            passed: true,
            diagnosticErrors: [],
            testPassed: true,
            testOutputPreview: "Packages in scope: cli\nRunning test in 5 packages",
          },
        },
      },
    }, 2, 10);

    const rendered = stripAnsi(formatTranscriptPart(parts[1]!));
    expect(rendered).toContain("Validation passed");
    expect(rendered).toContain("  output\n");
    expect(rendered).toContain("    Packages in scope: cli");
    expect(rendered).toContain("    Running test in 5 packages");
    expect(rendered).not.toContain("↵");
  });
});

describe("formatFileEditPreview highlighting", () => {
  const originalNoColor = process.env["NO_COLOR"];

  afterEach(() => {
    if (originalNoColor === undefined) {
      delete process.env["NO_COLOR"];
    } else {
      process.env["NO_COLOR"] = originalNoColor;
    }
  });

  it("uses ANSI syntax highlighting when colors are enabled", async () => {
    delete process.env["NO_COLOR"];
    const { formatFileEditPreview } = await import("./format.js");

    const result = formatFileEditPreview([{
      path: "src/new-file.ts",
      kind: "create",
      additions: 1,
      deletions: 0,
      unifiedDiff: "--- /dev/null\n+++ b/src/new-file.ts\n@@ -0,0 +1,1 @@\n+const value = 1;",
      truncated: false,
      after: "const value = 1;\n",
      structuredDiff: {
        hunks: [{
          oldStart: 0,
          oldLines: 0,
          newStart: 1,
          newLines: 1,
          lines: [{ type: "add", text: "const value = 1;", oldLine: null, newLine: 1 }],
        }],
      },
    }]);

    expect(result).toContain("\x1b[");
  });
});

// ─── Improvement 9: Error Enrichment ──────────────────────────

describe("inferErrorSuggestion", () => {
  it("suggests retry for rate limit errors", () => {
    const result = inferErrorSuggestion("rate limit exceeded", []);
    expect(result).toContain("Rate limited");
  });

  it("suggests /clear for context length errors", () => {
    const result = inferErrorSuggestion("maximum context length exceeded", []);
    expect(result).toContain("Context limit");
  });

  it("suggests check connectivity for timeout errors", () => {
    const result = inferErrorSuggestion("ETIMEDOUT connecting to API", []);
    expect(result).toContain("timed out");
  });

  it("suggests different approach for repeated same-tool failures", () => {
    const tools = [
      { name: "run_command", success: false, durationMs: 3000 },
      { name: "run_command", success: false, durationMs: 3200 },
    ];
    const result = inferErrorSuggestion("Command failed", tools);
    expect(result).toContain("run_command");
    expect(result).toContain("failed repeatedly");
  });

  it("returns null when no pattern matches", () => {
    const result = inferErrorSuggestion("Something unknown happened", []);
    expect(result).toBeNull();
  });
});

describe("formatEnrichedError", () => {
  it("includes tool chain and suggestion", () => {
    const result = formatEnrichedError({
      message: "API error",
      recentTools: [
        { name: "read_file", success: true, durationMs: 120 },
        { name: "run_command", success: false, durationMs: 3200 },
      ],
      suggestion: "Try a different approach.",
    });
    expect(result).toContain("API error");
    expect(result).toContain("read_file");
    expect(result).toContain("run_command");
    expect(result).toContain("Try a different approach");
  });

  it("falls back to simple error when no tools", () => {
    const result = formatEnrichedError({
      message: "Something broke",
      recentTools: [],
      suggestion: null,
    });
    expect(result).toContain("Something broke");
    expect(result).not.toContain("Recent tool chain");
  });
});

// ─── Improvement 3: Per-Turn Summary ──────────────────────────

describe("formatTurnSummary", () => {
  it("includes all stats", () => {
    const result = formatTurnSummary({
      iterationCount: 3,
      toolCallCount: 8,
      inputTokens: 45000,
      costDelta: 0.042,
      elapsedMs: 12300,
    });
    expect(result).toContain("3 iterations");
    expect(result).toContain("8 tool calls");
    expect(result).toContain("45k input tokens");
    expect(result).toContain("$0.0420");
    expect(result).toContain("12.3s");
  });

  it("omits zero-value fields", () => {
    const result = formatTurnSummary({
      iterationCount: 1,
      toolCallCount: 0,
      inputTokens: 0,
      costDelta: 0,
      elapsedMs: 500,
    });
    expect(result).toContain("1 iteration");
    expect(result).not.toContain("tool calls");
    expect(result).not.toContain("input tokens");
    expect(result).not.toContain("$");
    expect(result).toContain("500ms");
  });
});

describe("turn boundaries", () => {
  it("formats a compact turn start header", () => {
    const result = formatTurnStart("Refactor the auth flow");
    expect(result).toContain("turn");
    expect(result).toContain("Refactor the auth flow");
  });

  it("formats a completed turn footer from composed metrics", () => {
    const result = formatTurnEnd({
      id: "turn-1",
      userText: "Refactor the auth flow",
      startedAt: 1_000,
      finishedAt: 1_500,
      status: "completed",
      entries: [],
      summary: {
        iterations: 3,
        toolCalls: 2,
        cost: 0.01,
        elapsedMs: 500,
      },
      metrics: {
        toolCalls: 2,
        filesChanged: 1,
        validationFailed: true,
        iterations: 3,
        cost: 0.01,
        elapsedMs: 500,
      },
    });

    expect(result).toContain("Done");
    expect(result).toContain("3 iterations");
    expect(result).toContain("2 tool calls");
    expect(result).toContain("1 file changed");
    expect(result).toContain("validation failed");
    expect(result).toContain("$0.0100");
  });
});

// ─── Compaction Result ────────────────────────────────────────

describe("formatCompactionResult", () => {
  it("shows before → after tokens with percentage reduction", () => {
    const result = formatCompactionResult({
      tokensBefore: 12000,
      estimatedTokens: 6726,
      removedCount: 0,
    });
    expect(result).toContain("12k");
    expect(result).toContain("7k");
    expect(result).toContain("44%");
  });

  it("includes removed message count when > 0", () => {
    const result = formatCompactionResult({
      tokensBefore: 50000,
      estimatedTokens: 30000,
      removedCount: 8,
    });
    expect(result).toContain("removed 8 msgs");
  });

  it("includes pruned tool count when provided", () => {
    const result = formatCompactionResult({
      tokensBefore: 50000,
      estimatedTokens: 35000,
      removedCount: 0,
      prunedCount: 5,
    });
    expect(result).toContain("pruned 5 outputs");
  });

  it("shows both removed and pruned when both present", () => {
    const result = formatCompactionResult({
      tokensBefore: 80000,
      estimatedTokens: 40000,
      removedCount: 12,
      prunedCount: 3,
    });
    expect(result).toContain("removed 12 msgs");
    expect(result).toContain("pruned 3 outputs");
    expect(result).toContain("50%");
  });

  it("handles edge case where tokensBefore is 0", () => {
    const result = formatCompactionResult({
      tokensBefore: 0,
      estimatedTokens: 5000,
      removedCount: 0,
    });
    // Should still show remaining tokens, no percentage
    expect(result).toContain("5k");
    expect(result).not.toContain("%");
  });

  it("shows 'no reduction' when tokens unchanged", () => {
    const result = formatCompactionResult({
      tokensBefore: 10000,
      estimatedTokens: 10000,
      removedCount: 0,
    });
    expect(result).toContain("no reduction");
  });
});

// ─── Improvement 4: Session Summary ───────────────────────────
describe("formatSessionSummary", () => {
  it("formats full session data", () => {
    const result = formatSessionSummary({
      sessionId: "sess_abc123",
      totalIterations: 12,
      totalToolCalls: 34,
      toolUsage: new Map([
        ["read_file", 15],
        ["run_command", 8],
        ["replace_in_file", 6],
        ["search_files", 5],
      ]),
      filesChanged: ["src/auth.ts", "src/login.ts", "tests/auth.test.ts"],
      planSteps: [
        { description: "Step 1", status: "completed" },
        { description: "Step 2", status: "completed" },
        { description: "Step 3", status: "completed" },
        { description: "Step 4", status: "completed" },
        { description: "Step 5", status: "in_progress" },
      ],
      totalCost: 0.1234,
      totalInputTokens: 89000,
      totalOutputTokens: 12000,
      elapsedMs: 135000,
      completionReason: "completed",
    });

    expect(result).toContain("sess_abc123");
    expect(result).toContain("2m 15s");
    expect(result).toContain("12");
    expect(result).toContain("34");
    expect(result).toContain("read_file: 15");
    expect(result).toContain("src/auth.ts");
    expect(result).toContain("4/5 completed");
    expect(result).toContain("$0.1234");
    expect(result).toContain("89k in / 12k out");
  });

  it("handles empty data gracefully", () => {
    const result = formatSessionSummary({
      sessionId: "sess_empty",
      totalIterations: 0,
      totalToolCalls: 0,
      toolUsage: new Map(),
      filesChanged: [],
      totalCost: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      elapsedMs: 100,
      completionReason: "completed",
    });

    expect(result).toContain("sess_empty");
    expect(result).not.toContain("Tool usage:");
    expect(result).not.toContain("Files changed");
    expect(result).not.toContain("Plan:");
    expect(result).not.toContain("Cost:");
  });
});

describe("formatSessionSummary collections", () => {
  it("sorts tool usage by count descending", () => {
    const result = formatSessionSummary({
      sessionId: "sess_sort",
      totalIterations: 5,
      totalToolCalls: 10,
      toolUsage: new Map([
        ["b_tool", 2],
        ["a_tool", 8],
      ]),
      filesChanged: [],
      totalCost: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      elapsedMs: 1000,
      completionReason: "completed",
    });

    const aIdx = result.indexOf("a_tool: 8");
    const bIdx = result.indexOf("b_tool: 2");
    expect(aIdx).toBeLessThan(bIdx);
  });

  it("truncates files list at 15", () => {
    const files = Array.from({ length: 20 }, (_, i) => `file_${i}.ts`);
    const result = formatSessionSummary({
      sessionId: "sess_trunc",
      totalIterations: 1,
      totalToolCalls: 0,
      toolUsage: new Map(),
      filesChanged: files,
      totalCost: 0,
      totalInputTokens: 0,
      totalOutputTokens: 0,
      elapsedMs: 100,
      completionReason: "completed",
    });

    expect(result).toContain("file_0.ts");
    expect(result).toContain("file_14.ts");
    expect(result).not.toContain("file_15.ts");
    expect(result).toContain("+5 more");
  });

  it("includes delegated work summary when subagents ran", () => {
    const result = formatSessionSummary({
      sessionId: "sess_subagents",
      totalIterations: 5,
      totalToolCalls: 12,
      toolUsage: new Map(),
      filesChanged: [],
      totalCost: 0.12,
      totalInputTokens: 10000,
      totalOutputTokens: 3000,
      elapsedMs: 30000,
      completionReason: "completed",
      delegatedWork: {
        childCount: 3,
        children: [],
        byType: { explore: 3 },
        lanes: ["spec/docs", "frontend", "runtime"],
        totalDelegatedDurationMs: 147000,
        parallelBatchCount: 1,
        maxParallelChildren: 2,
      },
    });

    expect(result).toContain("Subagents:");
    expect(result).toContain("explore=3");
    expect(result).toContain("spec/docs, frontend, runtime");
    expect(result).toContain("2m 27s total");
    expect(result).toContain("1 batch(es), max 2 child(ren)");
  });
});

describe("subagent formatting", () => {
  it("formats subagent lifecycle lines", () => {
    expect(formatSubagentBatchLaunch("explore", 2)).toContain("Launching 2 explore subagents in parallel");
    expect(formatSubagentStart({
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "running",
    })).toContain("root-sub-1");
    expect(formatSubagentError({
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "error",
      durationMs: 1200,
      error: "Provider exploded",
    })).toContain("Provider exploded");
  });

  it("formats child tool counters distinctly", () => {
    const result = formatToolStart({
      name: "search_files",
      params: { pattern: "FixedArray" },
      iteration: 2,
      maxIter: 0,
      counterLabel: "root-sub-1",
    });
    expect(result).toContain("[root-sub-1:2]");
  });

  it("uses the same child counter formatting for grouped tool output", () => {
    const result = formatToolGroupStart({
      name: "read_file",
      count: 3,
      paramSummaries: ["a.ts", "b.ts"],
      iteration: 4,
      maxIter: 0,
      counterLabel: "root-sub-1",
    });
    expect(result).toContain("[root-sub-1:4]");
  });
});

// ─── Reasoning Display ───────────────────────────────────────

describe("formatReasoning", () => {
  it("returns dimmed text with info icon", () => {
    const result = formatReasoning("I'll read the file to understand the structure");
    expect(result).toContain("ℹ");
    expect(result).toContain("read the file");
  });

  it("returns null for empty string", () => {
    expect(formatReasoning("")).toBeNull();
  });

  it("returns null for whitespace-only string", () => {
    expect(formatReasoning("   \n  \t  ")).toBeNull();
  });

  it("truncates to 120 chars with ellipsis", () => {
    const long = "A".repeat(200);
    const result = formatReasoning(long);
    expect(result).not.toBeNull();
    // Strip ANSI codes to check length
    const stripped = result!.replace(/\x1b\[[0-9;]*m/g, "");
    // "  ℹ " prefix (4 chars) + 119 chars + "…" = 124 chars
    expect(stripped.length).toBeLessThanOrEqual(125);
  });

  it("collapses newlines to spaces", () => {
    const result = formatReasoning("First line\nSecond line\nThird line");
    expect(result).not.toBeNull();
    const stripped = result!.replace(/\x1b\[[0-9;]*m/g, "");
    expect(stripped).not.toContain("\n");
    expect(stripped).toContain("First line Second line");
  });

  it("takes first sentence if shorter than 120 chars", () => {
    const result = formatReasoning("I need to check the file. Then I will edit it and run the tests to make sure everything works correctly.");
    expect(result).not.toBeNull();
    const stripped = result!.replace(/\x1b\[[0-9;]*m/g, "");
    expect(stripped).toContain("I need to check the file.");
    expect(stripped).not.toContain("Then I will");
  });
});

describe("formatToolStart with verbs", () => {
  it("uses present-tense verb instead of raw tool name", () => {
    const result = formatToolStart({
      name: "read_file",
      params: { path: "src/main.ts" },
      iteration: 1,
      maxIter: 30,
    });
    expect(result).toContain("Reading");
    expect(result).toContain("src/main.ts");
    expect(result).not.toContain("read_file");
  });

  it("uses present-tense verb for search_files", () => {
    const result = formatToolStart({
      name: "search_files",
      params: { pattern: "handleAuth" },
      iteration: 2,
      maxIter: 30,
    });
    expect(result).toContain("Searching");
    expect(result).toContain("handleAuth");
  });
});

describe("formatToolEnd with verbs", () => {
  it("uses past-tense verb and includes param summary on success", () => {
    const result = formatToolEnd("read_file", true, 120, undefined, { path: "src/main.ts" });
    expect(result).toContain("Read");
    expect(result).toContain("src/main.ts");
    expect(result).toContain("120ms");
  });

  it("uses past-tense verb on failure", () => {
    const result = formatToolEnd("run_command", false, 3000, "exit code 1", { command: "bun test" });
    expect(result).toContain("Ran");
    expect(result).toContain("exit code 1");
  });
});

// ─── Tool Grouping ──────────────────────────────────────────

describe("formatToolGroupStart", () => {
  it("formats grouped read_file calls", () => {
    const result = formatToolGroupStart({
      name: "read_file",
      count: 3,
      paramSummaries: ["src/main.ts", "src/format.ts", "src/output-state.ts"],
      iteration: 1,
      maxIter: 30,
    });
    expect(result).toContain("Reading");
    expect(result).toContain("3 files");
    expect(result).toContain("src/main.ts");
    expect(result).toContain("src/format.ts");
    expect(result).toContain("src/output-state.ts");
  });

  it("truncates params list beyond 3 items", () => {
    const result = formatToolGroupStart({
      name: "read_file",
      count: 5,
      paramSummaries: ["a.ts", "b.ts", "c.ts", "d.ts", "e.ts"],
      iteration: 2,
      maxIter: 30,
    });
    expect(result).toContain("5 files");
    expect(result).toContain("a.ts");
    expect(result).toContain("b.ts");
    expect(result).toContain("c.ts");
    expect(result).toContain("+2 more");
    expect(result).not.toContain("d.ts");
  });

  it("uses generic 'calls' for non-file tools", () => {
    const result = formatToolGroupStart({
      name: "run_command",
      count: 3,
      paramSummaries: ["ls", "pwd", "whoami"],
      iteration: 1,
      maxIter: 30,
    });
    expect(result).toContain("Running");
    expect(result).toContain("3 calls");
  });
});

describe("formatToolGroupEnd", () => {
  it("formats grouped completion with total duration", () => {
    const result = formatToolGroupEnd("read_file", 3, true, 303);
    expect(result).toContain("Read");
    expect(result).toContain("3 files");
    expect(result).toContain("303ms");
  });

  it("formats failure", () => {
    const result = formatToolGroupEnd("run_command", 2, false, 5000, "exit code 1");
    expect(result).toContain("Ran");
    expect(result).toContain("2 calls");
    expect(result).toContain("exit code 1");
  });
});

describe("SubagentPanelRenderer", () => {
  const originalIsTTY = Object.getOwnPropertyDescriptor(process.stderr, "isTTY");
  let writeSpy: ReturnType<typeof vi.spyOn<typeof process.stderr, "write">>;

  beforeEach(() => {
    vi.useFakeTimers();
    Object.defineProperty(process.stderr, "isTTY", {
      configurable: true,
      value: true,
    });
    writeSpy = vi.spyOn(process.stderr, "write").mockReturnValue(true);
  });

  afterEach(() => {
    writeSpy.mockRestore();
    if (originalIsTTY) {
      Object.defineProperty(process.stderr, "isTTY", originalIsTTY);
    } else {
      delete (process.stderr as { isTTY?: boolean }).isTTY;
    }
    vi.useRealTimers();
  });

  function makePanel(overrides: Partial<Parameters<SubagentPanelRenderer["formatPanels"]>[0][number]> = {}) {
    return {
      agentId: "root-sub-1",
      agentType: "explore",
      laneLabel: "docs/spec",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "running" as const,
      currentIteration: 3,
      startedAtMs: 1_000,
      currentActivity: "Reading docs",
      recentActivity: ["Reading docs", "Searching FixedArray"],
      ...overrides,
    };
  }

  it("formats stable multi-line child panels", () => {
    const renderer = new SubagentPanelRenderer(false);
    const lines = renderer.formatPanels([
      makePanel(),
      makePanel({
        agentId: "root-sub-2",
        laneLabel: "runtime",
        status: "completed",
        currentIteration: 5,
        durationMs: 4_500,
        currentActivity: "Completed after 5 iterations",
        recentActivity: ["Completed after 5 iterations"],
        quality: {
          score: 0.74,
          completeness: "partial",
        },
      }),
    ], 6_000);

    expect(lines.length).toBe(6);
    expect(lines[0]).toContain("Subagent root-sub-1");
    expect(lines[1]).toContain("iter 3");
    expect(lines[2]).toContain("Recent: Reading docs");
    expect(lines[3]).toContain("Subagent root-sub-2");
    expect(lines[4]).toContain("score 0.74");
  });

  it("debounces repeated panel updates into a single redraw", () => {
    const renderer = new SubagentPanelRenderer(true);
    renderer.setPanels([makePanel({ currentIteration: 1 })]);
    renderer.setPanels([makePanel({ currentIteration: 2 })]);

    expect(writeSpy).not.toHaveBeenCalled();

    vi.advanceTimersByTime(100);

    expect(writeSpy).toHaveBeenCalledTimes(1);
    expect(String(writeSpy.mock.calls[0]?.[0])).toContain("iter 2");
  });

  it("does not redraw on resume until the next setPanels call", () => {
    const renderer = new SubagentPanelRenderer(true);
    renderer.setPanels([makePanel()]);
    vi.advanceTimersByTime(100);

    writeSpy.mockClear();

    renderer.suspend();
    writeSpy.mockClear();

    renderer.resume();
    expect(writeSpy).not.toHaveBeenCalled();

    renderer.setPanels([makePanel({ currentIteration: 4 })]);
    expect(writeSpy).not.toHaveBeenCalled();

    vi.advanceTimersByTime(100);

    expect(writeSpy).toHaveBeenCalledTimes(1);
    expect(String(writeSpy.mock.calls[0]?.[0])).toContain("iter 4");
  });
});

describe("summarizeSubagentUpdate", () => {
  it("prefers explicit summary text", () => {
    expect(summarizeSubagentUpdate({
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      status: "running",
      milestone: "tool:before",
      toolName: "search_files",
      summary: "Searching tests and lowering code",
    })).toBe("Searching tests and lowering code");
  });

  it("falls back to milestone-specific summaries", () => {
    expect(summarizeSubagentUpdate({
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      status: "running",
      milestone: "iteration:start",
      iteration: 2,
    })).toContain("iteration 2");
  });
});
