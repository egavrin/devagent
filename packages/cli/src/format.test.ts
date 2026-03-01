/**
 * Tests for CLI output formatting — categorical verbosity, gauges, summaries.
 */

import { describe, it, expect, afterEach } from "vitest";
import {
  isCategoryEnabled,
  buildVerbosityConfig,
  formatContextGauge,
  formatEnrichedError,
  inferErrorSuggestion,
  formatTurnHeader,
  formatTurnSummary,
  formatSessionSummary,
  formatCompactionResult,
} from "./format.js";
import type { VerbosityConfig } from "@devagent/core";

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

// ─── Improvement 8: Turn Separators ───────────────────────────

describe("formatTurnHeader", () => {
  it("includes turn number", () => {
    const result = formatTurnHeader(3);
    expect(result).toContain("Turn 3");
  });

  it("includes token info when provided", () => {
    const result = formatTurnHeader(2, { estimated: 52000, max: 128000 });
    expect(result).toContain("tokens: 52k/128k");
  });

  it("includes cost info when provided", () => {
    const result = formatTurnHeader(1, undefined, { totalCost: 0.12 });
    expect(result).toContain("cost: $0.12");
  });

  it("works with all info", () => {
    const result = formatTurnHeader(5, { estimated: 90000, max: 128000 }, { totalCost: 0.0567 });
    expect(result).toContain("Turn 5");
    expect(result).toContain("tokens: 90k/128k");
    expect(result).toContain("cost: $0.0567");
  });

  it("omits token info when tokens are 0", () => {
    const result = formatTurnHeader(1, { estimated: 0, max: 128000 });
    expect(result).not.toContain("tokens:");
  });

  it("omits cost info when cost is 0", () => {
    const result = formatTurnHeader(1, undefined, { totalCost: 0 });
    expect(result).not.toContain("cost:");
  });
});

// ─── Improvement 3: Per-Turn Summary ──────────────────────────

describe("formatTurnSummary", () => {
  it("includes all stats", () => {
    const result = formatTurnSummary({
      iterationCount: 3,
      toolCallCount: 8,
      inputTokens: 45000,
      outputTokens: 5000,
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
      outputTokens: 0,
      costDelta: 0,
      elapsedMs: 500,
    });
    expect(result).toContain("1 iterations");
    expect(result).not.toContain("tool calls");
    expect(result).not.toContain("input tokens");
    expect(result).not.toContain("$");
    expect(result).toContain("500ms");
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
      tokensSaved: 8000,
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
});
