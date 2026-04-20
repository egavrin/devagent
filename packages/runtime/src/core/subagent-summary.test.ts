import { describe, expect, it } from "vitest";

import {
  aggregateDelegatedWork,
  formatDuration,
  loggedSubagentRunFromEvent,
} from "./subagent-summary.js";
import { AgentType } from "./types.js";

describe("formatDuration", () => {
  it("formats milliseconds, seconds, and minutes", () => {
    expect(formatDuration(250)).toBe("250ms");
    expect(formatDuration(1_500)).toBe("1.5s");
    expect(formatDuration(65_000)).toBe("1m 5s");
  });
});

describe("loggedSubagentRunFromEvent", () => {
  it("initializes a running subagent with preserved tool calls", () => {
    const result = loggedSubagentRunFromEvent({
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "running",
      batchId: "batch-1",
      batchSize: 2,
    }, {
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "running",
      toolCalls: 3,
    });

    expect(result).toMatchObject({
      status: "running",
      toolCalls: 3,
      batchId: "batch-1",
      batchSize: 2,
    });
  });

  it("preserves tool calls and quality on completion", () => {
    const result = loggedSubagentRunFromEvent({
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "completed",
      durationMs: 4_200,
      iterations: 3,
      batchId: "batch-1",
      batchSize: 2,
      cost: {
        inputTokens: 100,
        outputTokens: 20,
        cacheReadTokens: 0,
        cacheWriteTokens: 0,
        totalCost: 0.01,
      },
      parsedOutputKeys: ["answer"],
      quality: {
        score: 0.74,
        completeness: "partial",
      },
    }, {
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "running",
      batchId: "batch-1",
      batchSize: 2,
      toolCalls: 5,
    });

    expect(result).toMatchObject({
      status: "completed",
      toolCalls: 5,
      durationMs: 4_200,
      iterations: 3,
      quality: {
        score: 0.74,
        completeness: "partial",
      },
    });
  });

  it("preserves tool calls on error", () => {
    const result = loggedSubagentRunFromEvent({
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "error",
      durationMs: 1_500,
      error: "Provider exploded",
      batchId: "batch-1",
      batchSize: 2,
    }, {
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "frontend",
      objective: "Inspect frontend lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "running",
      toolCalls: 2,
    });

    expect(result).toMatchObject({
      status: "error",
      toolCalls: 2,
      durationMs: 1_500,
    });
  });
});

describe("aggregateDelegatedWork", () => {
  it("aggregates lanes, durations, type counts, and parallel batches", () => {
    const summary = aggregateDelegatedWork([
      {
        agentId: "root-sub-1",
        parentAgentId: "root",
        depth: 1,
        agentType: AgentType.EXPLORE,
        laneLabel: "frontend",
        objective: "Inspect frontend lane",
        model: "gpt-5.4-mini",
        reasoningEffort: "low",
        status: "completed",
        durationMs: 4_200,
        iterations: 3,
        batchId: "batch-1",
        batchSize: 2,
        toolCalls: 1,
      },
      {
        agentId: "root-sub-2",
        parentAgentId: "root",
        depth: 1,
        agentType: AgentType.EXPLORE,
        laneLabel: "runtime",
        objective: "Inspect runtime lane",
        model: "gpt-5.4-mini",
        reasoningEffort: "low",
        status: "error",
        durationMs: 2_800,
        batchId: "batch-1",
        batchSize: 2,
        toolCalls: 2,
      },
      {
        agentId: "root-sub-3",
        parentAgentId: "root",
        depth: 1,
        agentType: AgentType.REVIEWER,
        laneLabel: "docs",
        objective: "Review findings",
        model: "gpt-5.4",
        reasoningEffort: "medium",
        status: "completed",
        durationMs: 1_000,
        iterations: 1,
        toolCalls: 0,
      },
    ]);

    expect(summary.childCount).toBe(3);
    expect(summary.byType).toEqual({ explore: 2, reviewer: 1 });
    expect(summary.lanes).toEqual(["frontend", "runtime", "docs"]);
    expect(summary.totalDelegatedDurationMs).toBe(8_000);
    expect(summary.parallelBatchCount).toBe(1);
    expect(summary.maxParallelChildren).toBe(2);
  });
});
