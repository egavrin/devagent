import { AgentType } from "@devagent/runtime";
import { describe, expect, it } from "vitest";

import { OutputState } from "./output-state.js";

describe("OutputState delegated work summary", () => {
  it("preserves rich subagent fields when building the summary", () => {
    const state = new OutputState();
    state.sessionSubagents.set("root-sub-1", {
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      agentType: AgentType.EXPLORE,
      laneLabel: "spec/docs",
      objective: "Inspect docs lane",
      model: "gpt-5.4-mini",
      reasoningEffort: "low",
      status: "completed",
      durationMs: 1234,
      iterations: 4,
      batchId: "batch-1",
      batchSize: 3,
      toolCalls: 7,
      quality: {
        score: 0.8,
        completeness: "partial",
      },
    });

    const summary = state.buildDelegatedWorkSummary();
    expect(summary.childCount).toBe(1);
    expect(summary.children[0]).toMatchObject({
      agentId: "root-sub-1",
      parentAgentId: "root",
      depth: 1,
      objective: "Inspect docs lane",
      model: "gpt-5.4-mini",
      toolCalls: 7,
      iterations: 4,
      batchId: "batch-1",
      batchSize: 3,
    });
    expect(summary.parallelBatchCount).toBe(1);
    expect(summary.maxParallelChildren).toBe(3);
  });

  it("clears durable transcript bookkeeping on turn reset", () => {
    const state = new OutputState();
    state.announcedSubagentBatches.add("batch-1");
    state.subagentDisplay.set("root-sub-1", {
      agentId: "root-sub-1",
      agentType: AgentType.EXPLORE,
      model: "gpt-5.4-mini",
      status: "running",
      currentIteration: 1,
      startedAtMs: 123,
      currentActivity: "Reading docs",
      recentActivity: [],
    });

    state.resetTurn();

    expect(state.announcedSubagentBatches.size).toBe(0);
    expect(state.subagentDisplay.size).toBe(0);
  });
});
