import { describe, it, expect } from "vitest";
import { createPlanTool } from "./plan-tool.js";
import { EventBus } from "@devagent/core";
import { SessionState } from "./session-state.js";

function makePlanTool() {
  const bus = new EventBus();
  const state = new SessionState();
  const tool = createPlanTool(bus, () => state);
  return { tool, bus, state };
}

describe("plan-tool validation", () => {
  it("rejects invalid step status (fail-fast, no silent coercion)", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "done" }, // "done" is not valid
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Invalid plan step status");
    expect(result.error).toContain("done");
  });

  it("rejects plan with more than one in_progress step", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step A", status: "in_progress" },
          { description: "Step B", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("at most one in_progress step");
  });

  it("accepts valid plan with one in_progress step", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "in_progress" },
          { description: "Step 3", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(true);
  });

  it("accepts plan with zero in_progress steps", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(true);
  });

  it("rejects empty steps array", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      { steps: JSON.stringify([]) },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("at least one step");
  });

  it("rejects invalid JSON", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      { steps: "not json" },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Invalid steps JSON");
  });

  it("rejects update when it reverts previously completed steps", async () => {
    const { tool } = makePlanTool();

    // First call: 2/3 completed
    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "completed" },
          { description: "Step 3", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    // Second call: resets all to pending (0/3 completed)
    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "pending" },
          { description: "Step 2", status: "pending" },
          { description: "Step 3", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Plan regression detected");
    expect(result.error).toContain("Step 1");
    expect(result.error).toContain("Step 2");
  });

  it("does not warn when completed count stays the same or increases", async () => {
    const { tool } = makePlanTool();

    // First call: 1/3 completed
    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "in_progress" },
          { description: "Step 3", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    // Second call: 2/3 completed (increase)
    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "completed" },
          { description: "Step 3", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(true);
    expect(result.output).not.toContain("WARNING");
  });

  it("rejects regression when a completed step is reverted but another is newly completed (same count)", async () => {
    const { tool } = makePlanTool();

    // First call: Step 1 completed
    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    // Second call: Step 1 reverted to pending, Step 2 completed (same count: 1)
    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "pending" },
          { description: "Step 2", status: "completed" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Plan regression detected");
    expect(result.error).toContain("Step 1");
  });

  it("allows explicit regression when allow_regression is true", async () => {
    const { tool } = makePlanTool();

    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "pending" },
          { description: "Step 2", status: "pending" },
        ]),
        allow_regression: true,
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("Plan updated (0/2 completed)");
  });

  it("rejects active-plan churn when most prior steps are replaced", async () => {
    const { tool } = makePlanTool();

    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Collect scoped diffs", status: "in_progress" },
          { description: "Assess risks", status: "pending" },
          { description: "Report findings", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Scan context usage", status: "in_progress" },
          { description: "Investigate compaction", status: "pending" },
          { description: "Draft remediation roadmap", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Plan churn detected");
  });

  it("allows cosmetic formatting differences in unchanged steps", async () => {
    const { tool } = makePlanTool();

    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Collect scoped diffs", status: "in_progress" },
          { description: "Assess risks regressions", status: "pending" },
          { description: "Report findings", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "  collect   scoped diffs  ", status: "completed" },
          { description: "Assess-risks / regressions", status: "in_progress" },
          { description: "Report findings!", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(true);
  });

  it("rejects missing step description", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Invalid plan step description");
  });

  it("rejects empty string step description", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: "", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Invalid plan step description");
  });

  it("rejects non-string step description", async () => {
    const { tool } = makePlanTool();

    const result = await tool.handler(
      {
        steps: JSON.stringify([
          { description: 42, status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("Invalid plan step description");
  });

  it("syncs plan to session state", async () => {
    const { tool, state } = makePlanTool();

    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const plan = state.getPlan();
    expect(plan).toHaveLength(2);
    expect(plan![0]!.description).toBe("Step 1");
    expect(plan![0]!.status).toBe("completed");
    expect(plan![1]!.description).toBe("Step 2");
    expect(plan![1]!.status).toBe("in_progress");
  });
});

// ─── Plan Enrichment ─────────────────────────────────────────

describe("plan enrichment", () => {
  function makePlanToolWithIteration(iteration: number) {
    const bus = new EventBus();
    const state = new SessionState();
    const tool = createPlanTool(bus, () => state, () => iteration);
    return { tool, bus, state };
  }

  it("new steps get transition metadata", async () => {
    const { tool, state } = makePlanToolWithIteration(3);

    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "pending" },
          { description: "Step 2", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const plan = state.getPlan()!;
    expect(plan[0]!.lastTransitionIteration).toBe(3);
    expect(plan[0]!.lastTransitionTimestamp).toBeGreaterThan(0);
    expect(plan[1]!.lastTransitionIteration).toBe(3);
  });

  it("unchanged status preserves prior metadata", async () => {
    const { tool, state } = makePlanToolWithIteration(1);

    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const firstPlan = state.getPlan()!;
    const firstTs = firstPlan[0]!.lastTransitionTimestamp;

    // Same status, later iteration — should preserve original metadata
    const bus2 = new EventBus();
    const tool2 = createPlanTool(bus2, () => state, () => 5);
    await tool2.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const secondPlan = state.getPlan()!;
    expect(secondPlan[0]!.lastTransitionIteration).toBe(1);
    expect(secondPlan[0]!.lastTransitionTimestamp).toBe(firstTs);
  });

  it("changed status updates metadata", async () => {
    const { tool, state } = makePlanToolWithIteration(2);

    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "pending" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    // Change status
    const bus2 = new EventBus();
    const tool2 = createPlanTool(bus2, () => state, () => 7);
    await tool2.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const plan = state.getPlan()!;
    expect(plan[0]!.lastTransitionIteration).toBe(7);
  });

  it("round-trips transition metadata through SessionState JSON", async () => {
    const { tool, state } = makePlanToolWithIteration(4);

    await tool.handler(
      {
        steps: JSON.stringify([
          { description: "Step 1", status: "completed" },
          { description: "Step 2", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    const json = state.toJSON();
    const restored = SessionState.fromJSON(json);
    const plan = restored.getPlan()!;
    expect(plan[0]!.lastTransitionIteration).toBe(4);
    expect(plan[1]!.lastTransitionIteration).toBe(4);
    expect(plan[0]!.lastTransitionTimestamp).toBeGreaterThan(0);
  });
});
