import { describe, expect, it } from "vitest";
import {
  createWorkflowFinalTextValidator,
  shouldEnableWorkflowPlanTool,
} from "./workflow-engine.js";

describe("shouldEnableWorkflowPlanTool", () => {
  it("disables update_plan for prompt-only workflow stages", () => {
    for (const taskType of [
      "task-intake",
      "design",
      "breakdown",
      "issue-generation",
      "triage",
      "plan",
      "test-plan",
      "completion",
    ]) {
      expect(shouldEnableWorkflowPlanTool(taskType)).toBe(false);
    }
  });

  it("keeps update_plan available for execution stages", () => {
    for (const taskType of ["implement", "verify", "review", "repair", undefined]) {
      expect(shouldEnableWorkflowPlanTool(taskType)).toBe(true);
    }
  });
});

describe("createWorkflowFinalTextValidator", () => {
  it("enables a strict final-text guard for issue-generation", () => {
    const validator = createWorkflowFinalTextValidator("issue-generation");

    expect(validator).toBeTypeOf("function");
    expect(validator?.("Progress: still working")).toMatchObject({
      valid: false,
      retryMessage: expect.stringContaining("issue-generation"),
    });
    expect(validator?.(JSON.stringify({
      structured: { summary: "ok", issues: [] },
      rendered: "# ok",
    }))).toEqual({ valid: true });
  });

  it("does not enable the strict final-text guard for markdown stages", () => {
    expect(createWorkflowFinalTextValidator("plan")).toBeUndefined();
    expect(createWorkflowFinalTextValidator(undefined)).toBeUndefined();
  });
});
