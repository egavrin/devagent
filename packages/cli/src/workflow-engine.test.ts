import { MessageRole, type SessionStateJSON } from "@devagent/runtime";
import { describe, expect, it } from "vitest";

import {
  createWorkflowFinalTextValidator,
  resolveWorkflowContinuation,
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

describe("resolveWorkflowContinuation", () => {
  const sessionState: SessionStateJSON = {
    version: 1,
    plan: [{ description: "Resume work", status: "in_progress" }],
    modifiedFiles: ["/tmp/file.ts"],
    envFacts: [],
    toolSummaries: [],
  };

  const previousSession = {
    kind: "devagent-headless-v1",
    payload: {
      version: 1,
      messages: [
        { role: MessageRole.USER, content: "Original task" },
        { role: MessageRole.ASSISTANT, content: "Prior progress" },
      ],
      sessionState,
    },
  } as const;

  it("restores prior messages and state for resume continuations", () => {
    const result = resolveWorkflowContinuation("Finish the patch", {
      mode: "resume",
      reason: "retry_no_progress",
      instructions: "Pick up from the last partial edit.",
      session: previousSession,
    });

    expect(result.initialMessages).toEqual(previousSession.payload.messages);
    expect(result.sessionState).toEqual(sessionState);
    expect(result.query).toContain("Continue the prior session using the preserved context below.");
    expect(result.query).toContain("Continuation reason: retry_no_progress");
    expect(result.query).toContain("Continuation instructions:\nPick up from the last partial edit.");
    expect(result.query).toContain("Finish the patch");
  });

  it("starts fresh but preserves continuation metadata for fresh continuations", () => {
    const result = resolveWorkflowContinuation("Retry with a new approach", {
      mode: "fresh",
      reason: "plan_rework",
      instructions: "Do not reuse the prior implementation attempt.",
      session: previousSession,
    });

    expect(result.initialMessages).toBeUndefined();
    expect(result.sessionState).toBeUndefined();
    expect(result.query).not.toContain("Continue the prior session using the preserved context below.");
    expect(result.query).toContain("Continuation reason: plan_rework");
    expect(result.query).toContain("Continuation instructions:\nDo not reuse the prior implementation attempt.");
    expect(result.query).toContain("Retry with a new approach");
  });

  it("returns the original query when there is no continuation metadata", () => {
    expect(resolveWorkflowContinuation("Just run the task", undefined)).toEqual({
      query: "Just run the task",
      initialMessages: undefined,
      sessionState: undefined,
    });
  });
});
