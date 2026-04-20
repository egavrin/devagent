/**
 * Integration tests for plan state lifecycle.
 * Verifies cross-cutting concerns: resume state integrity,
 * plan-mode tool gating, approval semantics, and briefing extraction.
 */
import { describe, it, expect } from "vitest";

import { extractHeuristicBriefing } from "./briefing.js";
import { createPlanTool } from "./plan-tool.js";
import { SessionState } from "./session-state.js";
import { ApprovalGate, EventBus, ApprovalMode, MessageRole } from "../core/index.js";
import type { Message, ApprovalPolicy } from "../core/index.js";

function makePolicy(mode: ApprovalMode): ApprovalPolicy {
  return {
    mode,
    auditLog: false,
    toolOverrides: {},
    pathRules: [],
  };
}

describe("plan state integration", () => {
  it("resume swaps sessionState and plan tool uses the new instance", async () => {
    const bus = new EventBus();

    // Simulate initial creation (main.ts:387)
    let sessionState: SessionState | undefined = new SessionState();
    const initialState = sessionState;

    // Register plan tool with getter (main.ts:390)
    const planTool = createPlanTool(bus, () => sessionState);

    // Simulate resume: replace with restored state (main.ts:731)
    sessionState = SessionState.fromJSON(
      {
        version: 1,
        plan: [{ description: "Old step", status: "completed" as const }],
        modifiedFiles: [],
        envFacts: [],
        toolSummaries: [],
      },
    );

    // Call update_plan — should update the NEW instance
    const result = await planTool.handler(
      {
        steps: JSON.stringify([
          { description: "Old step", status: "completed" },
          { description: "New step", status: "in_progress" },
        ]),
      },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(true);
    expect(sessionState.getPlan()).toHaveLength(2);
    expect(sessionState.getPlan()![1]!.status).toBe("in_progress");
    // Initial instance should be untouched
    expect(initialState.getPlan()).toBeNull();
  });

  it("approval gate allows state tools in all modes", () => {
    const request = {
      toolName: "update_plan",
      toolCategory: "state" as const,
      description: "Update plan",
    };

    for (const mode of [ApprovalMode.SUGGEST, ApprovalMode.AUTO_EDIT, ApprovalMode.FULL_AUTO]) {
      const gate = new ApprovalGate(makePolicy(mode));
      expect(gate.decide(request)).toBe("allow");
    }
  });

  it("briefing skips denied update_plan and uses last successful one", () => {
    const messages: Message[] = [
      { role: MessageRole.USER, content: "Do the task" },
      {
        role: MessageRole.ASSISTANT,
        content: "",
        toolCalls: [{
          name: "update_plan",
          arguments: {
            steps: [{ description: "Bad plan", status: "pending" }],
          },
          callId: "denied-1",
        }],
      },
      {
        role: MessageRole.TOOL,
        content: "Error: Tool execution denied: user rejected",
        toolCallId: "denied-1",
      },
      {
        role: MessageRole.ASSISTANT,
        content: "",
        toolCalls: [{
          name: "update_plan",
          arguments: {
            steps: [{ description: "Good plan", status: "in_progress" }],
          },
          callId: "ok-1",
        }],
      },
      {
        role: MessageRole.TOOL,
        content: "Plan updated (0/1 completed)",
        toolCallId: "ok-1",
      },
    ];

    const briefing = extractHeuristicBriefing(messages, 1);

    expect(briefing.planSteps).toHaveLength(1);
    expect(briefing.planSteps![0]!.description).toBe("Good plan");
  });
});
