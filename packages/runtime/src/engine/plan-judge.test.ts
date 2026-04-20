import { describe, it, expect, vi } from "vitest";

import { judgePlanQuality, isStructuralChange } from "./plan-judge.js";
import type { PlanStep } from "./plan-tool.js";
import { SessionState } from "./session-state.js";
import type { LLMProvider, StreamChunk } from "../core/index.js";

// ─── Mock helpers ────────────────────────────────────────────

function mockProvider(responseJson: string): LLMProvider {
  return {
    id: "mock",
    chat: vi.fn(async function* (): AsyncIterable<StreamChunk> {
      yield { type: "text", content: responseJson };
      yield { type: "done", content: "" };
    }),
    abort: vi.fn(),
  };
}

function throwingProvider(): LLMProvider {
  return {
    id: "mock-error",
    chat: vi.fn(async function* (): AsyncIterable<StreamChunk> {
      throw new Error("Provider error");
    }),
    abort: vi.fn(),
  };
}

// ─── Tests ───────────────────────────────────────────────────

describe("plan-judge — judgePlanQuality", () => {
  it("returns null on provider error", async () => {
    const provider = throwingProvider();
    const steps: PlanStep[] = [
      { description: "Step 1", status: "pending" },
      { description: "Step 2", status: "pending" },
      { description: "Step 3", status: "pending" },
    ];

    const result = await judgePlanQuality(
      provider, "Fix the login bug", steps, null, null, 10,
    );
    expect(result).toBeNull();
  });

  it("returns high quality_score for well-formed relevant plan", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.9, "issues": [], "suggestion": null}',
    );
    const steps: PlanStep[] = [
      { description: "Analyze login handler", status: "completed" },
      { description: "Fix auth validation", status: "in_progress" },
      { description: "Add unit tests", status: "pending" },
    ];

    const result = await judgePlanQuality(
      provider, "Fix the login bug", steps, null, null, 10,
    );
    expect(result).not.toBeNull();
    expect(result!.quality_score).toBe(0.9);
    expect(result!.issues).toHaveLength(0);
  });

  it("returns low quality_score for plan with obvious gaps", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.3, "issues": ["No testing step for code changes", "Steps are too vague"], "suggestion": "Add a testing step and make step descriptions more specific"}',
    );
    const steps: PlanStep[] = [
      { description: "Do everything", status: "pending" },
      { description: "Ship it", status: "pending" },
      { description: "Verify", status: "pending" },
    ];

    const result = await judgePlanQuality(
      provider, "Refactor the auth system", steps, null, null, 10,
    );
    expect(result).not.toBeNull();
    expect(result!.quality_score).toBe(0.3);
    expect(result!.issues.length).toBeGreaterThan(0);
    expect(result!.suggestion).toBeTruthy();
  });

  it("detects drift from original request", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.4, "issues": ["Plan has drifted from original goal"], "suggestion": "Realign plan with user request"}',
    );
    const oldPlan: PlanStep[] = [
      { description: "Fix login validation", status: "completed" },
      { description: "Add error messages", status: "in_progress" },
    ];
    const newPlan: PlanStep[] = [
      { description: "Fix login validation", status: "completed" },
      { description: "Refactor database layer", status: "in_progress" },
      { description: "Add caching", status: "pending" },
    ];

    const result = await judgePlanQuality(
      provider, "Fix the login bug", newPlan, oldPlan, null, 15,
    );
    expect(result).not.toBeNull();
    expect(result!.quality_score).toBe(0.4);
    expect(result!.issues).toContain("Plan has drifted from original goal");
  });

  it("includes session state context when provided", async () => {
    const provider = mockProvider(
      '{"quality_score": 0.8, "issues": [], "suggestion": null}',
    );
    const state = new SessionState();
    state.recordModifiedFile("src/auth.ts");

    const steps: PlanStep[] = [
      { description: "Step 1", status: "pending" },
      { description: "Step 2", status: "pending" },
      { description: "Step 3", status: "pending" },
    ];

    await judgePlanQuality(provider, "Fix auth", steps, null, state, 10);

    const chatCall = (provider.chat as ReturnType<typeof vi.fn>).mock.calls[0]!;
    const messages = chatCall[0] as Array<{ content: string | null }>;
    const userMsg = messages.find((m) => m.content?.includes("Modified files"));
    expect(userMsg).toBeDefined();
  });
});

describe("plan-judge — isStructuralChange", () => {
  it("returns false for status-only updates", () => {
    const oldPlan: PlanStep[] = [
      { description: "Step 1", status: "pending" },
      { description: "Step 2", status: "pending" },
    ];
    const newPlan: PlanStep[] = [
      { description: "Step 1", status: "completed" },
      { description: "Step 2", status: "in_progress" },
    ];
    expect(isStructuralChange(oldPlan, newPlan)).toBe(false);
  });

  it("returns true when step descriptions change", () => {
    const oldPlan: PlanStep[] = [
      { description: "Step 1", status: "pending" },
      { description: "Step 2", status: "pending" },
    ];
    const newPlan: PlanStep[] = [
      { description: "Step 1", status: "pending" },
      { description: "New step 2", status: "pending" },
    ];
    expect(isStructuralChange(oldPlan, newPlan)).toBe(true);
  });

  it("returns true when step count changes", () => {
    const oldPlan: PlanStep[] = [
      { description: "Step 1", status: "pending" },
    ];
    const newPlan: PlanStep[] = [
      { description: "Step 1", status: "pending" },
      { description: "Step 2", status: "pending" },
    ];
    expect(isStructuralChange(oldPlan, newPlan)).toBe(true);
  });

  it("returns true when old plan is null", () => {
    const newPlan: PlanStep[] = [
      { description: "Step 1", status: "pending" },
    ];
    expect(isStructuralChange(null, newPlan)).toBe(true);
  });
});
