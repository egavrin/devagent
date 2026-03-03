import { describe, it, expect } from "vitest";
import { EventBus } from "@devagent/core";
import { StagnationDetector } from "./stagnation-detector.js";
import { SessionState } from "./session-state.js";

function makeReadonlyCall(name = "read_file") {
  return { name, arguments: {}, callId: "c1" };
}

function makeDetector(sessionState: SessionState | null = null) {
  const bus = new EventBus();
  const detector = new StagnationDetector({
    bus,
    sessionState,
    resolveCategory: (toolName) =>
      toolName === "edit_file" ? "mutating" : "readonly",
  });
  return { detector, bus };
}

describe("StagnationDetector — phase-aware stagnation", () => {
  it("suppresses NO_PROGRESS_LOOP when all plan steps are completed", () => {
    const state = new SessionState();
    state.setPlan([
      { description: "Step 1", status: "completed" },
      { description: "Step 2", status: "completed" },
    ]);
    const { detector } = makeDetector(state);
    const calls = [makeReadonlyCall()];

    // Drive 6 readonly-only cycles — more than NO_PROGRESS_THRESHOLD (5)
    for (let i = 0; i < 6; i++) {
      const result = detector.maybeInjectNoProgressNudge(calls);
      expect(result).toBeNull();
    }
  });

  it("fires NO_PROGRESS_LOOP normally when plan has incomplete steps", () => {
    const state = new SessionState();
    state.setPlan([
      { description: "Step 1", status: "completed" },
      { description: "Step 2", status: "in_progress" },
    ]);
    const { detector } = makeDetector(state);
    const calls = [makeReadonlyCall()];

    // Drive 1 snapshot-init call + 5 stagnant calls = 6 calls to reach threshold.
    // The 6th call fires and returns the nudge; check that at least one was non-null.
    let nudge: string | null = null;
    for (let i = 0; i < 6; i++) {
      const result = detector.maybeInjectNoProgressNudge(calls);
      if (result !== null) nudge = result;
    }
    expect(nudge).not.toBeNull();
    expect(nudge).toContain("Readonly inspections");
  });

  it("fires NO_PROGRESS_LOOP normally when no plan exists", () => {
    const state = new SessionState();
    const { detector } = makeDetector(state);
    const calls = [makeReadonlyCall()];

    // Drive 1 snapshot-init call + 5 stagnant calls = 6 calls to reach threshold.
    let nudge: string | null = null;
    for (let i = 0; i < 6; i++) {
      const result = detector.maybeInjectNoProgressNudge(calls);
      if (result !== null) nudge = result;
    }
    expect(nudge).not.toBeNull();
  });

  it("suppresses NO_PROGRESS_LOOP when all steps are done even if tool fatigue is also active", () => {
    const state = new SessionState();
    state.setPlan([
      { description: "Step 1", status: "completed" },
    ]);
    const { detector } = makeDetector(state);

    // Activate fatigue for a tool
    for (let i = 0; i < 6; i++) {
      detector.recordToolResult("run_command", { cmd: `attempt-${i}` }, false);
    }
    detector.checkToolFatigue([{ name: "run_command", arguments: {}, callId: "c" }]);

    // Phase guard must still suppress despite fatigue guard also being active
    const calls = [makeReadonlyCall()];
    for (let i = 0; i < 6; i++) {
      expect(detector.maybeInjectNoProgressNudge(calls)).toBeNull();
    }
  });
});

describe("StagnationDetector — tool fatigue suppression", () => {
  it("suppresses NO_PROGRESS_LOOP while TOOL_FATIGUE is active", () => {
    const state = new SessionState();
    const { detector } = makeDetector(state);
    const readonlyCalls = [makeReadonlyCall()];

    // Trigger tool fatigue: 5+ failures of the same tool
    for (let i = 0; i < 6; i++) {
      detector.recordToolResult("run_command", { cmd: `attempt-${i}` }, false);
    }
    // Activate the fatigue warning
    const fatigueWarning = detector.checkToolFatigue([
      { name: "run_command", arguments: {}, callId: "c" },
    ]);
    expect(fatigueWarning).not.toBeNull();

    // Drive readonly cycles — should NOT fire stagnation while fatigued.
    // Drive more than enough cycles (threshold is 5 + 1 for snapshot init = 6).
    let nudge: string | null = null;
    for (let i = 0; i < 8; i++) {
      const result = detector.maybeInjectNoProgressNudge(readonlyCalls);
      if (result !== null) nudge = result;
    }
    expect(nudge).toBeNull();
  });

  it("re-enables NO_PROGRESS_LOOP after tool fatigue resets on success", () => {
    const state = new SessionState();
    const { detector } = makeDetector(state);
    const readonlyCalls = [makeReadonlyCall()];

    // Trigger and activate fatigue
    for (let i = 0; i < 6; i++) {
      detector.recordToolResult("run_command", { cmd: `attempt-${i}` }, false);
    }
    detector.checkToolFatigue([{ name: "run_command", arguments: {}, callId: "c" }]);

    // Tool succeeds — resets fatigue tracking
    detector.recordToolResult("run_command", {}, true);

    // Stagnation should fire again after enough readonly cycles.
    // Drive 1 snapshot-init call + 5 stagnant calls = 6 calls to reach threshold.
    let nudge: string | null = null;
    for (let i = 0; i < 6; i++) {
      const result = detector.maybeInjectNoProgressNudge(readonlyCalls);
      if (result !== null) nudge = result;
    }
    expect(nudge).not.toBeNull();
  });
});
