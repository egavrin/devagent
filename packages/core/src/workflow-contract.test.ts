import { describe, it, expect } from "vitest";
import {
  isValidPhase,
  isValidApprovalMode,
  isValidReasoningLevel,
  WORKFLOW_PHASES,
  WORKFLOW_APPROVAL_MODES,
  REASONING_LEVELS,
} from "./workflow-contract.js";

describe("isValidPhase", () => {
  it("accepts all valid phases", () => {
    for (const phase of WORKFLOW_PHASES) {
      expect(isValidPhase(phase)).toBe(true);
    }
  });

  it("rejects invalid phases", () => {
    expect(isValidPhase("analyze")).toBe(false);
    expect(isValidPhase("understand")).toBe(false);
    expect(isValidPhase("")).toBe(false);
    expect(isValidPhase("TRIAGE")).toBe(false);
  });
});

describe("isValidApprovalMode", () => {
  it("accepts all valid modes", () => {
    for (const mode of WORKFLOW_APPROVAL_MODES) {
      expect(isValidApprovalMode(mode)).toBe(true);
    }
  });

  it("rejects invalid modes", () => {
    expect(isValidApprovalMode("auto")).toBe(false);
    expect(isValidApprovalMode("manual")).toBe(false);
    expect(isValidApprovalMode("FULL_AUTO")).toBe(false);
  });
});

describe("isValidReasoningLevel", () => {
  it("accepts all valid levels", () => {
    for (const level of REASONING_LEVELS) {
      expect(isValidReasoningLevel(level)).toBe(true);
    }
  });

  it("rejects invalid levels", () => {
    expect(isValidReasoningLevel("ultra")).toBe(false);
    expect(isValidReasoningLevel("none")).toBe(false);
    expect(isValidReasoningLevel("HIGH")).toBe(false);
  });
});
