import { describe, expect, it } from "vitest";
import { AgentType } from "../core/index.js";
import {
  buildExplorationLaneRequest,
  buildDelegationQuery,
  parseStructuredAgentOutput,
} from "./subagent-contract.js";

describe("parseStructuredAgentOutput", () => {
  it("parses pure JSON output", () => {
    const parsed = parseStructuredAgentOutput(
      AgentType.EXPLORE,
      JSON.stringify({
        answer: "Found it",
        evidence: ["src/a.ts:1"],
        relatedFiles: ["src/a.ts"],
        unresolved: [],
      }),
    );

    expect(parsed).toMatchObject({ answer: "Found it" });
  });

  it("parses fenced JSON output", () => {
    const parsed = parseStructuredAgentOutput(
      AgentType.GENERAL,
      "```json\n{\"summary\":\"done\",\"filesTouched\":[\"a.ts\"],\"checksRun\":[],\"unresolved\":[]}\n```",
    );

    expect(parsed).toMatchObject({ summary: "done" });
  });

  it("parses leading JSON followed by prose", () => {
    const parsed = parseStructuredAgentOutput(
      AgentType.REVIEWER,
      "{\"findings\":[],\"openQuestions\":[],\"summary\":\"clean\"}\n\nNo correctness issues found.",
    );

    expect(parsed).toMatchObject({ summary: "clean" });
  });

  it("returns null for invalid JSON", () => {
    const parsed = parseStructuredAgentOutput(
      AgentType.ARCHITECT,
      "{not valid json}\n\nPlan follows.",
    );

    expect(parsed).toBeNull();
  });
});

describe("buildDelegationQuery", () => {
  it("builds exploration lane requests with explicit scope boundaries", () => {
    const request = buildExplorationLaneRequest({
      objective: "Inspect frontend compile-time restrictions",
      laneLabel: "frontend / compile-time behavior",
      scope: "arkcompiler_ets_frontend only",
      exclusions: ["runtime/tests behavior"],
      successCriteria: ["Return exact error sites"],
      parentContext: "This is one evidence lane in a broader contradiction analysis.",
    });

    expect(request).toEqual({
      objective: "Inspect frontend compile-time restrictions",
      laneLabel: "frontend / compile-time behavior",
      scope: "arkcompiler_ets_frontend only",
      exclusions: ["runtime/tests behavior"],
      successCriteria: ["Return exact error sites"],
      parentContext: "This is one evidence lane in a broader contradiction analysis.",
    });
  });

  it("includes strong delegation context and hard constraints", () => {
    const query = buildDelegationQuery({
      objective: "Inspect verifier code paths",
      laneLabel: "runtime/tests lane",
      scope: "runtime verifier only",
      constraints: ["Read-only", "Do not inspect unrelated subsystems"],
      exclusions: ["docs/spec claims", "frontend behavior"],
      successCriteria: ["Return grounded findings"],
      parentContext: "The parent must use a delegated explore pass for validation coverage.",
    }, 12);

    expect(query).toContain("This task was intentionally delegated");
    expect(query).toContain("Lane: runtime/tests lane");
    expect(query).toContain("Delegated because: The parent must use a delegated explore pass for validation coverage.");
    expect(query).toContain("Treat the iteration budget, scope, constraints, and success criteria as hard limits.");
    expect(query).toContain("Iteration budget: 12");
    expect(query).toContain("Out of scope:");
    expect(query).toContain("docs/spec claims");
  });
});
