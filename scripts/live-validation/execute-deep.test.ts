import { describe, expect, it } from "bun:test";
import {
  buildScenarioGroups,
  computeReleaseRecommendation,
  computeRunStatus,
  parseArgs,
  renderExecuteDeepMarkdown,
} from "./execute-deep";
import type {
  ExecuteDeepSummary,
  ExecuteDeepGroupId,
} from "./execute-deep";
import type {
  AggregateValidationSummary,
  ValidationScenario,
} from "./types";

function makeSummary(
  overrides: Partial<ExecuteDeepSummary> = {},
): ExecuteDeepSummary {
  const emptyGroup = (durationMs = 0): AggregateValidationSummary => ({
    provider: "chatgpt",
    model: "gpt-5.4",
    suite: "scenario",
    total: 0,
    passed: 0,
    failed: 0,
    blocked: 0,
    durationMs,
    reports: [],
  });

  return {
    provider: "chatgpt",
    model: "gpt-5.4",
    outputRoot: "/tmp/execute-deep",
    selectedGroups: ["canonical", "continuity", "remainder"],
    coverageMode: "full",
    prerequisiteResults: [],
    canonicalFlow: emptyGroup(10),
    continuityChecks: emptyGroup(20),
    fullSuiteRemainder: emptyGroup(30),
    timing: {
      totalDurationMs: 60,
      prerequisiteDurationMs: 0,
      groupDurationMs: {
        canonical: 10,
        continuity: 20,
        remainder: 30,
      },
    },
    runStatus: "passed",
    releaseRecommendation: "ship",
    ...overrides,
  };
}

function makeScenario(
  id: string,
  suites: ReadonlyArray<"smoke" | "full">,
): ValidationScenario {
  return {
    id,
    description: id,
    suites,
    targetRepo: "arkcompiler_runtime_core_docs",
    surface: "execute",
    taskShape: "readonly",
    isolationMode: "temp-copy",
    invocation: {
      type: "execute",
      taskType: "plan",
      workItemTitle: id,
      summary: id,
    },
    expectedArtifacts: [],
    assertions: [],
    verificationCommands: [],
    cleanupPolicy: "destroy",
  };
}

describe("parseArgs", () => {
  it("selects every group by default", () => {
    const options = parseArgs(["bun", "execute-deep.ts"]);
    expect(options.selectedGroups).toEqual(["canonical", "continuity", "remainder"]);
  });

  it("deduplicates repeated --only flags while keeping canonical order", () => {
    const options = parseArgs([
      "bun",
      "execute-deep.ts",
      "--only",
      "remainder",
      "--only",
      "canonical",
      "--only",
      "canonical",
    ]);

    expect(options.selectedGroups).toEqual(["canonical", "remainder"]);
  });

  it("rejects unknown --only groups", () => {
    expect(() => parseArgs([
      "bun",
      "execute-deep.ts",
      "--only",
      "bad-group",
    ])).toThrow(/Invalid --only group/i);
  });
});

describe("buildScenarioGroups", () => {
  it("keeps remainder separate from canonical and continuity scenarios", () => {
    const scenarios = [
      makeScenario("runtime-core-docs-execute-design", ["full"]),
      makeScenario("runtime-core-docs-execute-breakdown", ["full"]),
      makeScenario("runtime-core-docs-execute-issue-generation", ["full"]),
      makeScenario("runtime-core-execute-plan", ["full"]),
      makeScenario("runtime-core-execute-triage", ["full"]),
      makeScenario("ets-frontend-execute-repair", ["full"]),
      makeScenario("doctor-provider-model-mismatch", ["full"]),
      makeScenario("runtime-core-cli-review", ["full"]),
    ];

    const groups = buildScenarioGroups(scenarios);
    expect(groups.canonical).toEqual([
      "runtime-core-docs-execute-design",
      "runtime-core-docs-execute-breakdown",
      "runtime-core-docs-execute-issue-generation",
    ]);
    expect(groups.continuity).toEqual([
      "runtime-core-execute-plan",
      "runtime-core-execute-triage",
      "ets-frontend-execute-repair",
    ]);
    expect(groups.remainder).toEqual([
      "doctor-provider-model-mismatch",
      "runtime-core-cli-review",
    ]);
  });
});

describe("run status and release recommendation", () => {
  it("marks successful focused runs as partial recommendations", () => {
    const summary = makeSummary({
      selectedGroups: ["canonical"],
      coverageMode: "partial",
      timing: {
        totalDurationMs: 10,
        prerequisiteDurationMs: 0,
        groupDurationMs: {
          canonical: 10,
          continuity: 0,
          remainder: 0,
        },
      },
      continuityChecks: {
        ...makeSummary().continuityChecks,
        durationMs: 0,
      },
      fullSuiteRemainder: {
        ...makeSummary().fullSuiteRemainder,
        durationMs: 0,
      },
    });

    const runStatus = computeRunStatus(summary);
    expect(runStatus).toBe("passed");
    expect(computeReleaseRecommendation({ ...summary, runStatus })).toBe("partial");
  });

  it("treats prereq failures as blocked", () => {
    const summary = makeSummary({
      prerequisiteResults: [
        {
          label: "build:publish",
          command: "bun run build:publish",
          exitCode: 1,
          stdout: "",
          stderr: "failed",
          durationMs: 100,
        },
      ],
      runStatus: "blocked",
    });

    expect(computeRunStatus(summary)).toBe("blocked");
    expect(computeReleaseRecommendation(summary)).toBe("blocked");
  });

  it("maps failed full runs to fix-before-ship", () => {
    const failingGroup: AggregateValidationSummary = {
      provider: "chatgpt",
      model: "gpt-5.4",
      suite: "scenario",
      total: 1,
      passed: 0,
      failed: 1,
      blocked: 0,
      durationMs: 50,
      reports: [],
    };
    const summary = makeSummary({
      canonicalFlow: failingGroup,
      timing: {
        totalDurationMs: 50,
        prerequisiteDurationMs: 0,
        groupDurationMs: {
          canonical: 50,
          continuity: 20,
          remainder: 30,
        },
      },
      runStatus: "failed",
    });

    expect(computeRunStatus(summary)).toBe("failed");
    expect(computeReleaseRecommendation(summary)).toBe("fix-before-ship");
  });
});

describe("renderExecuteDeepMarkdown", () => {
  it("renders partial coverage and timing details", () => {
    const selectedGroups: ReadonlyArray<ExecuteDeepGroupId> = ["canonical", "remainder"];
    const summary = makeSummary({
      selectedGroups,
      coverageMode: "partial",
      continuityChecks: {
        ...makeSummary().continuityChecks,
        durationMs: 0,
      },
      timing: {
        totalDurationMs: 42,
        prerequisiteDurationMs: 12,
        groupDurationMs: {
          canonical: 10,
          continuity: 0,
          remainder: 20,
        },
      },
      releaseRecommendation: "partial",
    });

    const markdown = renderExecuteDeepMarkdown(summary);
    expect(markdown).toContain("- Coverage mode: partial");
    expect(markdown).toContain("- Selected groups: canonical, remainder");
    expect(markdown).toContain("This packet covers a focused subset");
    expect(markdown).toContain("- Duration: 10 ms");
    expect(markdown).toContain("- Selected: no");
    expect(markdown).toContain("- Not selected in this run.");
  });
});
