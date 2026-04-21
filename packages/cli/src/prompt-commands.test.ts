import { describe, expect, it } from "vitest";

import {
  createPromptCommandFinalTextValidator,
  parsePromptCommandQuery,
  preparePromptCommandQuery,
  resolvePromptCommandTargetHint,
} from "./prompt-commands.js";

describe("parsePromptCommandQuery", () => {
  it("finds embedded commands in the middle of a query", () => {
    const parsed = parsePromptCommandQuery("please /review these changes");

    expect(parsed).not.toBeNull();
    expect(parsed?.leadingText).toBe("please");
    expect(parsed?.segments).toMatchObject([
      {
        name: "review",
        body: "these changes",
        targetHint: { kind: "auto" },
      },
    ]);
  });

  it("parses multiple command segments in order", () => {
    const parsed = parsePromptCommandQuery("/review staged and then /simplify the runtime changes");

    expect(parsed?.segments).toMatchObject([
      {
        name: "review",
        body: "staged and then",
        targetHint: { kind: "staged" },
      },
      {
        name: "simplify",
        body: "the runtime changes",
        targetHint: { kind: "auto" },
      },
    ]);
  });

  it("detects last-commit scope hints", () => {
    const parsed = parsePromptCommandQuery("please /review last-commit only");

    expect(parsed?.segments[0]?.targetHint).toEqual({ kind: "last-commit" });
  });

  it("detects commit ref scope hints", () => {
    expect(resolvePromptCommandTargetHint("review commit abc123")).toEqual({
      kind: "commit",
      ref: "abc123",
    });
  });

  it("extracts path filters and structured flags", () => {
    const parsed = parsePromptCommandQuery(
      "please /simplify staged only packages/cli/src/main.ts focus on perf with no delegates and skip tests",
    );

    expect(parsed?.segments).toMatchObject([
      {
        name: "simplify",
        pathFilters: ["packages/cli/src/main.ts"],
        focusAreas: ["performance", "tests"],
        delegatePreference: "forbid",
        verificationPreference: "skip",
      },
    ]);
  });

  it("supports multiple embedded commands with segment-local filters", () => {
    const parsed = parsePromptCommandQuery(
      "please /review staged in packages/cli and then /simplify unstaged in packages/runtime",
    );

    expect(parsed?.segments).toMatchObject([
      {
        name: "review",
        targetHint: { kind: "staged" },
        pathFilters: ["packages/cli"],
      },
      {
        name: "simplify",
        targetHint: { kind: "unstaged" },
        pathFilters: ["packages/runtime"],
      },
    ]);
  });
});
describe("preparePromptCommandQuery", () => {
  it("rewrites review commands into a skill-driven local workflow", async () => {
    const prepared = await preparePromptCommandQuery("please /review these changes", {
      resolveAutoTarget: async () => ({ kind: "unstaged" }),
      loadDiff: async () => "diff --git a/file.ts b/file.ts",
    });

    expect(prepared).not.toBeNull();
    expect(prepared?.rewrittenQuery).toContain("Invoke the `review` skill");
    expect(prepared?.rewrittenQuery).toContain("Launch three readonly `reviewer` delegates before concluding.");
    expect(prepared?.rewrittenQuery).toContain("correctness/regressions, tests/contracts, and performance/fail-fast risks");
    expect(prepared?.rewrittenQuery).toContain("blocking defects vs non-blocking suggestions");
    expect(prepared?.rewrittenQuery).toContain("Do not narrate your process");
    expect(prepared?.rewrittenQuery).toContain("Start immediately with `Blocking Findings`");
    expect(prepared?.rewrittenQuery).toContain("Blocking Findings, Non-blocking Suggestions, Open Questions / Assumptions, Short Summary");
    expect(prepared?.rewrittenQuery).toContain("Overall: improves code health");
    expect(prepared?.preloadedDiffs).toEqual([
      {
        target: { kind: "unstaged" },
        pathFilters: [],
        content: expect.stringContaining("[Pre-loaded local unstaged working tree diff]"),
      },
    ]);
  });

  it("rewrites simplify commands with minimization and surface-challenge coverage", async () => {
    const prepared = await preparePromptCommandQuery("fix the bug, then /simplify the touched code", {
      resolveAutoTarget: async () => ({ kind: "last-commit" }),
      loadDiff: async () => "diff --git a/file.ts b/file.ts",
    });

    expect(prepared?.rewrittenQuery).toContain("Invoke the `simplify` skill");
    expect(prepared?.rewrittenQuery).toContain("explicit minimization or deletion");
    expect(prepared?.rewrittenQuery).toContain("Challenge newly added surface area");
    expect(prepared?.rewrittenQuery).toContain("new files, exports, config keys, tests, and docs");
    expect(prepared?.resolvedSegments[0]?.target).toEqual({ kind: "last-commit" });
  });

  it("preserves mixed command order and loads each target once", async () => {
    const prepared = await preparePromptCommandQuery(
      "/review staged and then /simplify the runtime changes and /review staged",
      {
        resolveAutoTarget: async () => ({ kind: "unstaged" }),
        loadDiff: async (target) => `diff for ${target.kind}`,
      },
    );

    expect(prepared?.rewrittenQuery.indexOf("1. `/review`")).toBeLessThan(
      prepared?.rewrittenQuery.indexOf("2. `/simplify`") ?? Number.MAX_SAFE_INTEGER,
    );
    expect(prepared?.rewrittenQuery.indexOf("2. `/simplify`")).toBeLessThan(
      prepared?.rewrittenQuery.indexOf("3. `/review`") ?? Number.MAX_SAFE_INTEGER,
    );
    expect(prepared?.preloadedDiffs).toEqual([
      {
        target: { kind: "staged" },
        pathFilters: [],
        content: expect.stringContaining("diff for staged"),
      },
      {
        target: { kind: "unstaged" },
        pathFilters: [],
        content: expect.stringContaining("diff for unstaged"),
      },
    ]);
  });

  it("keeps small simplify scopes local by default", async () => {
    const prepared = await preparePromptCommandQuery("/simplify only packages/cli/src/foo.ts", {
      resolveAutoTarget: async () => ({ kind: "unstaged" }),
      loadDiff: async () => [
        "diff --git a/packages/cli/src/foo.ts b/packages/cli/src/foo.ts",
        "@@ -1,2 +1,2 @@",
        "-old",
        "+new",
      ].join("\n"),
    });

    expect(prepared?.rewrittenQuery).toContain("Stay local for this step");
    expect(prepared?.rewrittenQuery).not.toContain("Launch three readonly `reviewer` delegates");
  });

  it("forces parallel simplify lanes for larger scopes", async () => {
    const prepared = await preparePromptCommandQuery("/simplify staged", {
      resolveAutoTarget: async () => ({ kind: "staged" }),
      loadDiff: async () => [
        "diff --git a/a.ts b/a.ts",
        "@@ -1,2 +1,60 @@",
        ...Array.from({ length: 70 }, (_, index) => `+line ${index + 1}`),
        "diff --git a/b.ts b/b.ts",
        "@@ -1,2 +1,60 @@",
        ...Array.from({ length: 70 }, (_, index) => `+other ${index + 1}`),
      ].join("\n"),
    });

    expect(prepared?.rewrittenQuery).toContain("Launch three readonly `reviewer` delegates");
    expect(prepared?.rewrittenQuery).toContain("remove code aggressively in the main agent before refactoring what remains");
  });
});

describe("preparePromptCommandQuery explicit preferences", () => {
  it("honors no-delegates and skip-tests directives explicitly", async () => {
    const prepared = await preparePromptCommandQuery(
      "/simplify staged only packages/runtime no delegates skip tests",
      {
        resolveAutoTarget: async () => ({ kind: "staged" }),
        loadDiff: async () => "diff --git a/packages/runtime/src/x.ts b/packages/runtime/src/x.ts",
      },
    );

    expect(prepared?.rewrittenQuery).toContain("Do not spawn reviewer delegates for this step");
    expect(prepared?.rewrittenQuery).toContain("skeptical, and deletion-first");
    expect(prepared?.rewrittenQuery).toContain("Skip verification commands unless the user later asks for them");
  });

  it("supports explicit commit refs and path-scoped preloads", async () => {
    const prepared = await preparePromptCommandQuery(
      "/review commit abc123 only packages/cli/src/main.ts",
      {
        resolveAutoTarget: async () => ({ kind: "unstaged" }),
        loadDiff: async (target, pathFilters) => `${target.kind}:${"ref" in target ? target.ref : ""}:${pathFilters.join(",")}`,
      },
    );

    expect(prepared?.resolvedSegments[0]?.target).toEqual({ kind: "commit", ref: "abc123" });
    expect(prepared?.resolvedSegments[0]?.pathFilters).toEqual(["packages/cli/src/main.ts"]);
    expect(prepared?.preloadedDiffs[0]?.content).toContain("commit abc123 diff");
    expect(prepared?.preloadedDiffs[0]?.content).toContain("packages/cli/src/main.ts");
  });

  it("does not attach the review-only validator to mixed command pipelines", async () => {
    const prepared = await preparePromptCommandQuery("/review staged then /simplify staged", {
      resolveAutoTarget: async () => ({ kind: "staged" }),
      loadDiff: async () => "diff --git a/file.ts b/file.ts",
    });

    expect(prepared?.finalTextValidator).toBeUndefined();
  });

  it("honors explicit no-delegates directives for review", async () => {
    const prepared = await preparePromptCommandQuery("/review staged no delegates", {
      resolveAutoTarget: async () => ({ kind: "staged" }),
      loadDiff: async () => "diff --git a/file.ts b/file.ts",
    });

    expect(prepared?.rewrittenQuery).toContain("Do not spawn reviewer delegates for this step unless you need targeted independent evidence.");
    expect(prepared?.rewrittenQuery).not.toContain("Launch three readonly `reviewer` delegates before concluding.");
  });
});
const reviewSegment = {
  name: "review" as const,
  body: "",
  rawBody: "",
  targetHint: { kind: "staged" } as const,
  target: { kind: "staged" } as const,
  pathFilters: [],
  focusAreas: [],
  delegatePreference: "auto" as const,
  verificationPreference: "normal" as const,
  start: 0,
  end: 7,
};

describe("createPromptCommandFinalTextValidator success cases", () => {
  it("accepts review output with blocking and non-blocking sections", () => {
    const validator = createPromptCommandFinalTextValidator([
      reviewSegment,
    ]);

    expect(
      validator?.([
        "Blocking Findings",
        "[warning] packages/cli/src/main.ts:2263 - The staged diff preload path can be skipped when path filters hide all unstaged files.",
        "",
        "Non-blocking Suggestions",
        "- packages/cli/src/main.ts:2269 - Consider naming the preload key helper after the diff scope to make the workflow easier to scan.",
        "",
        "Open Questions / Assumptions",
        "None.",
        "",
        "Short Summary",
        "Overall: does not improve code health until the staged preload path is fixed.",
      ].join("\n")),
    ).toEqual({ valid: true });
  });

  it("accepts explicit none sections", () => {
    const validator = createPromptCommandFinalTextValidator([
      reviewSegment,
    ]);

    expect(
      validator?.([
        "Blocking Findings",
        "None.",
        "",
        "Non-blocking Suggestions",
        "None.",
        "",
        "Open Questions / Assumptions",
        "None.",
        "",
        "Short Summary",
        "Overall: improves code health. No blocking issues found.",
      ].join("\n")),
    ).toEqual({ valid: true });
  });
});

describe("createPromptCommandFinalTextValidator", () => {
  it("rejects malformed review findings", () => {
    const validator = createPromptCommandFinalTextValidator([
      reviewSegment,
    ]);

    expect(
      validator?.([
        "Blocking Findings",
        "There might be a bug somewhere.",
        "",
        "Non-blocking Suggestions",
        "None.",
        "",
        "Open Questions / Assumptions",
        "None.",
        "",
        "Short Summary",
        "Overall: does not improve code health.",
      ].join("\n")),
    ).toMatchObject({
      valid: false,
      retryMessage: expect.stringContaining("Blocking Findings"),
    });
  });

  it("rejects review output when sections are out of order", () => {
    const validator = createPromptCommandFinalTextValidator([
      reviewSegment,
    ]);

    expect(
      validator?.([
        "Non-blocking Suggestions",
        "None.",
        "",
        "Blocking Findings",
        "None.",
        "",
        "Open Questions / Assumptions",
        "None.",
        "",
        "Short Summary",
        "Overall: improves code health.",
      ].join("\n")),
    ).toMatchObject({
      valid: false,
      retryMessage: expect.stringContaining("exactly four sections"),
    });
  });

  it("rejects review output with a preamble before Blocking Findings", () => {
    const validator = createPromptCommandFinalTextValidator([
      reviewSegment,
    ]);

    expect(
      validator?.([
        "Reviewing touched code now.",
        "",
        "Blocking Findings",
        "None.",
        "",
        "Non-blocking Suggestions",
        "None.",
        "",
        "Open Questions / Assumptions",
        "None.",
        "",
        "Short Summary",
        "Overall: improves code health.",
      ].join("\n")),
    ).toMatchObject({
      valid: false,
      retryMessage: expect.stringContaining("Start immediately with `Blocking Findings`"),
    });
  });

  it("rejects review output without a code-health verdict", () => {
    const validator = createPromptCommandFinalTextValidator([
      reviewSegment,
    ]);

    expect(
      validator?.([
        "Blocking Findings",
        "None.",
        "",
        "Non-blocking Suggestions",
        "None.",
        "",
        "Open Questions / Assumptions",
        "None.",
        "",
        "Short Summary",
        "No blocking issues found.",
      ].join("\n")),
    ).toMatchObject({
      valid: false,
      retryMessage: expect.stringContaining("Overall: ... code health"),
    });
  });
});
