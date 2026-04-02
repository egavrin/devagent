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
    expect(prepared?.rewrittenQuery).toContain("findings first");
    expect(prepared?.preloadedDiffs).toEqual([
      {
        target: { kind: "unstaged" },
        pathFilters: [],
        content: expect.stringContaining("[Pre-loaded local unstaged working tree diff]"),
      },
    ]);
  });

  it("rewrites simplify commands with efficiency coverage", async () => {
    const prepared = await preparePromptCommandQuery("fix the bug, then /simplify the touched code", {
      resolveAutoTarget: async () => ({ kind: "last-commit" }),
      loadDiff: async () => "diff --git a/file.ts b/file.ts",
    });

    expect(prepared?.rewrittenQuery).toContain("Invoke the `simplify` skill");
    expect(prepared?.rewrittenQuery).toContain("efficiency or performance");
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
  });

  it("honors no-delegates and skip-tests directives explicitly", async () => {
    const prepared = await preparePromptCommandQuery(
      "/simplify staged only packages/runtime no delegates skip tests",
      {
        resolveAutoTarget: async () => ({ kind: "staged" }),
        loadDiff: async () => "diff --git a/packages/runtime/src/x.ts b/packages/runtime/src/x.ts",
      },
    );

    expect(prepared?.rewrittenQuery).toContain("Do not spawn reviewer delegates for this step");
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
});

describe("createPromptCommandFinalTextValidator", () => {
  it("accepts review output with structured findings", () => {
    const validator = createPromptCommandFinalTextValidator([
      {
        name: "review",
        body: "",
        rawBody: "",
        targetHint: { kind: "staged" },
        target: { kind: "staged" },
        pathFilters: [],
        focusAreas: [],
        delegatePreference: "auto",
        verificationPreference: "normal",
        start: 0,
        end: 7,
      },
    ]);

    expect(
      validator?.([
        "Findings",
        "[warning] packages/cli/src/main.ts:2263 - The staged diff preload path can be skipped when path filters hide all unstaged files.",
        "",
        "Open Questions / Assumptions",
        "None.",
        "",
        "Summary",
        "1 warning.",
      ].join("\n")),
    ).toEqual({ valid: true });
  });

  it("rejects malformed review findings", () => {
    const validator = createPromptCommandFinalTextValidator([
      {
        name: "review",
        body: "",
        rawBody: "",
        targetHint: { kind: "staged" },
        target: { kind: "staged" },
        pathFilters: [],
        focusAreas: [],
        delegatePreference: "auto",
        verificationPreference: "normal",
        start: 0,
        end: 7,
      },
    ]);

    expect(
      validator?.([
        "Findings",
        "There might be a bug somewhere.",
        "",
        "Summary",
        "Looks risky.",
      ].join("\n")),
    ).toMatchObject({
      valid: false,
      retryMessage: expect.stringContaining("Findings"),
    });
  });
});
