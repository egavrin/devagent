import { mkdtempSync, mkdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, describe, expect, it } from "vitest";

import { buildContextPrompt, loadRepoContext } from "./repo-context.js";

describe("repo-context", () => {
  let repoRoot: string | null = null;

  afterEach(() => {
    if (repoRoot) {
      rmSync(repoRoot, { recursive: true, force: true });
      repoRoot = null;
    }
  });

  it("loads nested instruction files without Bun glob support", () => {
    repoRoot = mkdtempSync(join(tmpdir(), "devagent-repo-context-"));
    const instructionsDir = join(repoRoot, ".github", "instructions", "nested");
    mkdirSync(instructionsDir, { recursive: true });
    writeFileSync(
      join(instructionsDir, "feature.instructions.md"),
      [
        "---",
        "applyTo: src/**/*.ts",
        "---",
        "Prefer focused edits.",
      ].join("\n"),
      "utf-8",
    );

    const context = loadRepoContext(repoRoot);

    expect(context.pathInstructions).toEqual([
      {
        glob: "src/**/*.ts",
        content: "Prefer focused edits.",
        filePath: ".github/instructions/nested/feature.instructions.md",
      },
    ]);
  });

  it("matches double-star globs when building path-specific context", () => {
    const prompt = buildContextPrompt(
      {
        workflowMd: null,
        agentsMd: null,
        copilotInstructions: null,
        pathInstructions: [
          {
            glob: "src/**/*.ts",
            content: "TypeScript-only guidance.",
            filePath: ".github/instructions/ts.instructions.md",
          },
        ],
      },
      ["src/feature/index.ts"],
    );

    expect(prompt).toContain("TypeScript-only guidance.");
  });
});
