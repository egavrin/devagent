import { describe, it, expect } from "vitest";

import { ToolError } from "../core/index.js";

/**
 * We test the git tool safety validation by importing the internal
 * assertSafeArg logic indirectly through the exported tool handlers.
 *
 * The git_diff tool validates both `path` and `ref` params via assertSafeArg
 * and also rejects refs starting with `-`.
 */
import { gitDiffTool, gitCommitTool } from "./builtins/git.js";

// Minimal context stub for tool handlers
const stubContext = {
  repoRoot: "/tmp/nonexistent-repo",
  bus: {} as never,
  config: {} as never,
  approvalGate: {} as never,
  sessionState: {} as never,
};

describe("git tool safety: assertSafeArg", () => {
  const shellMetachars = [";", "&", "|", "`", "$", "<", ">", "\n", "\r", "\0"];

  for (const char of shellMetachars) {
    const display = JSON.stringify(char);

    it(`git_diff rejects ref containing ${display}`, async () => {
      await expect(
        gitDiffTool.handler({ ref: `main${char}evil` }, stubContext as never),
      ).rejects.toThrow(ToolError);
    });

    it(`git_diff rejects path containing ${display}`, async () => {
      await expect(
        gitDiffTool.handler({ path: `file${char}name` }, stubContext as never),
      ).rejects.toThrow(ToolError);
    });
  }

  it("git_diff rejects refs starting with -", async () => {
    await expect(
      gitDiffTool.handler({ ref: "--evil-flag" }, stubContext as never),
    ).rejects.toThrow("option-style refs are not allowed");
  });

  it("git_diff accepts clean ref and path arguments", async () => {
    // This will fail because the repo doesn't exist, but it should NOT
    // fail with a ToolError about disallowed characters or option-style refs.
    // It should fail with a git execution error instead.
    const err: ToolError = await gitDiffTool
      .handler(
        { ref: "HEAD~1", path: "src/index.ts" },
        stubContext as never,
      )
      .then(
        () => {
          throw new Error("expected handler to reject");
        },
        (e: ToolError) => e,
      );
    // Should be a git execution error, not a safety validation error
    expect(err).toBeInstanceOf(ToolError);
    expect(err.message).not.toMatch(/disallowed/);
    expect(err.message).not.toMatch(/option-style/);
  });
});

describe("git_commit: file argument parsing safety", () => {
  const dangerousFiles = ["file;rm -rf /", "file|cat /etc/passwd", "$(evil)", "file`cmd`"];

  for (const file of dangerousFiles) {
    it(`rejects dangerous file arg: ${JSON.stringify(file)}`, async () => {
      await expect(
        gitCommitTool.handler(
          { message: "test", files: file },
          stubContext as never,
        ),
      ).rejects.toThrow(ToolError);
    });
  }
});
