import { afterEach, describe, expect, it, vi } from "vitest";

import { loadQueryFromFile, parseArgs, renderHelpText, resolveAutoPromptCommandTarget } from "./main.js";

describe("parseArgs", () => {
  it("parses --file <path>", () => {
    expect(parseArgs(["node", "devagent", "--file", "task.md"]))
      .toMatchObject({ file: "task.md", query: null });
  });

  it("parses -f <path> with other flags", () => {
    expect(parseArgs(["node", "devagent", "-f", "task.md", "--full-auto", "--provider", "openai"]))
      .toMatchObject({ file: "task.md", provider: "openai", query: null });
  });
});

describe("loadQueryFromFile", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("reads file contents and trims surrounding whitespace", () => {
    const readFileSync = vi.fn(() => "\n  hello\nworld  \n");

    expect(loadQueryFromFile("task.md", readFileSync)).toBe("hello\nworld");
    expect(readFileSync).toHaveBeenCalledWith("task.md", "utf-8");
  });

  it("fails when both --file and inline query are provided", () => {
    expect(() => loadQueryFromFile("task.md", vi.fn(), "inline query"))
      .toThrow("Cannot specify both --file and an inline query");
  });

  it("fails when the input file does not exist", () => {
    const readFileSync = vi.fn(() => {
      throw new Error("ENOENT: no such file or directory");
    });

    expect(() => loadQueryFromFile("missing.md", readFileSync))
      .toThrow("Input file not found: missing.md");
  });

  it("fails when the input file is empty", () => {
    expect(() => loadQueryFromFile("empty.md", vi.fn(() => "  \n\t  ")))
      .toThrow("Input file is empty: empty.md");
  });
});

describe("renderHelpText", () => {
  it("includes the file flag", () => {
    expect(renderHelpText()).toContain("-f, --file <path>    Read query from file");
  });

  it("includes devagent-api in the provider list", () => {
    expect(renderHelpText()).toContain("devagent-api");
  });

  it("does not advertise removed interactive or plan surfaces", () => {
    const help = renderHelpText();
    expect(help).not.toContain("devagent chat");
    expect(help).not.toContain("--plan");
    expect(help).not.toContain("session inspect");
  });
});

describe("resolveAutoPromptCommandTarget", () => {
  it("falls back to last-commit when the repo is clean and no path filters are provided", () => {
    const runner = vi.fn(() => "");

    expect(resolveAutoPromptCommandTarget("/repo", [], runner)).toEqual({ kind: "last-commit" });
    expect(runner).toHaveBeenNthCalledWith(1, "git diff --name-only", "/repo");
    expect(runner).toHaveBeenNthCalledWith(2, "git diff --cached --name-only", "/repo");
  });

  it("stays on local scope when path-filtered diffs are empty", () => {
    const runner = vi.fn(() => "");

    expect(resolveAutoPromptCommandTarget("/repo", ["packages/cli"], runner)).toEqual({ kind: "unstaged" });
    expect(runner).toHaveBeenNthCalledWith(1, "git diff --name-only -- 'packages/cli'", "/repo");
    expect(runner).toHaveBeenNthCalledWith(2, "git diff --cached --name-only -- 'packages/cli'", "/repo");
  });
});
