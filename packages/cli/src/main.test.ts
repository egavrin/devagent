import { afterEach, describe, expect, it, vi } from "vitest";

import type { Session } from "@devagent/runtime";
import { loadQueryFromFile, parseArgs, renderHelpText, renderSessionsList, resolveAutoPromptCommandTarget } from "./main.js";

describe("parseArgs", () => {
  it("parses --file <path>", () => {
    expect(parseArgs(["node", "devagent", "--file", "task.md"]))
      .toMatchObject({ file: "task.md", query: null });
  });

  it("parses -f <path> with other flags", () => {
    expect(parseArgs(["node", "devagent", "-f", "task.md", "--full-auto", "--provider", "openai"]))
      .toMatchObject({ file: "task.md", provider: "openai", query: null });
  });

  it("preserves structured subcommand args without flattening quoted values", () => {
    expect(parseArgs(["node", "devagent", "config", "set", "providers.openai.apiKey", "env:OPENAI_API_KEY"]))
      .toMatchObject({
        subcommand: {
          name: "config",
          args: ["set", "providers.openai.apiKey", "env:OPENAI_API_KEY"],
        },
      });
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

  it("describes the TUI and provider-specific env vars", () => {
    const help = renderHelpText();
    expect(help).toContain("Interactive TUI");
    expect(help).toContain("OPENAI_API_KEY");
    expect(help).toContain("ANTHROPIC_API_KEY");
    expect(help).toContain("--max-iterations <n>  Max tool-call iterations (default: 0 (unlimited))");
    expect(help).not.toContain("Interactive mode (REPL)");
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

describe("renderSessionsList", () => {
  it("prints full session ids and the reusable resume hint", () => {
    const sessions: Session[] = [{
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      createdAt: 1,
      updatedAt: 1,
      messages: [],
      metadata: { totalCost: 0.125 },
    }];

    const output = renderSessionsList(sessions);

    expect(output).toContain("12345678-aaaa-bbbb-cccc-1234567890ab");
    expect(output).toContain("--resume <full-id-or-unique-prefix>");
  });
});
