import { afterEach, describe, expect, it, vi } from "vitest";

import { handleVersionFlag, loadQueryFromFile, parseArgs, renderHelpText } from "./main.js";

describe("handleVersionFlag", () => {
  it("prints version for --version", () => {
    const requireFn = vi.fn(() => ({ version: "0.1.0" }));
    const stdout = { write: vi.fn(() => true) };
    const exit = vi.fn();
    const packageJsonPath = "/tmp/package.json";

    const handled = handleVersionFlag(["node", "devagent", "--version"], {
      requireFn,
      packageJsonPath,
      stdout,
      exit: exit as unknown as (code?: number) => never,
    });

    expect(handled).toBe(true);
    expect(requireFn).toHaveBeenCalledWith(packageJsonPath);
    expect(stdout.write).toHaveBeenCalledWith("devagent 0.1.0\n");
    expect(exit).toHaveBeenCalledWith(0);
  });

  it("prints version for -V", () => {
    const requireFn = vi.fn(() => ({ version: "9.9.9" }));
    const stdout = { write: vi.fn(() => true) };
    const exit = vi.fn();
    const packageJsonPath = "/tmp/package.json";

    const handled = handleVersionFlag(["node", "devagent", "-V"], {
      requireFn,
      packageJsonPath,
      stdout,
      exit: exit as unknown as (code?: number) => never,
    });

    expect(handled).toBe(true);
    expect(requireFn).toHaveBeenCalledWith(packageJsonPath);
    expect(stdout.write).toHaveBeenCalledWith("devagent 9.9.9\n");
    expect(exit).toHaveBeenCalledWith(0);
  });

  it("returns false when no version flag is present", () => {
    const requireFn = vi.fn();
    const stdout = { write: vi.fn(() => true) };
    const exit = vi.fn();

    const handled = handleVersionFlag(["node", "devagent", "chat"], {
      requireFn,
      stdout,
      exit: exit as unknown as (code?: number) => never,
    });

    expect(handled).toBe(false);
    expect(requireFn).not.toHaveBeenCalled();
    expect(stdout.write).not.toHaveBeenCalled();
    expect(exit).not.toHaveBeenCalled();
  });
});

describe("parseArgs", () => {
  it("parses --file <path>", () => {
    expect(parseArgs(["node", "devagent", "--file", "task.md"]))
      .toMatchObject({ file: "task.md", query: null, interactive: false });
  });

  it("parses -f <path> with other flags", () => {
    expect(parseArgs(["node", "devagent", "-f", "task.md", "--full-auto", "--provider", "openai"]))
      .toMatchObject({ file: "task.md", provider: "openai", query: null, interactive: false });
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

  it("includes the version flag", () => {
    expect(renderHelpText()).toContain("-V, --version         Show CLI version");
  });
});
