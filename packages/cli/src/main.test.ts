import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { Session } from "@devagent/runtime";
import { loadQueryFromFile, parseArgs, renderHelpText, renderSessionsList, resolveAutoPromptCommandTarget } from "./main.js";

const cliSrcDir = dirname(fileURLToPath(import.meta.url));
const cliPackageDir = join(cliSrcDir, "..");

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

  it("recognizes configure and keeps setup as a retired command", () => {
    expect(parseArgs(["node", "devagent", "configure"]))
      .toMatchObject({
        subcommand: {
          name: "configure",
          args: [],
        },
      });

    expect(parseArgs(["node", "devagent", "setup", "--help"]))
      .toMatchObject({
        subcommand: {
          name: "setup",
          args: ["--help"],
        },
      });
  });

  it("recognizes bare help as a top-level help command", () => {
    expect(parseArgs(["node", "devagent", "help"]))
      .toMatchObject({
        subcommand: {
          name: "help",
          args: [],
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

  it("presents configure as the public onboarding command", () => {
    const help = renderHelpText();
    expect(help).toContain("devagent configure");
    expect(help).toContain("Inspect or edit global config directly");
    expect(help).not.toContain("devagent setup");
    expect(help).not.toContain("devagent init");
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

  it("matches the built CLI help output after build", () => {
    execFileSync("bun", ["run", "build"], {
      cwd: cliPackageDir,
      stdio: "pipe",
    });

    const builtHelp = execFileSync("bun", ["dist/index.js", "--help"], {
      cwd: cliPackageDir,
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "pipe"],
    });

    expect(builtHelp.trim()).toBe(renderHelpText().trim());
  });

  it("matches the built CLI help alias output after build", () => {
    execFileSync("bun", ["run", "build"], {
      cwd: cliPackageDir,
      stdio: "pipe",
    });

    const builtHelp = execFileSync("bun", ["dist/index.js", "help"], {
      cwd: cliPackageDir,
      encoding: "utf-8",
      stdio: ["ignore", "pipe", "pipe"],
    });

    expect(builtHelp.trim()).toBe(renderHelpText().trim());
  });

  it("lets the help alias succeed when config references an unset env var", () => {
    execFileSync("bun", ["run", "build"], {
      cwd: cliPackageDir,
      stdio: "pipe",
    });

    const home = mkdtempSync(join(tmpdir(), "devagent-help-home-"));
    try {
      const configDir = join(home, ".config", "devagent");
      const configPath = join(configDir, "config.toml");
      mkdirSync(configDir, { recursive: true });
      writeFileSync(configPath, 'provider = "openai"\n\n[providers.openai]\napi_key = "env:OPENAI_API_KEY"\n');

      const builtHelp = execFileSync("bun", ["dist/index.js", "help"], {
        cwd: cliPackageDir,
        encoding: "utf-8",
        stdio: ["ignore", "pipe", "pipe"],
        env: {
          ...process.env,
          HOME: home,
        },
      });

      expect(builtHelp.trim()).toBe(renderHelpText().trim());
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
  });

  it("rejects extra arguments after the help alias with usage guidance", () => {
    execFileSync("bun", ["run", "build"], {
      cwd: cliPackageDir,
      stdio: "pipe",
    });

    try {
      execFileSync("bun", ["dist/index.js", "help", "config"], {
        cwd: cliPackageDir,
        encoding: "utf-8",
        stdio: ["ignore", "pipe", "pipe"],
      });
      expect.unreachable("expected help alias with extra args to fail");
    } catch (error) {
      const execError = error as Error & { stdout?: string; stderr?: string; status?: number };
      expect(execError.status).toBe(2);
      expect(execError.stderr).toContain('Usage: devagent help');
      expect(execError.stderr).toContain(renderHelpText().trim());
      expect(execError.stderr).not.toContain('Environment variable "OPENAI_API_KEY" referenced in config but not set');
    }
  });

  it("reports the selected provider when an inactive config env ref is unset", () => {
    execFileSync("bun", ["run", "build"], {
      cwd: join(cliPackageDir, "..", "runtime"),
      stdio: "pipe",
    });

    execFileSync("bun", ["run", "build"], {
      cwd: cliPackageDir,
      stdio: "pipe",
    });

    const home = mkdtempSync(join(tmpdir(), "devagent-provider-override-home-"));
    try {
      const configDir = join(home, ".config", "devagent");
      const configPath = join(configDir, "config.toml");
      mkdirSync(configDir, { recursive: true });
      writeFileSync(
        configPath,
        'provider = "openai"\n\n[providers.openai]\napi_key = "env:OPENAI_API_KEY"\n',
      );

      try {
        execFileSync("bun", ["dist/index.js", "--provider", "devagent-api", "--model", "cortex", "test"], {
          cwd: cliPackageDir,
          encoding: "utf-8",
          stdio: ["ignore", "pipe", "pipe"],
          env: {
            ...process.env,
            HOME: home,
          },
        });
        expect.unreachable("expected provider setup to fail without a devagent-api key");
      } catch (error) {
        const execError = error as Error & { stdout?: string; stderr?: string; status?: number };
        expect(execError.status).toBe(1);
        expect(execError.stderr).toContain('No API key configured for provider "devagent-api".');
        expect(execError.stderr).not.toContain('Environment variable "OPENAI_API_KEY" referenced in config but not set');
      }
    } finally {
      rmSync(home, { recursive: true, force: true });
    }
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
