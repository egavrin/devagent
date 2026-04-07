import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { EventBus, SessionState } from "@devagent/runtime";
import { afterEach, describe, expect, it, vi } from "vitest";

import type { DevAgentConfig, LLMProvider, Session, SessionStore } from "@devagent/runtime";
import {
  checkForUpdates,
  loadQueryFromFile,
  parseArgs,
  renderHelpText,
  renderSessionsList,
  resolveAutoPromptCommandTarget,
  setupSessionPersistence,
} from "./main.js";

const cliSrcDir = dirname(fileURLToPath(import.meta.url));
const cliPackageDir = join(cliSrcDir, "..");

describe("parseArgs", () => {
  it("parses --file <path>", () => {
    expect(parseArgs(["node", "devagent", "--file", "task.md"]))
      .toMatchObject({ file: "task.md", query: null });
  });

  it("parses -f <path> with --mode and other flags", () => {
    expect(parseArgs(["node", "devagent", "-f", "task.md", "--mode", "default", "--provider", "openai"]))
      .toMatchObject({ file: "task.md", provider: "openai", query: null, safetyMode: "default" });
  });

  it("rejects unknown --mode values", () => {
    expect(parseArgs(["node", "devagent", "--mode", "invalid-mode"]))
      .toMatchObject({ modeParseError: "Invalid --mode value: invalid-mode. Expected one of: default, autopilot." });
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

  it("preserves auth subcommand arguments for non-interactive logout", () => {
    expect(parseArgs(["node", "devagent", "auth", "logout", "chatgpt"]))
      .toMatchObject({
        authCommand: {
          subcommand: "logout",
          args: ["chatgpt"],
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

  it("includes execute as a first-class public command", () => {
    const help = renderHelpText();
    expect(help).toContain("devagent execute --request <file> --artifact-dir <dir>");
    expect(help).toContain("devagent execute");
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
    expect(help).toContain("devagent sessions");
    expect(help).toContain("devagent auth <...>");
    expect(help).toContain("devagent auth logout [provider|--all]");
    expect(help).toContain("devagent install-lsp");
    expect(help).toContain("--mode <mode>        Interactive safety mode: default, autopilot");
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
  it("renders titled session previews with repo context and recency", () => {
    const sessions: Session[] = [{
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      createdAt: 1,
      updatedAt: 60_000,
      messages: [],
      metadata: {
        title: "Fix auth retry loop",
        repoLabel: "devagent",
        totalCost: 0.125,
      },
    }];

    const output = renderSessionsList(sessions, 2 * 60_000);

    expect(output).toContain("Fix auth retry loop");
    expect(output).toContain("12345678  devagent  1m ago  $0.1250");
    expect(output).toContain("--resume <full-id-or-unique-prefix>");
  });

  it("derives a readable title and repo fallback for legacy sessions", () => {
    const sessions: Session[] = [{
      id: "87654321-aaaa-bbbb-cccc-1234567890ab",
      createdAt: 1,
      updatedAt: 10_000,
      messages: [],
      metadata: {
        query: "\"Fix\n failing tests in auth module???\"",
        repoRoot: "/Users/egavrin/Documents/devagent",
      },
    }];

    const output = renderSessionsList(sessions, 90_000);

    expect(output).toContain("Fix failing tests in auth module");
    expect(output).toContain("87654321  devagent  1m ago");
  });
});

function createSessionTestConfig(): DevAgentConfig {
  return {
    provider: "openai",
    model: "gpt-5",
    context: {
      turnIsolation: false,
    },
    logging: {
      enabled: false,
    },
  } as unknown as DevAgentConfig;
}

describe("setupSessionPersistence", () => {
  function createMemorySessionStore(): SessionStore {
    const sessions = new Map<string, Session>();
    const sessionState = new Map<string, Record<string, unknown>>();
    let nextId = 1;

    return {
      createSession(metadata?: Record<string, unknown>): Session {
        const now = Date.now();
        const session = {
          id: `session-${nextId++}`,
          createdAt: now,
          updatedAt: now,
          messages: [],
          metadata: metadata ?? {},
        } satisfies Session;
        sessions.set(session.id, session);
        return session;
      },
      getSession(id: string): Session | null {
        return sessions.get(id) ?? null;
      },
      listSessions(limit: number = 50): ReadonlyArray<Session> {
        return [...sessions.values()]
          .sort((a, b) => b.updatedAt - a.updatedAt)
          .slice(0, limit);
      },
      updateSessionMetadata(id: string, patch: Record<string, unknown>): Session | null {
        const session = sessions.get(id);
        if (!session) {
          return null;
        }
        const updated = {
          ...session,
          updatedAt: Date.now(),
          metadata: {
            ...session.metadata,
            ...patch,
          },
        } satisfies Session;
        sessions.set(id, updated);
        return updated;
      },
      addMessage(sessionId: string, message: Session["messages"][number]): void {
        const session = sessions.get(sessionId);
        if (!session) {
          throw new Error(`missing session ${sessionId}`);
        }
        const updated = {
          ...session,
          updatedAt: Date.now(),
          messages: [...session.messages, message],
        } satisfies Session;
        sessions.set(sessionId, updated);
      },
      addCostRecord(): void {},
      saveCompactionEvent(): void {},
      saveSessionState(id: string, state: object): void {
        sessionState.set(id, state as Record<string, unknown>);
      },
      loadSessionState(id: string): Record<string, unknown> | null {
        return sessionState.get(id) ?? null;
      },
      close(): void {},
    } as unknown as SessionStore;
  }

  it("does not create a session before activation", async () => {
    const store = createMemorySessionStore();
    const persistence = await setupSessionPersistence(
      createSessionTestConfig(),
      parseArgs(["node", "devagent", "--quiet"]),
      "/Users/egavrin/Documents/devagent",
      {} as LLMProvider,
      new EventBus(),
      new SessionState(),
      {
        sessionStore: store,
        createCrashReporter: () => ({ printSessionId() {}, dispose() {} }),
      },
    );

    expect(store.listSessions()).toHaveLength(0);
    expect(persistence.hasActiveSession()).toBe(false);
  });

  it("creates one session on first activation and reuses it across turns", async () => {
    const store = createMemorySessionStore();
    const persistence = await setupSessionPersistence(
      createSessionTestConfig(),
      parseArgs(["node", "devagent", "--quiet"]),
      "/Users/egavrin/Documents/devagent",
      {} as LLMProvider,
      new EventBus(),
      new SessionState(),
      {
        sessionStore: store,
        createCrashReporter: () => ({ printSessionId() {}, dispose() {} }),
      },
    );

    const first = persistence.activateSession("first prompt");
    const second = persistence.activateSession("second prompt");

    expect(first.id).toBe(second.id);
    expect(store.listSessions()).toHaveLength(1);
    expect(store.getSession(first.id)?.metadata).toMatchObject({
      query: "first prompt",
      title: "first prompt",
      repoLabel: "devagent",
      repoRoot: "/Users/egavrin/Documents/devagent",
      provider: "openai",
      model: "gpt-5",
    });
  });

  it("restores resumed state before binding it to the new session", async () => {
    const store = createMemorySessionStore();
    const previous = store.createSession({ query: "earlier prompt" });
    const restored = new SessionState();
    restored.recordModifiedFile("packages/cli/src/main.ts");
    store.saveSessionState(previous.id, restored.toJSON());

    const persistence = await setupSessionPersistence(
      createSessionTestConfig(),
      parseArgs(["node", "devagent", "--quiet", "--continue"]),
      "/Users/egavrin/Documents/devagent",
      {} as LLMProvider,
      new EventBus(),
      new SessionState(),
      {
        sessionStore: store,
        createCrashReporter: () => ({ printSessionId() {}, dispose() {} }),
      },
    );

    expect(persistence.sessionState.getModifiedFiles()).toEqual(["packages/cli/src/main.ts"]);

    const next = persistence.activateSession("follow-up prompt");
    persistence.sessionState.recordModifiedFile("packages/runtime/src/core/session.ts");

    expect(store.listSessions()).toHaveLength(2);
    expect((store.loadSessionState(previous.id) as { modifiedFiles?: string[] }).modifiedFiles).toEqual([
      "packages/cli/src/main.ts",
    ]);
    expect((store.loadSessionState(next.id) as { modifiedFiles?: string[] }).modifiedFiles).toEqual([
      "packages/cli/src/main.ts",
      "packages/runtime/src/core/session.ts",
    ]);
  });

  it("creates a fresh session only after clear-style deactivation", async () => {
    const store = createMemorySessionStore();
    const persistence = await setupSessionPersistence(
      createSessionTestConfig(),
      parseArgs(["node", "devagent", "--quiet"]),
      "/Users/egavrin/Documents/devagent",
      {} as LLMProvider,
      new EventBus(),
      new SessionState(),
      {
        sessionStore: store,
        createCrashReporter: () => ({ printSessionId() {}, dispose() {} }),
      },
    );

    const first = persistence.activateSession("first prompt");
    persistence.deactivateSession();
    const second = persistence.activateSession("second prompt");

    expect(first.id).not.toBe(second.id);
    expect(store.listSessions()).toHaveLength(2);
  });

  it("marks continue with no sessions as terminal without creating a session", async () => {
    const store = createMemorySessionStore();
    const persistence = await setupSessionPersistence(
      createSessionTestConfig(),
      parseArgs(["node", "devagent", "--quiet", "--continue"]),
      "/Users/egavrin/Documents/devagent",
      {} as LLMProvider,
      new EventBus(),
      new SessionState(),
      {
        sessionStore: store,
        createCrashReporter: () => ({ printSessionId() {}, dispose() {} }),
      },
    );

    expect(persistence.resumeTargetMissing).toBe(true);
    expect(store.listSessions()).toHaveLength(0);
  });

  it("includes title and repo context in ambiguous resume errors", async () => {
    const store = createMemorySessionStore();
    store.createSession({
      title: "Fix parser tests",
      repoLabel: "devagent",
    });
    store.createSession({
      title: "Review provider config",
      repoLabel: "providers",
    });

    await expect(setupSessionPersistence(
      createSessionTestConfig(),
      parseArgs(["node", "devagent", "--quiet", "--resume", "session-"]),
      "/Users/egavrin/Documents/devagent",
      {} as LLMProvider,
      new EventBus(),
      new SessionState(),
      {
        sessionStore: store,
        createCrashReporter: () => ({ printSessionId() {}, dispose() {} }),
      },
    )).rejects.toThrow([
      'Ambiguous session prefix "session-". Matching sessions:',
      "- session-  Fix parser tests  devagent",
      "- session-  Review provider config  providers",
    ].join("\n"));
  });
});

describe("review command validation", () => {
  it("fails on missing --rule before provider setup", () => {
    execFileSync("bun", ["run", "build"], {
      cwd: cliPackageDir,
      stdio: "pipe",
    });

    const home = mkdtempSync(join(tmpdir(), "devagent-review-home-"));
    const workDir = mkdtempSync(join(tmpdir(), "devagent-review-work-"));
    const patchPath = join(workDir, "patch.diff");
    try {
      mkdirSync(join(home, ".config", "devagent"), { recursive: true });
      writeFileSync(
        join(home, ".config", "devagent", "config.toml"),
        'provider = "openai"\n\n[providers.openai]\napi_key = "env:OPENAI_API_KEY"\n',
      );
      writeFileSync(patchPath, "diff --git a/a.txt b/a.txt\n");

      try {
        execFileSync("bun", ["dist/index.js", "review", patchPath], {
          cwd: cliPackageDir,
          encoding: "utf-8",
          stdio: ["ignore", "pipe", "pipe"],
          env: {
            ...process.env,
            HOME: home,
          },
        });
        expect.unreachable("expected review without --rule to fail");
      } catch (error) {
        const execError = error as Error & { stderr?: string; status?: number };
        expect(execError.status).toBe(1);
        expect(execError.stderr).toContain("Rule file required: devagent review <file> --rule <rule_file>");
        expect(execError.stderr).not.toContain("OPENAI_API_KEY");
      }
    } finally {
      rmSync(home, { recursive: true, force: true });
      rmSync(workDir, { recursive: true, force: true });
    }
  });
});

describe("checkForUpdates", () => {
  afterEach(() => {
    delete process.env["DEVAGENT_DISABLE_UPDATE_CHECK"];
    vi.restoreAllMocks();
  });

  it("skips the update request when disabled by env", () => {
    process.env["DEVAGENT_DISABLE_UPDATE_CHECK"] = "1";
    const fetchSpy = vi.fn();
    const originalFetch = globalThis.fetch;
    (globalThis as typeof globalThis & { fetch: typeof fetchSpy }).fetch = fetchSpy;

    checkForUpdates();

    expect(fetchSpy).not.toHaveBeenCalled();
    globalThis.fetch = originalFetch;
  });
});
