import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getGlobalConfigPath, loadGlobalConfigObject, writeGlobalConfigObject } from "./global-config.js";
import { runCompletions, runConfig, runConfigure, runInit, runSetup } from "./commands.js";

describe("command help", () => {
  let tempHome: string;
  let tempRepo: string;
  let originalHome: string | undefined;
  let originalCwd: string;

  beforeEach(() => {
    tempHome = mkdtempSync(join(tmpdir(), "devagent-cli-home-"));
    tempRepo = mkdtempSync(join(tmpdir(), "devagent-cli-repo-"));
    originalHome = process.env["HOME"];
    originalCwd = process.cwd();
    process.env["HOME"] = tempHome;
    process.chdir(tempRepo);
    vi.spyOn(console, "log").mockImplementation(() => {});
    vi.spyOn(console, "error").mockImplementation(() => {});
  });

  afterEach(() => {
    vi.restoreAllMocks();
    process.chdir(originalCwd);
    if (originalHome === undefined) delete process.env["HOME"];
    else process.env["HOME"] = originalHome;
    rmSync(tempHome, { recursive: true, force: true });
    rmSync(tempRepo, { recursive: true, force: true });
  });

  it("config --help does not create the global config file", () => {
    runConfig(["--help"]);

    expect(existsSync(getGlobalConfigPath(tempHome))).toBe(false);
  });

  it("configure and setup help do not create the global config file", async () => {
    await runConfigure(["--help"]);
    expect(existsSync(getGlobalConfigPath(tempHome))).toBe(false);

    await runSetup(["--help"]);
    expect(existsSync(getGlobalConfigPath(tempHome))).toBe(false);
  });

  it("init --help does not create project instruction files", () => {
    runInit(["--help"]);

    expect(existsSync(join(tempRepo, "AGENTS.md"))).toBe(false);
    expect(existsSync(join(tempRepo, ".devagent"))).toBe(false);
  });

  it("retired init exits with guidance instead of scaffolding files", () => {
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`process.exit:${code ?? 0}`);
    }) as never);

    expect(() => runInit([])).toThrow("process.exit:2");
    expect(exitSpy).toHaveBeenCalledWith(2);
    expect(existsSync(join(tempRepo, "AGENTS.md"))).toBe(false);
    expect(existsSync(join(tempRepo, ".devagent"))).toBe(false);
  });

  it("config set provider syncs an incompatible model to the provider default", () => {
    writeGlobalConfigObject({
      provider: "anthropic",
      model: "claude-sonnet-4-20250514",
    });

    runConfig(["set", "provider", "chatgpt"]);

    expect(loadGlobalConfigObject()).toMatchObject({
      provider: "chatgpt",
      model: "gpt-5.4",
    });
    expect(console.log).toHaveBeenCalledWith("provider = chatgpt");
    expect(console.log).toHaveBeenCalledWith("model = gpt-5.4");
  });

  it("config set provider preserves a shared model that is already valid for that provider", () => {
    writeGlobalConfigObject({
      provider: "github-copilot",
      model: "gpt-4.1",
    });

    runConfig(["set", "provider", "openai"]);

    expect(loadGlobalConfigObject()).toMatchObject({
      provider: "openai",
      model: "gpt-4.1",
    });
    expect(console.log).toHaveBeenCalledWith("provider = openai");
  });

  it("config set model rejects a model that is not registered for the configured provider", () => {
    writeGlobalConfigObject({
      provider: "chatgpt",
      model: "gpt-4.1",
    });

    const exitSpy = vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`process.exit:${code ?? 0}`);
    }) as never);

    expect(() => runConfig(["set", "model", "claude-sonnet-4-20250514"])).toThrow("process.exit:2");
    expect(exitSpy).toHaveBeenCalledWith(2);
    expect(console.error).toHaveBeenCalled();
    expect(loadGlobalConfigObject()).toMatchObject({
      provider: "chatgpt",
      model: "gpt-4.1",
    });
  });

  it("config set safety.mode writes the canonical safety section", () => {
    runConfig(["set", "safety.mode", "default"]);

    expect(loadGlobalConfigObject()).toMatchObject({
      safety: { mode: "default" },
    });
    expect(console.log).toHaveBeenCalledWith("safety.mode = default");
  });

  it("config set approval.mode rewrites the legacy alias to safety.mode", () => {
    runConfig(["set", "approval.mode", "full-auto"]);

    expect(loadGlobalConfigObject()).toMatchObject({
      safety: { mode: "autopilot" },
    });
    expect(console.log).toHaveBeenCalledWith("safety.mode = autopilot");
  });

  it("bash completions advertise setup and not configure", () => {
    runCompletions(["bash"]);

    const emitted = (console.log as ReturnType<typeof vi.fn>).mock.calls.map(([line]) => String(line)).join("\n");
    expect(emitted).toContain("setup");
    expect(emitted).not.toContain("configure");
  });
});
