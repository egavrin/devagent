import { existsSync, mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { getGlobalConfigPath } from "./global-config.js";
import { runConfig, runConfigure, runInit, runSetup } from "./commands.js";

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

    runSetup(["--help"]);
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

  it("retired setup exits with guidance instead of launching the wizard", () => {
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`process.exit:${code ?? 0}`);
    }) as never);

    expect(() => runSetup([])).toThrow("process.exit:2");
    expect(exitSpy).toHaveBeenCalledWith(2);
    expect(existsSync(getGlobalConfigPath(tempHome))).toBe(false);
  });
});
