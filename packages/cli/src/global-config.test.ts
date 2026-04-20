import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it } from "vitest";

import {
  getGlobalConfigPath,
  getGlobalConfigValue,
  migrateLegacyGlobalConfigIfNeeded,
  migrateLegacyGlobalTomlIfNeeded,
  loadGlobalConfigObject,
  setGlobalConfigValue,
  writeGlobalConfigObject,
} from "./global-config.js";

let homeDir: string;
let originalHome: string | undefined;

describe("global config", () => {
  beforeEach(() => {
    homeDir = join(tmpdir(), `devagent-cli-config-${Date.now()}-${Math.random().toString(16).slice(2)}`);
    mkdirSync(homeDir, { recursive: true });
    originalHome = process.env["HOME"];
    process.env["HOME"] = homeDir;
  });

  afterEach(() => {
    if (originalHome === undefined) {
      delete process.env["HOME"];
    } else {
      process.env["HOME"] = originalHome;
    }
    rmSync(homeDir, { recursive: true, force: true });
  });

  it("uses config.toml as the canonical global config path", () => {
    expect(getGlobalConfigPath()).toBe(join(homeDir, ".config", "devagent", "config.toml"));
  });

  it("migrates legacy config.json into config.toml and removes the source file", () => {
    const configDir = join(homeDir, ".config", "devagent");
    mkdirSync(configDir, { recursive: true });
    const jsonPath = join(configDir, "config.json");
    writeFileSync(jsonPath, JSON.stringify({
      provider: "openai",
      model: "gpt-4.1",
      budget: {
        maxIterations: 42,
      },
    }, null, 2));

    const notices: string[] = [];
    const result = migrateLegacyGlobalConfigIfNeeded((message) => {
      notices.push(message);
    });

    expect(result.migrated).toBe(true);
    expect(getGlobalConfigValue("provider")).toBe("openai");
    expect(getGlobalConfigValue("budget.max_iterations")).toBe("42");
    expect(existsSync(jsonPath)).toBe(false);
    expect(result.backupPath).toBeTruthy();
    expect(result.backupPath ? existsSync(result.backupPath) : false).toBe(true);
    expect(notices[0]).toContain("Migrated legacy config.json");
  });

  it("does not overwrite an existing canonical config.toml during migration", () => {
    const configDir = join(homeDir, ".config", "devagent");
    mkdirSync(configDir, { recursive: true });
    writeFileSync(join(configDir, "config.json"), JSON.stringify({ provider: "openai" }));
    writeFileSync(join(configDir, "config.toml"), 'provider = "anthropic"\n');

    const result = migrateLegacyGlobalConfigIfNeeded();

    expect(result.migrated).toBe(false);
    expect(getGlobalConfigValue("provider")).toBe("anthropic");
    expect(existsSync(join(configDir, "config.json"))).toBe(true);
  });

  it("migrates legacy global TOML approval settings into safety mode", () => {
    const legacyPath = join(homeDir, ".devagent.toml");
    writeFileSync(legacyPath, 'provider = "openai"\n\n[approval]\nmode = "full-auto"\n');

    const result = migrateLegacyGlobalTomlIfNeeded();

    expect(result.migrated).toBe(true);
    expect(getGlobalConfigValue("safety.mode")).toBe("autopilot");

    const configToml = readFileSync(getGlobalConfigPath(), "utf-8");
    expect(configToml).toContain("[safety]");
    expect(configToml).not.toContain("[approval]");
  });

  it("round-trips canonical values through TOML", () => {
    setGlobalConfigValue("provider", "openai");
    setGlobalConfigValue("budget.maxIterations", "33");
    setGlobalConfigValue("providers.openai.apiKey", "env:OPENAI_API_KEY");

    expect(getGlobalConfigValue("provider")).toBe("openai");
    expect(getGlobalConfigValue("budget.max_iterations")).toBe("33");
    expect(getGlobalConfigValue("providers.openai.api_key")).toBe("env:OPENAI_API_KEY");

    const configToml = readFileSync(getGlobalConfigPath(), "utf-8");
    expect(configToml).toContain('provider = "openai"');
    expect(configToml).toContain("max_iterations = 33");
    expect(configToml).toContain('api_key = "env:OPENAI_API_KEY"');
  });

  it("rewrites legacy approval config to safety when loading an existing canonical config", () => {
    const configDir = join(homeDir, ".config", "devagent");
    mkdirSync(configDir, { recursive: true });
    writeFileSync(
      join(configDir, "config.toml"),
      'provider = "openai"\n\n[approval]\nmode = "suggest"\n',
    );

    expect(loadGlobalConfigObject()).toMatchObject({
      provider: "openai",
      safety: { mode: "default" },
    });

    const configToml = readFileSync(getGlobalConfigPath(), "utf-8");
    expect(configToml).toContain('[safety]\nmode = "default"');
    expect(configToml).not.toContain("[approval]");
  });

  it("drops legacy approval config when writing canonical safety settings", () => {
    writeGlobalConfigObject({
      provider: "openai",
      approval: { mode: "auto-edit", audit_log: true },
      safety: { mode: "autopilot" },
    });

    const configToml = readFileSync(getGlobalConfigPath(), "utf-8");
    expect(configToml).toContain('[safety]\nmode = "autopilot"');
    expect(configToml).not.toContain("[approval]");
  });

  it("fails fast on an unknown legacy approval mode when no safety mode is set", () => {
    const configDir = join(homeDir, ".config", "devagent");
    mkdirSync(configDir, { recursive: true });
    writeFileSync(
      join(configDir, "config.toml"),
      'provider = "openai"\n\n[approval]\nmode = "mystery"\n',
    );

    expect(() => loadGlobalConfigObject())
      .toThrow('Unsupported legacy approval.mode: "mystery"');
  });

  it("fails fast on unsupported config keys", () => {
    expect(() => setGlobalConfigValue("budget.unknownThing", "1"))
      .toThrow('Unsupported config key: "budget.unknownThing"');
  });
});
