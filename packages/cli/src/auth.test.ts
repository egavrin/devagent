import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CredentialStore } from "@devagent/runtime";

import { collectCredentialStatusEntries, runAuthCommand } from "./auth.js";

describe("collectCredentialStatusEntries", () => {
  it("reports provider-specific env vars without adding a duplicate generic gateway row", () => {
    const rows = collectCredentialStatusEntries({}, {
      OPENAI_API_KEY: "sk-openai",
      DEVAGENT_API_KEY: "ilg-gateway",
    });

    expect(rows).toEqual([
      { id: "devagent-api", source: "env:DEVAGENT_API_KEY", masked: "ilg-...eway" },
      { id: "openai", source: "env:OPENAI_API_KEY", masked: "sk-o...enai" },
    ]);
  });
});

describe("runAuthCommand logout", () => {
  let tempHome: string;
  let originalHome: string | undefined;

  beforeEach(() => {
    tempHome = mkdtempSync(join(tmpdir(), "devagent-auth-home-"));
    originalHome = process.env["HOME"];
    process.env["HOME"] = tempHome;
    vi.spyOn(process.stderr, "write").mockImplementation(() => true);
  });

  afterEach(() => {
    vi.restoreAllMocks();
    if (originalHome === undefined) delete process.env["HOME"];
    else process.env["HOME"] = originalHome;
    rmSync(tempHome, { recursive: true, force: true });
  });

  it("removes a named credential without prompting", async () => {
    const store = new CredentialStore();
    store.set("chatgpt", {
      type: "oauth",
      accessToken: "token-value",
      storedAt: Date.now(),
    });

    await runAuthCommand("logout", ["chatgpt"]);

    expect(new CredentialStore().all()["chatgpt"]).toBeUndefined();
  });

  it("removes all stored credentials with --all", async () => {
    const store = new CredentialStore();
    store.set("chatgpt", {
      type: "oauth",
      accessToken: "token-value",
      storedAt: Date.now(),
    });
    store.set("openai", {
      type: "api",
      key: "sk-test",
      storedAt: Date.now(),
    });

    await runAuthCommand("logout", ["--all"]);

    expect(new CredentialStore().all()).toEqual({});
  });

  it("fails with usage when --all is followed by extra arguments", async () => {
    const store = new CredentialStore();
    store.set("chatgpt", {
      type: "oauth",
      accessToken: "token-value",
      storedAt: Date.now(),
    });
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`process.exit:${code ?? 0}`);
    }) as never);

    await expect(runAuthCommand("logout", ["--all", "chatgpt"])).rejects.toThrow("process.exit:2");
    expect(exitSpy).toHaveBeenCalledWith(2);
    expect(new CredentialStore().all()["chatgpt"]).toBeDefined();
  });

  it("fails with usage in non-tty mode when no logout target is provided", async () => {
    const store = new CredentialStore();
    store.set("chatgpt", {
      type: "oauth",
      accessToken: "token-value",
      storedAt: Date.now(),
    });
    Object.defineProperty(process.stdin, "isTTY", {
      value: false,
      configurable: true,
    });
    const exitSpy = vi.spyOn(process, "exit").mockImplementation(((code?: number) => {
      throw new Error(`process.exit:${code ?? 0}`);
    }) as never);

    await expect(runAuthCommand("logout")).rejects.toThrow("process.exit:2");
    expect(exitSpy).toHaveBeenCalledWith(2);
    expect(new CredentialStore().all()["chatgpt"]).toBeDefined();
  });
});
