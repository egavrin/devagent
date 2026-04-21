import { writeFileSync, mkdirSync, rmSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { it, expect, beforeEach, afterEach } from "vitest";

import { CredentialStore } from "./credentials.js";
import { CredentialError } from "./errors.js";
let testDir: string;
let filePath: string;

beforeEach(() => {
  testDir = join(tmpdir(), `devagent-cred-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  filePath = join(testDir, "credentials.json");
});

afterEach(() => {
  try {
    rmSync(testDir, { recursive: true, force: true });
  } catch {
    // ignore cleanup errors
  }
});

it("returns null for unknown provider", () => {
  const store = new CredentialStore({ filePath });
  expect(store.get("anthropic")).toBeNull();
});

it("stores and retrieves an API key", () => {
  const store = new CredentialStore({ filePath });
  store.set("anthropic", {
    type: "api",
    key: "sk-ant-test-key",
    storedAt: Date.now(),
  });

  const cred = store.get("anthropic");
  expect(cred).not.toBeNull();
  expect(cred!.type).toBe("api");
  expect(cred!.key).toBe("sk-ant-test-key");
});

it("overwrites existing credential for same provider", () => {
  const store = new CredentialStore({ filePath });
  store.set("openai", { type: "api", key: "old-key", storedAt: 1 });
  store.set("openai", { type: "api", key: "new-key", storedAt: 2 });

  const cred = store.get("openai");
  expect(cred!.key).toBe("new-key");
  expect(cred!.storedAt).toBe(2);
});

it("removes a credential and returns true", () => {
  const store = new CredentialStore({ filePath });
  store.set("anthropic", { type: "api", key: "key", storedAt: 1 });

  expect(store.remove("anthropic")).toBe(true);
  expect(store.get("anthropic")).toBeNull();
});

it("returns false when removing nonexistent credential", () => {
  const store = new CredentialStore({ filePath });
  expect(store.remove("nonexistent")).toBe(false);
});

it("returns all stored credentials", () => {
  const store = new CredentialStore({ filePath });
  store.set("anthropic", { type: "api", key: "ant-key", storedAt: 1 });
  store.set("openai", { type: "api", key: "oai-key", storedAt: 2 });

  const all = store.all();
  expect(Object.keys(all)).toHaveLength(2);
  expect(all["anthropic"]!.key).toBe("ant-key");
  expect(all["openai"]!.key).toBe("oai-key");
});

it("creates directory if missing", () => {
  const deepPath = join(testDir, "deep", "nested", "credentials.json");
  const store = new CredentialStore({ filePath: deepPath });
  store.set("anthropic", { type: "api", key: "key", storedAt: 1 });

  expect(store.get("anthropic")!.key).toBe("key");
});

it("sets file permissions to 0o600", () => {
  const store = new CredentialStore({ filePath });
  store.set("anthropic", { type: "api", key: "key", storedAt: 1 });

  const stat = statSync(filePath);
  const perms = stat.mode & 0o777;
  expect(perms).toBe(0o600);
});

it("throws CredentialError on corrupted JSON", () => {
  mkdirSync(testDir, { recursive: true });
  writeFileSync(filePath, "not valid json{{{", { mode: 0o600 });

  const store = new CredentialStore({ filePath });
  expect(() => store.all()).toThrow(CredentialError);
});

it("preserves other providers when setting one", () => {
  const store = new CredentialStore({ filePath });
  store.set("anthropic", { type: "api", key: "ant-key", storedAt: 1 });
  store.set("openai", { type: "api", key: "oai-key", storedAt: 2 });

  // Verify both are preserved
  expect(store.get("anthropic")!.key).toBe("ant-key");
  expect(store.get("openai")!.key).toBe("oai-key");
});

it("has() returns correct boolean", () => {
  const store = new CredentialStore({ filePath });
  expect(store.has("anthropic")).toBe(false);

  store.set("anthropic", { type: "api", key: "key", storedAt: 1 });
  expect(store.has("anthropic")).toBe(true);
});

it("returns empty object for all() when no file exists", () => {
  const store = new CredentialStore({ filePath });
  const all = store.all();
  expect(all).toEqual({});
});

// ─── OAuth Credential Tests ──────────────────────────────────

it("stores and retrieves an OAuth credential", () => {
  const store = new CredentialStore({ filePath });
  store.set("chatgpt", {
    type: "oauth",
    accessToken: "access-123",
    refreshToken: "refresh-456",
    expiresAt: Date.now() + 3600_000,
    accountId: "org-abc",
    storedAt: Date.now(),
  });

  const cred = store.get("chatgpt");
  expect(cred).not.toBeNull();
  expect(cred!.type).toBe("oauth");
  if (cred!.type === "oauth") {
    expect(cred!.accessToken).toBe("access-123");
    expect(cred!.refreshToken).toBe("refresh-456");
    expect(cred!.accountId).toBe("org-abc");
  }
});

it("stores OAuth credential without optional fields (GitHub-style)", () => {
  const store = new CredentialStore({ filePath });
  store.set("github-copilot", {
    type: "oauth",
    accessToken: "gho_token",
    // No refreshToken, expiresAt, or accountId (GitHub tokens don't expire)
    storedAt: Date.now(),
  });

  const cred = store.get("github-copilot");
  expect(cred).not.toBeNull();
  expect(cred!.type).toBe("oauth");
  if (cred!.type === "oauth") {
    expect(cred!.accessToken).toBe("gho_token");
    expect(cred!.refreshToken).toBeUndefined();
    expect(cred!.expiresAt).toBeUndefined();
    expect(cred!.accountId).toBeUndefined();
  }
});

it("stores mixed API and OAuth credentials", () => {
  const store = new CredentialStore({ filePath });
  store.set("anthropic", { type: "api", key: "sk-ant-key", storedAt: 1 });
  store.set("chatgpt", {
    type: "oauth",
    accessToken: "access-123",
    refreshToken: "refresh-456",
    expiresAt: Date.now() + 3600_000,
    storedAt: 2,
  });

  const all = store.all();
  expect(Object.keys(all)).toHaveLength(2);
  expect(all["anthropic"]!.type).toBe("api");
  expect(all["chatgpt"]!.type).toBe("oauth");
});

it("overwrites API credential with OAuth credential", () => {
  const store = new CredentialStore({ filePath });
  store.set("openai", { type: "api", key: "sk-old", storedAt: 1 });
  store.set("openai", {
    type: "oauth",
    accessToken: "access-new",
    refreshToken: "refresh-new",
    expiresAt: Date.now() + 3600_000,
    storedAt: 2,
  });

  const cred = store.get("openai");
  expect(cred!.type).toBe("oauth");
});

it("accepts OAuth entries without refreshToken and expiresAt (GitHub-style)", () => {
  mkdirSync(testDir, { recursive: true });
  writeFileSync(
    filePath,
    JSON.stringify({
      valid_api: { type: "api", key: "k", storedAt: 1 },
      github_oauth: { type: "oauth", accessToken: "gho_tok", storedAt: 2 }, // no refreshToken/expiresAt
    }),
    { mode: 0o600 },
  );

  const store = new CredentialStore({ filePath });
  const all = store.all();
  expect(Object.keys(all)).toHaveLength(2);
  expect(all["valid_api"]).toBeDefined();
  expect(all["github_oauth"]).toBeDefined();
  if (all["github_oauth"]!.type === "oauth") {
    expect(all["github_oauth"]!.accessToken).toBe("gho_tok");
    expect(all["github_oauth"]!.refreshToken).toBeUndefined();
    expect(all["github_oauth"]!.expiresAt).toBeUndefined();
  }
});

it("skips invalid OAuth entries missing accessToken", () => {
  mkdirSync(testDir, { recursive: true });
  writeFileSync(
    filePath,
    JSON.stringify({
      valid: { type: "api", key: "k", storedAt: 1 },
      invalid_oauth: { type: "oauth", refreshToken: "rt" }, // missing accessToken
    }),
    { mode: 0o600 },
  );

  const store = new CredentialStore({ filePath });
  const all = store.all();
  expect(Object.keys(all)).toHaveLength(1);
  expect(all["valid"]).toBeDefined();
  expect(all["invalid_oauth"]).toBeUndefined();
});
