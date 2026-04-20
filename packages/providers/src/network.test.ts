import { afterEach, describe, expect, it, vi } from "vitest";

import { createProxyAwareFetch, hasProxyEnv, shouldBypassProxy } from "./network.js";

const PROXY_ENV_VARS = ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"] as const;

function withClearedProxyEnv(fn: () => void): void {
  const original = new Map<string, string | undefined>();
  for (const key of PROXY_ENV_VARS) {
    original.set(key, process.env[key]);
    delete process.env[key];
  }

  try {
    fn();
  } finally {
    for (const key of PROXY_ENV_VARS) {
      const value = original.get(key);
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

describe("network proxy helpers", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("detects when proxy env vars are configured", () => {
    withClearedProxyEnv(() => {
      expect(hasProxyEnv()).toBe(false);
      process.env["HTTPS_PROXY"] = "https://proxy.example.com:8443";
      expect(hasProxyEnv()).toBe(true);
    });
  });

  it("does not attach a dispatcher when no proxy env vars are set", async () => {
    await withClearedProxyEnvAsync(async () => {
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch);

      await proxyFetch("https://example.com/v1/models");

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0]?.[1]).toBeUndefined();
    });
  });

  it("attaches an undici dispatcher for remote requests when proxy env vars are set", async () => {
    await withClearedProxyEnvAsync(async () => {
      process.env["HTTPS_PROXY"] = "https://proxy.example.com:8443";
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch);

      await proxyFetch("https://example.com/v1/models", {
        headers: { "x-test": "1" },
      });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      const init = fetchMock.mock.calls[0]?.[1] as RequestInit & { dispatcher?: unknown };
      expect(init.headers).toEqual({ "x-test": "1" });
      expect(init.dispatcher).toBeTruthy();
    });
  });

  it("bypasses proxy dispatch for loopback hosts even when proxy env vars are set", async () => {
    await withClearedProxyEnvAsync(async () => {
      process.env["HTTPS_PROXY"] = "https://proxy.example.com:8443";
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch);

      await proxyFetch("http://localhost:11434/v1/models");
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect((fetchMock.mock.calls[0]?.[1] as { dispatcher?: unknown } | undefined)?.dispatcher).toBeUndefined();
      expect(shouldBypassProxy("http://127.0.0.1:11434/v1/models")).toBe(true);
      expect(shouldBypassProxy("http://[::1]:11434/v1/models")).toBe(true);
    });
  });
});

async function withClearedProxyEnvAsync(fn: () => Promise<void>): Promise<void> {
  const original = new Map<string, string | undefined>();
  for (const key of PROXY_ENV_VARS) {
    original.set(key, process.env[key]);
    delete process.env[key];
  }

  try {
    await fn();
  } finally {
    for (const key of PROXY_ENV_VARS) {
      const value = original.get(key);
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}
