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
      const loadUndici = vi.fn();
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch, { loadUndici });

      await proxyFetch("https://example.com/v1/models");

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0]?.[1]).toBeUndefined();
      expect(loadUndici).not.toHaveBeenCalled();
    });
  });

  it("attaches an undici dispatcher for remote requests when proxy env vars are set", async () => {
    await withClearedProxyEnvAsync(async () => {
      process.env["HTTPS_PROXY"] = "https://proxy.example.com:8443";
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const dispatcher = { dispatch: vi.fn() };
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch, {
        runtime: "node",
        loadUndici: async () => ({
          EnvHttpProxyAgent: class {
            constructor() {
              return dispatcher;
            }
          },
        }),
      });

      await proxyFetch("https://example.com/v1/models", {
        headers: { "x-test": "1" },
      });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      const init = fetchMock.mock.calls[0]?.[1] as RequestInit & { dispatcher?: unknown };
      expect(init.headers).toEqual({ "x-test": "1" });
      expect(init.dispatcher).toBe(dispatcher);
    });
  });

  it("fails clearly under Bun when proxy env vars are set", async () => {
    await withClearedProxyEnvAsync(async () => {
      process.env["HTTPS_PROXY"] = "https://proxy.example.com:8443";
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch, { runtime: "bun" });

      await expect(proxyFetch("https://example.com/v1/models")).rejects.toThrow(
        "proxy dispatchers require Node.js",
      );
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  it("bypasses proxy dispatch for loopback hosts even when proxy env vars are set", async () => {
    await withClearedProxyEnvAsync(async () => {
      process.env["HTTPS_PROXY"] = "https://proxy.example.com:8443";
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const loadUndici = vi.fn();
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch, { runtime: "node", loadUndici });

      await proxyFetch("http://localhost:11434/v1/models");
      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect((fetchMock.mock.calls[0]?.[1] as { dispatcher?: unknown } | undefined)?.dispatcher).toBeUndefined();
      expect(loadUndici).not.toHaveBeenCalled();
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
