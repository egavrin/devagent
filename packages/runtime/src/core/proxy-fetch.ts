import type { Dispatcher } from "undici";

export type FetchFn = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
export type ProxyRuntime = "bun" | "node";

export type UndiciProxyModule = {
  readonly EnvHttpProxyAgent: new () => Dispatcher;
};

export interface ProxyAwareFetchOptions {
  readonly runtime?: ProxyRuntime;
  readonly loadUndici?: () => Promise<UndiciProxyModule>;
  readonly createUnsupportedProxyError?: (message: string) => Error;
}

interface ProxyableRequestInit extends RequestInit {
  readonly dispatcher?: Dispatcher;
}

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

export const BUN_PROXY_UNSUPPORTED_MESSAGE =
  "Proxy environment variables are set, but proxy dispatchers require Node.js for now. Unset proxy env vars or run DevAgent with Node.js.";

let cachedProxySignature: string | null = null;
let cachedProxyDispatcher: Dispatcher | null = null;

function getProxyEnvValue(name: string): string | undefined {
  return process.env[name] ?? process.env[name.toLowerCase()];
}

function getProxySignature(): string | null {
  const httpProxy = getProxyEnvValue("HTTP_PROXY");
  const httpsProxy = getProxyEnvValue("HTTPS_PROXY");
  if (!httpProxy && !httpsProxy) return null;

  return JSON.stringify({
    HTTP_PROXY: httpProxy ?? null,
    HTTPS_PROXY: httpsProxy ?? null,
    NO_PROXY: getProxyEnvValue("NO_PROXY") ?? null,
  });
}

function getProxyRuntime(): ProxyRuntime {
  return typeof Bun === "undefined" ? "node" : "bun";
}

async function loadUndiciProxyModule(): Promise<UndiciProxyModule> {
  return await import("undici") as UndiciProxyModule;
}
async function getProxyDispatcher(options?: ProxyAwareFetchOptions): Promise<Dispatcher | null> {
  const signature = getProxySignature();
  if (!signature) return null;
  const runtime = options?.runtime ?? getProxyRuntime();
  if (runtime === "bun") {
    throw createUnsupportedProxyError(options);
  }
  if (signature === cachedProxySignature && cachedProxyDispatcher) {
    return cachedProxyDispatcher;
  }

  const { EnvHttpProxyAgent } = await (options?.loadUndici ?? loadUndiciProxyModule)();
  cachedProxySignature = signature;
  cachedProxyDispatcher = new EnvHttpProxyAgent();
  return cachedProxyDispatcher;
}

function createUnsupportedProxyError(options?: ProxyAwareFetchOptions) {
  return options?.createUnsupportedProxyError?.(BUN_PROXY_UNSUPPORTED_MESSAGE)
    ?? new Error(BUN_PROXY_UNSUPPORTED_MESSAGE);
}

function resolveRequestUrl(input: RequestInfo | URL): URL | null {
  try {
    if (typeof input === "string") return new URL(input);
    if (input instanceof URL) return input;
    if (typeof Request !== "undefined" && input instanceof Request) {
      return new URL(input.url);
    }
    if (typeof input === "object" && input !== null && "url" in input) {
      const url = (input as { url?: unknown }).url;
      return typeof url === "string" ? new URL(url) : null;
    }
  } catch {
    return null;
  }
  return null;
}

export function hasProxyEnv(): boolean {
  return getProxySignature() !== null;
}

export function shouldBypassProxy(input: RequestInfo | URL): boolean {
  const url = resolveRequestUrl(input);
  if (!url) return false;
  return LOOPBACK_HOSTS.has(url.hostname.toLowerCase());
}

export function createProxyAwareFetch(
  baseFetch: FetchFn = globalThis.fetch.bind(globalThis),
  options?: ProxyAwareFetchOptions,
): FetchFn {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    if (shouldBypassProxy(input)) {
      return baseFetch(input, init);
    }
    const dispatcher = await getProxyDispatcher(options);
    if (!dispatcher) {
      return baseFetch(input, init);
    }

    return baseFetch(input, {
      ...(init ?? {}),
      dispatcher,
    } as ProxyableRequestInit);
  };
}
