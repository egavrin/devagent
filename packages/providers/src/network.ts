import { EnvHttpProxyAgent, type Dispatcher } from "undici";

type FetchFn = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

interface ProxyableRequestInit extends RequestInit {
  dispatcher?: Dispatcher;
}

const LOOPBACK_HOSTS = new Set(["localhost", "127.0.0.1", "::1", "[::1]"]);

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

function getProxyDispatcher(): Dispatcher | null {
  const signature = getProxySignature();
  if (!signature) return null;
  if (signature === cachedProxySignature && cachedProxyDispatcher) {
    return cachedProxyDispatcher;
  }

  cachedProxySignature = signature;
  cachedProxyDispatcher = new EnvHttpProxyAgent();
  return cachedProxyDispatcher;
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
): FetchFn {
  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const dispatcher = getProxyDispatcher();
    if (!dispatcher || shouldBypassProxy(input)) {
      return baseFetch(input, init);
    }

    return baseFetch(input, {
      ...(init ?? {}),
      dispatcher,
    } as ProxyableRequestInit);
  };
}
