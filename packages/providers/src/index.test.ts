import { afterEach, describe, expect, it, vi } from "vitest";
import { MessageRole } from "@devagent/runtime";
import { createDefaultRegistry } from "./index.js";

describe("createDefaultRegistry", () => {
  const originalFetch = globalThis.fetch;
  const proxyEnvKeys = ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"] as const;
  const originalProxyEnv = new Map<string, string | undefined>();

  afterEach(() => {
    globalThis.fetch = originalFetch;
    for (const key of proxyEnvKeys) {
      const value = originalProxyEnv.get(key);
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
      originalProxyEnv.delete(key);
    }
    vi.restoreAllMocks();
  });

  it("registers devagent-api as a built-in provider", () => {
    const registry = createDefaultRegistry();

    expect(registry.has("devagent-api")).toBe(true);
    expect(registry.list()).toContain("devagent-api");
  });

  it("forces devagent-api through chat completions for cortex", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeChatStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const registry = createDefaultRegistry();
    const provider = registry.get("devagent-api", {
      model: "cortex",
      apiKey: "test-key",
      capabilities: {
        useResponsesApi: true,
        reasoning: false,
        supportsTemperature: true,
      },
    });

    let output = "";
    for await (const chunk of provider.chat([
      { role: MessageRole.USER, content: "ping" },
    ])) {
      if (chunk.type === "text") {
        output += chunk.content;
      }
    }

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = String(fetchMock.mock.calls[0]?.[0] ?? "");
    expect(url).toContain("/v1/chat/completions");
    expect(url).not.toContain("/v1/responses");
    expect(output).toBe("pong");

    const headers = new Headers(fetchMock.mock.calls[0]?.[1]?.headers);
    expect(headers.get("x-request-id")).toBeTruthy();
  });

  it("rewrites developer messages to system for the devagent-api gateway", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeChatStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const registry = createDefaultRegistry();
    const provider = registry.get("devagent-api", {
      model: "cortex",
      apiKey: "test-key",
      capabilities: {
        useResponsesApi: false,
        reasoning: true,
        supportsTemperature: false,
      },
    });

    for await (const _chunk of provider.chat([
      { role: MessageRole.SYSTEM, content: "system prompt" },
      { role: MessageRole.USER, content: "ping" },
    ])) {
      void _chunk;
    }

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body ?? "{}")) as {
      messages?: Array<{ role?: string }>;
    };
    expect(body.messages?.map((message) => message.role)).toContain("system");
    expect(body.messages?.map((message) => message.role)).not.toContain("developer");
  });

  it("keeps devagent-api request rewriting when proxy env vars are configured", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeChatStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;
    setProxyEnv(originalProxyEnv, {
      HTTPS_PROXY: "https://proxy.example.com:8443",
    });

    const registry = createDefaultRegistry();
    const provider = registry.get("devagent-api", {
      model: "cortex",
      apiKey: "test-key",
      capabilities: {
        useResponsesApi: false,
        reasoning: true,
        supportsTemperature: false,
      },
    });

    for await (const _chunk of provider.chat([
      { role: MessageRole.SYSTEM, content: "system prompt" },
      { role: MessageRole.USER, content: "ping" },
    ])) {
      void _chunk;
    }

    const init = fetchMock.mock.calls[0]?.[1] as RequestInit & { dispatcher?: unknown };
    const headers = new Headers(init.headers);
    const body = JSON.parse(String(init.body ?? "{}")) as {
      messages?: Array<{ role?: string }>;
    };
    expect(headers.get("x-request-id")).toBeTruthy();
    expect(init.dispatcher).toBeTruthy();
    expect(body.messages?.map((message) => message.role)).toContain("system");
    expect(body.messages?.map((message) => message.role)).not.toContain("developer");
  });
});

function makeChatStreamingResponse(): Response {
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      controller.enqueue(encoder.encode(
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":0,"model":"cortex","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":null}]}\n\n',
      ));
      controller.enqueue(encoder.encode(
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":0,"model":"cortex","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
      ));
      controller.enqueue(encoder.encode("data: [DONE]\n\n"));
      controller.close();
    },
  });

  return new Response(stream, {
    status: 200,
    headers: {
      "content-type": "text/event-stream",
    },
  });
}

function setProxyEnv(
  originalEnv: Map<string, string | undefined>,
  values: Partial<Record<string, string>>,
): void {
  for (const [key, value] of Object.entries(values)) {
    if (!originalEnv.has(key)) {
      originalEnv.set(key, process.env[key]);
    }
    process.env[key] = value;
  }
}
