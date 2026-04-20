import {
  AgentRegistry,
  ApprovalGate,
  ApprovalMode,
  createDelegateTool,
  EventBus,
  MessageRole,
  ToolRegistry,
} from "@devagent/runtime";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createOpenAIProvider, resolveCapabilities, stripNullArgs } from "./openai.js";
import { convertTools } from "./shared.js";
import type { ProviderConfig, ModelCapabilities } from "@devagent/runtime";

describe("createOpenAIProvider", () => {
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

  it("creates a provider without API key (for local endpoints)", () => {
    const config: ProviderConfig = { model: "llama3" };
    const provider = createOpenAIProvider(config);
    expect(provider.id).toBe("openai");
    expect(typeof provider.chat).toBe("function");
    expect(typeof provider.abort).toBe("function");
  });

  it("creates a provider with valid config", () => {
    const config: ProviderConfig = {
      model: "gpt-4o",
      apiKey: "test-key",
    };
    const provider = createOpenAIProvider(config);
    expect(provider.id).toBe("openai");
    expect(typeof provider.chat).toBe("function");
    expect(typeof provider.abort).toBe("function");
  });

  it("accepts a custom baseUrl", () => {
    const config: ProviderConfig = {
      model: "llama3",
      baseUrl: "http://localhost:11434/v1",
    };
    const provider = createOpenAIProvider(config);
    expect(provider.id).toBe("openai");
  });

  it("injects x-request-id when configured and missing", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const provider = createOpenAIProvider({
      model: "gpt-4o",
      apiKey: "test-key",
      requestIdHeaderName: "x-request-id",
    });

    await collectText(provider);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const headers = new Headers(fetchMock.mock.calls[0]?.[1]?.headers);
    expect(headers.get("x-request-id")).toBeTruthy();
  });

  it("preserves caller-supplied x-request-id", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const provider = createOpenAIProvider({
      model: "gpt-4o",
      apiKey: "test-key",
      requestIdHeaderName: "x-request-id",
      customHeaders: {
        "x-request-id": "preset-request-id",
      },
    });

    await collectText(provider);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const headers = new Headers(fetchMock.mock.calls[0]?.[1]?.headers);
    expect(headers.get("x-request-id")).toBe("preset-request-id");
  });

  it("uses proxy-aware transport for remote requests when proxy env vars are set", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;
    setProxyEnv(originalProxyEnv, {
      HTTPS_PROXY: "https://proxy.example.com:8443",
    });

    const provider = createOpenAIProvider({
      model: "gpt-4o",
      apiKey: "test-key",
      requestIdHeaderName: "x-request-id",
    });

    await collectText(provider);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const init = fetchMock.mock.calls[0]?.[1] as RequestInit & { dispatcher?: unknown };
    const headers = new Headers(init.headers);
    expect(headers.get("x-request-id")).toBeTruthy();
    expect(init.dispatcher).toBeTruthy();
  });

  it("bypasses proxy-aware transport for loopback baseUrls", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;
    setProxyEnv(originalProxyEnv, {
      HTTPS_PROXY: "https://proxy.example.com:8443",
    });

    const provider = createOpenAIProvider({
      model: "llama3",
      baseUrl: "http://localhost:11434/v1",
    });

    await collectText(provider);

    expect(fetchMock).toHaveBeenCalledTimes(1);
    const init = fetchMock.mock.calls[0]?.[1] as { dispatcher?: unknown } | undefined;
    expect(init?.dispatcher).toBeUndefined();
  });
});

describe("resolveCapabilities", () => {
  it("returns safe defaults when no capabilities configured", () => {
    const caps = resolveCapabilities(undefined);
    expect(caps.useResponsesApi).toBe(false);
    expect(caps.reasoning).toBe(false);
    expect(caps.supportsTemperature).toBe(true);
    expect(caps.defaultMaxTokens).toBe(4096);
  });

  it("uses explicit config for reasoning model", () => {
    const explicit: ModelCapabilities = {
      useResponsesApi: true,
      reasoning: true,
      supportsTemperature: false,
      defaultMaxTokens: 16384,
    };
    const caps = resolveCapabilities(explicit);
    expect(caps.useResponsesApi).toBe(true);
    expect(caps.reasoning).toBe(true);
    expect(caps.supportsTemperature).toBe(false);
    expect(caps.defaultMaxTokens).toBe(16384);
  });

  it("partial config fills gaps with safe defaults", () => {
    const explicit: ModelCapabilities = { defaultMaxTokens: 8192 };
    const caps = resolveCapabilities(explicit);
    expect(caps.useResponsesApi).toBe(false);
    expect(caps.reasoning).toBe(false);
    expect(caps.supportsTemperature).toBe(true);
    expect(caps.defaultMaxTokens).toBe(8192);
  });

  it("explicit false is preserved", () => {
    const explicit: ModelCapabilities = {
      reasoning: false,
      supportsTemperature: false,
    };
    const caps = resolveCapabilities(explicit);
    expect(caps.reasoning).toBe(false);
    expect(caps.supportsTemperature).toBe(false);
  });

  it("supports high token limits for large-context models", () => {
    const explicit: ModelCapabilities = { defaultMaxTokens: 131072 };
    const caps = resolveCapabilities(explicit);
    expect(caps.defaultMaxTokens).toBe(131072);
  });
});

async function collectText(provider: ReturnType<typeof createOpenAIProvider>): Promise<string> {
  let output = "";
  for await (const chunk of provider.chat([
    { role: MessageRole.USER, content: "ping" },
  ])) {
    if (chunk.type === "text") {
      output += chunk.content;
    }
  }
  return output;
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

function makeStreamingResponse(): Response {
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      controller.enqueue(encoder.encode(
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":0,"model":"gpt-4o","choices":[{"index":0,"delta":{"role":"assistant","content":"pong"},"finish_reason":null}]}\n\n',
      ));
      controller.enqueue(encoder.encode(
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":0,"model":"gpt-4o","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n',
      ));
      controller.enqueue(encoder.encode(
        "data: [DONE]\n\n",
      ));
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

describe("stripNullArgs", () => {
  it("removes null-valued keys (OpenAI strict schema sends null for unused optionals)", () => {
    const args = {
      path: "src/x.cpp",
      search: "std.core.",
      replace: "std:core.",
      replacements: null,
      all: null,
      expected_replacements: null,
    };
    const cleaned = stripNullArgs(args);
    expect(cleaned).toEqual({
      path: "src/x.cpp",
      search: "std.core.",
      replace: "std:core.",
    });
    expect("replacements" in cleaned).toBe(false);
    expect("all" in cleaned).toBe(false);
  });

  it("preserves non-null values including false, 0, and empty string", () => {
    const args = {
      path: "file.ts",
      all: false,
      count: 0,
      note: "",
      data: [1, 2],
    };
    const cleaned = stripNullArgs(args);
    expect(cleaned).toEqual(args);
  });

  it("returns empty object when all values are null", () => {
    const cleaned = stripNullArgs({ a: null, b: null });
    expect(cleaned).toEqual({});
  });

  it("preserves nested strict schemas required by OpenAI tools", () => {
    const tool = createDelegateTool({
      provider: createOpenAIProvider({ model: "gpt-4o", apiKey: "test-key" }),
      tools: new ToolRegistry(),
      bus: new EventBus(),
      approvalGate: new ApprovalGate({
        mode: ApprovalMode.FULL_AUTO,
        auditLog: false,
        toolOverrides: {},
        pathRules: [],
      }, new EventBus()),
      config: {
        provider: "openai",
        model: "gpt-4o",
        providers: {},
        approval: {
          mode: ApprovalMode.FULL_AUTO,
          auditLog: false,
          toolOverrides: {},
          pathRules: [],
        },
        budget: {
          maxIterations: 10,
          maxContextTokens: 100_000,
          responseHeadroom: 2_000,
          costWarningThreshold: 1,
          enableCostTracking: true,
        },
        context: {
          pruningStrategy: "hybrid",
          triggerRatio: 0.9,
          keepRecentMessages: 40,
          turnIsolation: true,
          midpointBriefingInterval: 15,
          briefingStrategy: "auto",
        },
        arkts: {
          enabled: false,
          strictMode: false,
          targetVersion: "5.0",
        },
      },
      repoRoot: "/tmp",
      agentRegistry: new AgentRegistry(),
      parentAgentId: "root",
    });

    const converted = convertTools([tool], { strict: true });
    const schema = (converted["delegate"] as { inputSchema: { jsonSchema: Record<string, unknown> } }).inputSchema.jsonSchema;
    const requestSchema = (schema.properties as Record<string, Record<string, unknown>>)["request"];

    expect(requestSchema.type).toBe("object");
    expect(requestSchema.additionalProperties).toBe(false);
    expect(requestSchema.required).toEqual(["objective"]);
  });
});
