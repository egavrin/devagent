import { MessageRole } from "@devagent/runtime";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createDefaultRegistry } from "./index.js";

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

describe("createDefaultRegistry basics", () => {
  it("registers devagent-api as a built-in provider", () => {
    const registry = createDefaultRegistry();

    expect(registry.has("devagent-api")).toBe(true);
    expect(registry.list()).toContain("devagent-api");
  });
});

describe("devagent-api registry provider", () => {
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

  it("keeps devagent-api request rewriting when proxy env vars are configured for loopback", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeChatStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;
    setProxyEnv(originalProxyEnv, {
      HTTPS_PROXY: "https://proxy.example.com:8443",
    });

    const registry = createDefaultRegistry();
    const provider = registry.get("devagent-api", {
      model: "cortex",
      apiKey: "test-key",
      baseUrl: "http://localhost:8080/v1",
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
    expect(init.dispatcher).toBeUndefined();
    expect(body.messages?.map((message) => message.role)).toContain("system");
    expect(body.messages?.map((message) => message.role)).not.toContain("developer");
  });
});

describe("DeepSeek registry provider", () => {
  it("forces DeepSeek through chat completions", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeChatStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const registry = createDefaultRegistry();
    const provider = registry.get("deepseek", {
      model: "deepseek-chat",
      apiKey: "test-key",
      capabilities: {
        useResponsesApi: false,
        reasoning: true,
        supportsTemperature: false,
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
  });

  it("rewrites developer messages to system for DeepSeek", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeChatStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const registry = createDefaultRegistry();
    const provider = registry.get("deepseek", {
      model: "deepseek-chat",
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

  it("keeps DeepSeek request rewriting when proxy env vars are configured for loopback", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeChatStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;
    setProxyEnv(originalProxyEnv, {
      HTTPS_PROXY: "https://proxy.example.com:8443",
    });

    const registry = createDefaultRegistry();
    const provider = registry.get("deepseek", {
      model: "deepseek-chat",
      apiKey: "test-key",
      baseUrl: "http://localhost:8080/v1",
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
    const body = JSON.parse(String(init.body ?? "{}")) as {
      messages?: Array<{ role?: string }>;
    };
    expect(init.dispatcher).toBeUndefined();
    expect(body.messages?.map((message) => message.role)).toContain("system");
    expect(body.messages?.map((message) => message.role)).not.toContain("developer");
  });

  it("streams DeepSeek reasoning content and split tool calls", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeDeepSeekToolStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const registry = createDefaultRegistry();
    const provider = registry.get("deepseek", {
      model: "deepseek-v4-pro",
      apiKey: "test-key",
      capabilities: {
        useResponsesApi: false,
        reasoning: true,
        supportsTemperature: false,
      },
      reasoningEffort: "high",
    });

    const chunks = await collectChunks(provider.chat([
      { role: MessageRole.USER, content: "where am i?" },
    ], [makeLocationTool()]));

    expect(chunks).toContainEqual({ type: "thinking", content: "I should inspect the cwd." });
    expect(chunks).toContainEqual({ type: "text", content: "Let me check." });
    expect(chunks).toContainEqual({
      type: "tool_call",
      content: "{\"cmd\":\"pwd\"}",
      toolCallId: "call_1",
      toolName: "run_command",
    });
    expect(chunks.at(-1)).toMatchObject({
      type: "done",
      usage: { promptTokens: 12, completionTokens: 34 },
    });

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body ?? "{}")) as Record<string, unknown>;
    expect(body["thinking"]).toEqual({ type: "enabled" });
    expect(body["reasoning_effort"]).toBe("high");
    expect(body).not.toHaveProperty("temperature");
    expect(body["tools"]).toBeDefined();
  });

  it("only replays DeepSeek reasoning_content for assistant tool calls", async () => {
    const fetchMock = vi.fn().mockResolvedValue(makeChatStreamingResponse());
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const registry = createDefaultRegistry();
    const provider = registry.get("deepseek", {
      model: "deepseek-v4-pro",
      apiKey: "test-key",
      capabilities: {
        useResponsesApi: false,
        reasoning: true,
        supportsTemperature: false,
      },
    });

    await collectChunks(provider.chat([
      { role: MessageRole.USER, content: "where am i?" },
      {
        role: MessageRole.ASSISTANT,
        content: "Let me check.",
        thinking: "I should inspect the cwd.",
        toolCalls: [{ name: "run_command", arguments: { cmd: "pwd" }, callId: "call_1" }],
      },
      { role: MessageRole.TOOL, toolCallId: "call_1", content: "/tmp/project" },
      {
        role: MessageRole.ASSISTANT,
        content: "Final answer.",
        thinking: "This should stay local after the turn.",
      },
      { role: MessageRole.USER, content: "what next?" },
    ]));

    const body = JSON.parse(String(fetchMock.mock.calls[0]?.[1]?.body ?? "{}")) as {
      messages?: Array<Record<string, unknown>>;
    };
    expect(body.messages?.[1]).toMatchObject({
      role: "assistant",
      content: "Let me check.",
      reasoning_content: "I should inspect the cwd.",
      tool_calls: [{
        id: "call_1",
        type: "function",
        function: {
          name: "run_command",
          arguments: "{\"cmd\":\"pwd\"}",
        },
      }],
    });
    expect(body.messages?.[3]).toEqual({
      role: "assistant",
      content: "Final answer.",
    });
  });

  it("classifies DeepSeek JSON errors without leaking credentials", async () => {
    const fetchMock = vi.fn().mockResolvedValue(new Response(
      JSON.stringify({ error: { message: "The `reasoning_content` in the thinking mode must be passed back to the API." } }),
      { status: 400, headers: { "content-type": "application/json" } },
    ));
    globalThis.fetch = fetchMock as typeof globalThis.fetch;

    const registry = createDefaultRegistry();
    const provider = registry.get("deepseek", {
      model: "deepseek-v4-pro",
      apiKey: "secret-test-key",
      capabilities: {
        useResponsesApi: false,
        reasoning: true,
        supportsTemperature: false,
      },
    });

    let message = "";
    try {
      await collectChunks(provider.chat([
        { role: MessageRole.USER, content: "ping" },
      ]));
    } catch (err) {
      message = err instanceof Error ? err.message : String(err);
    }

    expect(message).toContain("reasoning_content");
    expect(message).not.toContain("secret-test-key");
  });
});

async function collectChunks(stream: AsyncIterable<unknown>): Promise<unknown[]> {
  const chunks: unknown[] = [];
  for await (const chunk of stream) {
    chunks.push(chunk);
  }
  return chunks;
}

function makeLocationTool() {
  return {
    name: "run_command",
    description: "Run a command",
    category: "readonly" as const,
    paramSchema: {
      type: "object" as const,
      properties: { cmd: { type: "string" } },
      required: ["cmd"],
    },
    resultSchema: { type: "object" as const },
    handler: async () => ({
      success: true,
      output: "",
      error: null,
      artifacts: [],
    }),
  };
}

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

function makeDeepSeekToolStreamingResponse(): Response {
  const stream = new ReadableStream({
    start(controller) {
      const encoder = new TextEncoder();
      controller.enqueue(encoder.encode(
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":0,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"I should inspect the cwd."},"finish_reason":null}]}\n\n',
      ));
      controller.enqueue(encoder.encode(
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":0,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"content":"Let me check."},"finish_reason":null}]}\n\n',
      ));
      controller.enqueue(encoder.encode(
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":0,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"run_command","arguments":"{\\"cmd\\""}}]},"finish_reason":null}]}\n\n',
      ));
      controller.enqueue(encoder.encode(
        'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":0,"model":"deepseek-v4-pro","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":":\\"pwd\\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":34}}\n\n',
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
