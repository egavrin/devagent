import { afterEach, describe, expect, it, vi } from "vitest";
import { MessageRole } from "@devagent/runtime";
import { createDefaultRegistry } from "./index.js";

describe("createDefaultRegistry", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
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
