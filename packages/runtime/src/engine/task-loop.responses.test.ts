import { resolve } from "node:path";
import {
  it,
  expect,
  beforeEach,
  beforeAll,
} from "vitest";

import { TaskLoop } from "./task-loop.js";
import {
  createMockProvider,
  makeConfig,
  makeEchoTool,
} from "./task-loop.test-helpers.js";
import type {
  LLMProvider,
  Message,
  StreamChunk,
} from "../core/index.js";
import {
  EventBus,
  ApprovalGate,
  MessageRole,
  ProviderError,
  ProviderTlsCertificateError,
  loadModelRegistry,
} from "../core/index.js";
import { ToolRegistry } from "../tools/index.js";

let bus: EventBus;
let config: ReturnType<typeof makeConfig>;


beforeAll(() => {
    const modelsDir = resolve(import.meta.dirname ?? new URL(".", import.meta.url).pathname, "../../../../models");
    loadModelRegistry(undefined, [modelsDir]);
  });

beforeEach(() => {
    bus = new EventBus();
    config = makeConfig();
  });

  it("propagates ProviderError when provider throws", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "failing",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        throw new ProviderError("Connection refused");
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    await expect(loop.run("hello")).rejects.toThrow(ProviderError);
    // 1 initial + 3 retries = 4 total calls
    expect(callCount).toBe(4);
  });
  it("retries on ProviderError and succeeds on second attempt", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "flaky",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount === 1) {
          throw new ProviderError("Temporary failure");
        }
        yield { type: "text", content: "Success after retry" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("hello");
    // callCount = 1 (fail) + 1 (retry success) = 2
    expect(callCount).toBe(2);
    expect(result.iterations).toBe(1);

    const assistantMsgs = result.messages.filter(
      (m) => m.role === MessageRole.ASSISTANT,
    );
    expect(assistantMsgs[assistantMsgs.length - 1]!.content).toBe(
      "Success after retry",
    );
  });
  it("throws after exhausting all retry attempts", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "always-failing",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        throw new ProviderError("Persistent failure");
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    await expect(loop.run("hello")).rejects.toThrow("Persistent failure");
    expect(callCount).toBe(4); // 1 initial + 3 retries
  });
  it("does not retry non-ProviderError exceptions", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "type-error",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        throw new TypeError("Cannot read property 'x' of undefined");
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    await expect(loop.run("hello")).rejects.toThrow(TypeError);
    expect(callCount).toBe(1); // No retry
  });
  it("returns success status on normal completion", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "All done!" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Hello");
    expect(result.status).toBe("success");
    expect(result.lastText).toBe("All done!");
  });
  it("retries with summary request on empty response after tool calls", async () => {
    const provider = createMockProvider([
      // First: tool call
      [
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Second: empty response (no text, no tool calls)
      [{ type: "done", content: "" }],
      // Third: response after summary request injection
      [
        { type: "text", content: "Here is the summary" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Do the thing");
    expect(result.status).toBe("success");
    expect(result.lastText).toBe("Here is the summary");
  });
  it("returns empty_response when summary retry also produces no text", async () => {
    const provider = createMockProvider([
      // First: tool call
      [
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Second: empty response
      [{ type: "done", content: "" }],
      // Third: still empty after summary request
      [{ type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Do the thing");
    expect(result.status).toBe("empty_response");
  });
  it("retries once when the first response is empty and succeeds on the second response", async () => {
    const seenMessages: Message[][] = [];
    const provider: LLMProvider = {
      id: "capture",
      async *chat(messages): AsyncIterable<StreamChunk> {
        seenMessages.push([...messages]);
        if (seenMessages.length === 1) {
          yield { type: "done", content: "" };
          return;
        }
        yield { type: "text", content: "Recovered final artifact" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Return the artifact");
    expect(result.status).toBe("success");
    expect(result.lastText).toBe("Recovered final artifact");
    expect(seenMessages).toHaveLength(2);
    expect(seenMessages[1]?.some(
      (message) =>
        message.role === MessageRole.SYSTEM
        && message.content?.includes("Your previous response was empty"),
    )).toBe(true);
  });
  it("returns empty_response when the initial empty response repeats", async () => {
    const provider = createMockProvider([
      [{ type: "done", content: "" }],
      [{ type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Return the artifact");
    expect(result.status).toBe("empty_response");
    expect(result.lastText).toBeNull();
  });
  it("tracks lastText across tool call iterations", async () => {
    const provider = createMockProvider([
      // First: text + tool call
      [
        { type: "text", content: "Let me check..." },
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      // Second: final text
      [
        { type: "text", content: "Final answer" },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Do something");
    expect(result.status).toBe("success");
    expect(result.lastText).toBe("Final answer");
  });
  it("continues after a text-only progress update when the final-text validator rejects it", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Progress: I've inspected the files." },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: '{"structured":{"summary":"ok","issues":[]},"rendered":"# ok"}' },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
      finalTextValidator: (candidate) => ({
        valid: candidate.trim().startsWith("{"),
        retryMessage: "Return the final JSON artifact now.",
      }),
    });

    const result = await loop.run("Do something");

    expect(result.status).toBe("success");
    expect(result.iterations).toBe(3);
    expect(result.lastText).toBe('{"structured":{"summary":"ok","issues":[]},"rendered":"# ok"}');
    expect(result.messages.filter((m) => m.role === MessageRole.ASSISTANT).map((m) => m.content)).toContain(
      "Progress: I've inspected the files.",
    );
    expect(result.messages.filter((m) => m.role === MessageRole.SYSTEM).map((m) => m.content)).toContain(
      "Return the final JSON artifact now.",
    );
  });
  it("accepts a text-only progress update when no final-text validator is configured", async () => {
    const provider = createMockProvider([
      [
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Progress: I've inspected the files." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Do something");

    expect(result.status).toBe("success");
    expect(result.iterations).toBe(2);
    expect(result.lastText).toBe("Progress: I've inspected the files.");
  });
  it("allows a per-run final-text validator override", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "Not structured yet." },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: '{"structured":{"summary":"ok"},"rendered":"done"}' },
        { type: "done", content: "" },
      ],
      [
        { type: "text", content: "Plain text is fine again." },
        { type: "done", content: "" },
      ],
    ]);

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const validated = await loop.run("First turn", {
      finalTextValidator: (candidate) => ({
        valid: candidate.trim().startsWith("{"),
        retryMessage: "Return JSON.",
      }),
    });
    expect(validated.lastText).toBe('{"structured":{"summary":"ok"},"rendered":"done"}');

    const plain = await loop.run("Second turn");
    expect(plain.lastText).toBe("Plain text is fine again.");
  });
  it("does not retain text from assistant turns that also contain tool calls as lastText", async () => {
    const provider = createMockProvider([
      [
        { type: "text", content: "Invoked testing and execute-contract." },
        {
          type: "tool_call",
          content: '{"text": "test"}',
          toolCallId: "call_0",
          toolName: "echo",
        },
        { type: "done", content: "" },
      ],
      [{ type: "done", content: "" }],
      [{ type: "done", content: "" }],
    ]);

    const registry = new ToolRegistry();
    registry.register(makeEchoTool());

    const gate = new ApprovalGate(config.approval, bus);
    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("Do something");

    expect(result.status).toBe("empty_response");
    expect(result.lastText).toBeNull();
  });
  it("emits PROVIDER_RETRY error events during retries", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "retry-events",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount <= 2) {
          throw new ProviderError("Transient error");
        }
        yield { type: "text", content: "Finally works" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);

    const retryEvents: Array<{ message: string; code: string; fatal: boolean }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "PROVIDER_RETRY") {
        retryEvents.push(event as { message: string; code: string; fatal: boolean });
      }
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("hello");
    // callCount = 2 (transient errors) + 1 (success) = 3
    expect(callCount).toBe(3);
    expect(retryEvents.length).toBe(2);
    expect(retryEvents[0]!.code).toBe("PROVIDER_RETRY");
    expect(retryEvents[0]!.fatal).toBe(false);
    expect(retryEvents[1]!.code).toBe("PROVIDER_RETRY");
    expect(result.iterations).toBe(1);
  });
  it("emits TLS-specific retry guidance for certificate verification failures", async () => {
    let callCount = 0;
    const provider: LLMProvider = {
      id: "tls-retry-events",
      async *chat(): AsyncIterable<StreamChunk> {
        callCount++;
        if (callCount <= 2) {
          throw new ProviderTlsCertificateError("MockProvider", "self-signed certificate in certificate chain");
        }
        yield { type: "text", content: "Finally works" };
        yield { type: "done", content: "" };
      },
      abort() {},
    };

    const registry = new ToolRegistry();
    const gate = new ApprovalGate(config.approval, bus);

    const retryEvents: Array<{ message: string; code: string; fatal: boolean }> = [];
    bus.on("error", (event) => {
      if ((event as { code?: string }).code === "PROVIDER_RETRY") {
        retryEvents.push(event as { message: string; code: string; fatal: boolean });
      }
    });

    const loop = new TaskLoop({
      provider,
      tools: registry,
      bus,
      approvalGate: gate,
      config,
      systemPrompt: "Test",
      repoRoot: "/tmp",
    });

    const result = await loop.run("hello");

    expect(callCount).toBe(3);
    expect(retryEvents.length).toBe(2);
    expect(retryEvents[0]!.message).toContain("certificate verification failed");
    expect(retryEvents[0]!.message).toContain("NODE_EXTRA_CA_CERTS");
    expect(retryEvents[0]!.message).toContain("HTTPS_PROXY/HTTP_PROXY/NO_PROXY");
    expect(result.iterations).toBe(1);
  });
