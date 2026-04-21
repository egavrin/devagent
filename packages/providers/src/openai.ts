/**
 * OpenAI provider — uses Vercel AI SDK with @ai-sdk/openai.
 * Supports streaming, tool calling, and ChatGPT subscriptions.
 *
 * Model capabilities (Responses API, reasoning, temperature, maxTokens)
 * are driven by config.capabilities. No heuristics — configure explicitly.
 */

import { createOpenAI } from "@ai-sdk/openai";
import { streamText, type TextStreamPart } from "ai";
import { randomUUID } from "node:crypto";

import { createProxyAwareFetch, hasProxyEnv } from "./network.js";
import {
  classifyProviderError,
  convertMessages,
  convertTools,
  processProviderStream,
  resolveCapabilities,
  stripNullArgs,
} from "./shared.js";
import type { LLMProvider, ProviderConfig, Message, ToolSpec, StreamChunk } from "@devagent/runtime";

// Re-export shared utilities so existing consumers (tests, index.ts) don't break.
export { resolveCapabilities, stripNullArgs } from "./shared.js";

interface OpenAIRewriteOptions {
  readonly codexOpts: ProviderConfig["codexOptions"];
  readonly fieldsToStrip: ProviderConfig["stripFields"];
  readonly messageRoleOverrides: ProviderConfig["messageRoleOverrides"];
  readonly requestIdHeaderName: ProviderConfig["requestIdHeaderName"];
  readonly transportFetch: FetchLike;
}

type FetchLike = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;

interface OpenAIProviderInternals {
  readonly codexOpts: ProviderConfig["codexOptions"];
  readonly openai: ReturnType<typeof createOpenAI>;
}

export function createOpenAIProvider(config: ProviderConfig): LLMProvider {
  // Build custom headers — merge customHeaders + OAuth-specific headers
  const headers = buildOpenAIHeaders(config);

  // Custom fetch wrapper — handles two provider-specific needs:
  // 1. ChatGPT Codex: inject fields (store, include) and remove max_output_tokens
  // 2. Copilot: strip fields the endpoint rejects (store, metadata, prediction, etc.)
  const needsCustomFetch = needsOpenAIRewriteFetch(config);
  const transportFetch = createProxyAwareFetch(globalThis.fetch);
  const customFetch = needsCustomFetch
    ? createOpenAIRewriteFetch({
        codexOpts: config.codexOptions,
        fieldsToStrip: config.stripFields,
        messageRoleOverrides: config.messageRoleOverrides,
        requestIdHeaderName: config.requestIdHeaderName,
        transportFetch,
      })
    : undefined;

  // apiKey is optional — local endpoints (Ollama, LM Studio) don't require auth.
  // When oauthToken is present, set apiKey to "unused" (SDK requires non-empty).
  const internals: OpenAIProviderInternals = {
    codexOpts: config.codexOptions,
    openai: createOpenAI({
      apiKey: config.oauthToken ? "unused" : (config.apiKey ?? ""),
      baseURL: config.baseUrl,
      ...(Object.keys(headers).length > 0 ? { headers } : {}),
      ...(customFetch ? { fetch: customFetch as unknown as typeof globalThis.fetch } : {}),
    }),
  };

  let abortController: AbortController | null = null;

  return {
    id: "openai",
    async *chat(
      messages: ReadonlyArray<Message>,
      tools?: ReadonlyArray<ToolSpec>,
    ): AsyncIterable<StreamChunk> {
      abortController = new AbortController();

      try {
        const result = startOpenAIStream(config, messages, tools, abortController, internals);
        yield* processProviderStream({
          providerName: "OpenAI",
          fullStream: result.fullStream,
          abortController,
          transformArgs: stripNullArgs,
        });
      } finally {
        abortController = null;
      }
    },

    abort(): void {
      abortController?.abort();
    },
  };
}

function needsOpenAIRewriteFetch(config: ProviderConfig): boolean {
  return Boolean(
    config.codexOptions
      || config.stripFields?.length
      || config.requestIdHeaderName
      || Object.keys(config.messageRoleOverrides ?? {}).length
      || hasProxyEnv(),
  );
}

function startOpenAIStream(
  config: ProviderConfig,
  messages: ReadonlyArray<Message>,
  tools: ReadonlyArray<ToolSpec> | undefined,
  abortController: AbortController,
  internals: OpenAIProviderInternals,
): { readonly fullStream: AsyncIterable<TextStreamPart<Record<string, never>>> } {
  const caps = resolveCapabilities(config.capabilities);
  const providerOptions = buildOpenAIProviderOptions(config, internals.codexOpts);
  try {
    return streamText({
      model: caps.useResponsesApi
        ? internals.openai.responses(config.model)
        : internals.openai.chat(config.model),
      messages: convertMessages(messages),
      tools: tools ? convertTools(tools, { strict: true }) : undefined,
      maxOutputTokens: config.maxTokens ?? caps.defaultMaxTokens,
      ...(caps.supportsTemperature ? { temperature: config.temperature ?? 0 } : {}),
      abortSignal: abortController.signal,
      ...(providerOptions ? { providerOptions: { openai: providerOptions } } : {}),
    });
  } catch (err) {
    throw classifyProviderError(err, "OpenAI");
  }
}

function buildOpenAIProviderOptions(
  config: ProviderConfig,
  codexOpts: ProviderConfig["codexOptions"],
): Record<string, string | number | boolean | null> | null {
  const options: Record<string, string | number | boolean | null> = {};
  if (config.reasoningEffort) options["reasoningEffort"] = config.reasoningEffort;
  if (codexOpts?.store !== undefined) options["store"] = codexOpts.store;
  if (codexOpts?.instructions) options["instructions"] = codexOpts.instructions;
  return Object.keys(options).length > 0 ? options : null;
}

function buildOpenAIHeaders(config: ProviderConfig): Record<string, string> {
  const headers: Record<string, string> = { ...(config.customHeaders ?? {}) };
  if (!config.oauthToken) return headers;
  headers["Authorization"] = `Bearer ${config.oauthToken}`;
  if (config.oauthAccountId) {
    headers["ChatGPT-Account-Id"] = config.oauthAccountId;
  }
  headers["openai-beta"] = "responses=experimental";
  return headers;
}

function createOpenAIRewriteFetch(options: OpenAIRewriteOptions): typeof globalThis.fetch {
  const rewriteFetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const nextHeaders = withRequestId(init?.headers, options.requestIdHeaderName);
    const nextInit = rewriteOpenAIInit(init, nextHeaders, options);
    return options.transportFetch(input, nextInit);
  };
  return withFetchPreconnect(rewriteFetch);
}

function withFetchPreconnect(fetchFn: FetchLike): typeof globalThis.fetch {
  const fallbackPreconnect = globalThis.fetch.preconnect?.bind(globalThis.fetch) ?? (() => {});
  return Object.assign(fetchFn, {
    preconnect: fallbackPreconnect,
  });
}

function withRequestId(
  headers: HeadersInit | undefined,
  requestIdHeaderName: string | undefined,
): Headers {
  const nextHeaders = new Headers(headers);
  if (requestIdHeaderName && !nextHeaders.has(requestIdHeaderName)) {
    nextHeaders.set(requestIdHeaderName, randomUUID());
  }
  return nextHeaders;
}

function rewriteOpenAIInit(
  init: RequestInit | undefined,
  headers: Headers,
  options: OpenAIRewriteOptions,
): RequestInit {
  if (!init?.body || typeof init.body !== "string") {
    return { ...init, headers };
  }
  const body = parseRequestBody(init.body);
  if (!body) return { ...init, headers };
  applyCodexOptions(body, options.codexOpts);
  stripRejectedFields(body, options.fieldsToStrip);
  rewriteMessageRoles(body, options.messageRoleOverrides);
  return { ...init, headers, body: JSON.stringify(body) };
}

function parseRequestBody(body: string): Record<string, unknown> | null {
  try {
    return JSON.parse(body) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function applyCodexOptions(
  body: Record<string, unknown>,
  codexOpts: ProviderConfig["codexOptions"],
): void {
  if (!codexOpts) return;
  if (codexOpts.store !== undefined) body["store"] = codexOpts.store;
  if (Array.isArray(codexOpts.include)) body["include"] = [...codexOpts.include];
  delete body["max_output_tokens"];
}

function stripRejectedFields(
  body: Record<string, unknown>,
  fieldsToStrip: ProviderConfig["stripFields"],
): void {
  for (const field of fieldsToStrip ?? []) {
    delete body[field];
  }
}

function rewriteMessageRoles(
  body: Record<string, unknown>,
  overrides: ProviderConfig["messageRoleOverrides"],
): void {
  if (!overrides || !Array.isArray(body["messages"])) return;
  body["messages"] = body["messages"].map((entry) => rewriteMessageRole(entry, overrides));
}

function rewriteMessageRole(
  entry: unknown,
  overrides: NonNullable<ProviderConfig["messageRoleOverrides"]>,
): unknown {
  if (!entry || typeof entry !== "object") return entry;
  const message = { ...(entry as Record<string, unknown>) };
  const currentRole = message["role"];
  if (typeof currentRole === "string" && overrides[currentRole]) {
    message["role"] = overrides[currentRole];
  }
  return message;
}
