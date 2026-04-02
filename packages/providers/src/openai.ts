/**
 * OpenAI provider — uses Vercel AI SDK with @ai-sdk/openai.
 * Supports streaming, tool calling, and ChatGPT subscriptions.
 *
 * Model capabilities (Responses API, reasoning, temperature, maxTokens)
 * are driven by config.capabilities. No heuristics — configure explicitly.
 */

import { randomUUID } from "node:crypto";
import { createOpenAI } from "@ai-sdk/openai";
import { streamText } from "ai";
import type { LLMProvider, ProviderConfig, Message, ToolSpec, StreamChunk } from "@devagent/runtime";
import { ProviderError } from "@devagent/runtime";
import { classifyProviderError, convertMessages, convertTools, processProviderStream, resolveCapabilities, stripNullArgs } from "./shared.js";

// Re-export shared utilities so existing consumers (tests, index.ts) don't break.
export { resolveCapabilities, stripNullArgs } from "./shared.js";

/**
 * ChatGPT Codex-specific request body fields.
 * When set on ProviderConfig, these are injected into providerOptions
 * and via a custom fetch wrapper for fields the SDK doesn't natively support.
 */
export interface ChatGPTCodexOptions {
  /** Send `store: false` to prevent response storage (required by Codex endpoint). */
  readonly store?: boolean;
  /** Additional response `include` fields (e.g., ["reasoning.encrypted_content"]). */
  readonly include?: ReadonlyArray<string>;
  /** System-level instructions sent as top-level `instructions` field. */
  readonly instructions?: string;
}

export function createOpenAIProvider(config: ProviderConfig): LLMProvider {
  // Build custom headers — merge customHeaders + OAuth-specific headers
  const headers: Record<string, string> = { ...(config.customHeaders ?? {}) };
  if (config.oauthToken) {
    headers["Authorization"] = `Bearer ${config.oauthToken}`;
    if (config.oauthAccountId) {
      headers["ChatGPT-Account-Id"] = config.oauthAccountId;
    }
    headers["openai-beta"] = "responses=experimental";
  }

  // Custom fetch wrapper — handles two provider-specific needs:
  // 1. ChatGPT Codex: inject fields (store, include) and remove max_output_tokens
  // 2. Copilot: strip fields the endpoint rejects (store, metadata, prediction, etc.)
  const codexOpts = config.codexOptions;
  const fieldsToStrip = config.stripFields;
  const requestIdHeaderName = config.requestIdHeaderName;
  const needsCustomFetch = codexOpts || (fieldsToStrip && fieldsToStrip.length > 0) || requestIdHeaderName;
  const customFetch = needsCustomFetch
    ? async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        const nextHeaders = new Headers(init?.headers);
        if (requestIdHeaderName && !nextHeaders.has(requestIdHeaderName)) {
          nextHeaders.set(requestIdHeaderName, randomUUID());
        }

        if (init?.body && typeof init.body === "string") {
          try {
            const body = JSON.parse(init.body) as Record<string, unknown>;
            // ChatGPT Codex: inject required fields
            if (codexOpts) {
              if (codexOpts.store !== undefined) body["store"] = codexOpts.store;
              if (codexOpts.include && Array.isArray(codexOpts.include)) body["include"] = [...codexOpts.include];
              delete body["max_output_tokens"];
            }
            // Strip fields the endpoint rejects
            if (fieldsToStrip) {
              for (const field of fieldsToStrip) {
                delete body[field];
              }
            }
            init = { ...init, headers: nextHeaders, body: JSON.stringify(body) };
          } catch {
            init = { ...init, headers: nextHeaders };
          }
        } else {
          init = { ...init, headers: nextHeaders };
        }
        return globalThis.fetch(input, init);
      }
    : undefined;

  // apiKey is optional — local endpoints (Ollama, LM Studio) don't require auth.
  // When oauthToken is present, set apiKey to "unused" (SDK requires non-empty).
  const openai = createOpenAI({
    apiKey: config.oauthToken ? "unused" : (config.apiKey ?? ""),
    baseURL: config.baseUrl,
    ...(Object.keys(headers).length > 0 ? { headers } : {}),
    ...(customFetch ? { fetch: customFetch as unknown as typeof globalThis.fetch } : {}),
  });

  let abortController: AbortController | null = null;

  return {
    id: "openai",

    async *chat(
      messages: ReadonlyArray<Message>,
      tools?: ReadonlyArray<ToolSpec>,
    ): AsyncIterable<StreamChunk> {
      abortController = new AbortController();

      const aiMessages = convertMessages(messages);
      const aiTools = tools ? convertTools(tools, { strict: true }) : undefined;

      try {
        // Resolve capabilities from explicit config (no heuristics)
        const caps = resolveCapabilities(config.capabilities);

        const model = caps.useResponsesApi
          ? openai.responses(config.model)
          : openai(config.model);

        // Build providerOptions — merge reasoning effort with Codex-specific options
        const openaiProviderOpts: Record<string, string | number | boolean | null> = {};
        if (config.reasoningEffort) {
          openaiProviderOpts["reasoningEffort"] = config.reasoningEffort;
        }
        if (codexOpts?.store !== undefined) {
          openaiProviderOpts["store"] = codexOpts.store;
        }
        if (codexOpts?.instructions) {
          openaiProviderOpts["instructions"] = codexOpts.instructions;
        }
        const hasProviderOpts = Object.keys(openaiProviderOpts).length > 0;

        let result;
        try {
          result = streamText({
            model,
            messages: aiMessages,
            tools: aiTools,
            maxTokens: config.maxTokens ?? caps.defaultMaxTokens,
            ...(caps.supportsTemperature ? { temperature: config.temperature ?? 0 } : {}),
            abortSignal: abortController.signal,
            ...(hasProviderOpts
              ? { providerOptions: { openai: openaiProviderOpts } }
              : {}),
          });
        } catch (err) {
          throw classifyProviderError(err, "OpenAI");
        }

        // Explicit iteration instead of yield* to ensure errors are caught on Node.js
        const stream = processProviderStream({
          providerName: "OpenAI",
          fullStream: result.fullStream,
          abortController,
          transformArgs: stripNullArgs,
        });
        try {
          for await (const chunk of stream) {
            yield chunk;
          }
        } catch (err) {
          if (err instanceof ProviderError) throw err;
          throw classifyProviderError(err, "OpenAI");
        }
      } finally {
        abortController = null;
      }
    },

    abort(): void {
      abortController?.abort();
    },
  };
}

