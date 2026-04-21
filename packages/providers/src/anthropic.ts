/**
 * Anthropic provider — uses Vercel AI SDK with @ai-sdk/anthropic.
 * Supports streaming, tool calling, and prompt caching.
 */

import { createAnthropic } from "@ai-sdk/anthropic";
import { ProviderError } from "@devagent/runtime";
import { streamText, type TextStreamPart } from "ai";

import { classifyProviderError, convertMessages, convertTools, processProviderStream } from "./shared.js";
import type { LLMProvider, ProviderConfig, Message, ToolSpec, StreamChunk } from "@devagent/runtime";

export function createAnthropicProvider(config: ProviderConfig): LLMProvider {
  if (!config.apiKey) {
    throw new ProviderError("Anthropic provider requires an API key");
  }

  const anthropic = createAnthropic({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
  });

  let abortController: AbortController | null = null;

  return {
    id: "anthropic",
    async *chat(
      messages: ReadonlyArray<Message>,
      tools?: ReadonlyArray<ToolSpec>,
    ): AsyncIterable<StreamChunk> {
      abortController = new AbortController();

      try {
        const result = startAnthropicStream(config, messages, tools, abortController, anthropic);
        yield* processProviderStream({
          providerName: "Anthropic",
          fullStream: result.fullStream,
          abortController,
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

function startAnthropicStream(
  config: ProviderConfig,
  messages: ReadonlyArray<Message>,
  tools: ReadonlyArray<ToolSpec> | undefined,
  abortController: AbortController,
  anthropic: ReturnType<typeof createAnthropic>,
): { readonly fullStream: AsyncIterable<TextStreamPart<Record<string, never>>> } {
  const caps = config.capabilities;
  try {
    return streamText({
      model: anthropic(config.model),
      messages: convertMessages(messages),
      tools: tools ? convertTools(tools) : undefined,
      maxOutputTokens: config.maxTokens ?? caps?.defaultMaxTokens ?? 4096,
      ...(caps?.supportsTemperature ?? true ? { temperature: config.temperature ?? 0 } : {}),
      abortSignal: abortController.signal,
    });
  } catch (err) {
    throw classifyProviderError(err, "Anthropic");
  }
}
