/**
 * Anthropic provider — uses Vercel AI SDK with @ai-sdk/anthropic.
 * Supports streaming, tool calling, and prompt caching.
 */

import { createAnthropic } from "@ai-sdk/anthropic";
import { streamText } from "ai";
import type { LLMProvider, ProviderConfig, Message, ToolSpec, StreamChunk } from "@devagent/runtime";
import { ProviderError } from "@devagent/runtime";
import { convertMessages, convertTools, processProviderStream } from "./shared.js";

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

      const aiMessages = convertMessages(messages);
      const aiTools = tools ? convertTools(tools) : undefined;

      try {
        const caps = config.capabilities;
        const defaultMaxTokens = caps?.defaultMaxTokens ?? 4096;
        const supportsTemp = caps?.supportsTemperature ?? true;

        const result = streamText({
          model: anthropic(config.model),
          messages: aiMessages,
          tools: aiTools,
          maxTokens: config.maxTokens ?? defaultMaxTokens,
          ...(supportsTemp ? { temperature: config.temperature ?? 0 } : {}),
          abortSignal: abortController.signal,
        });

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

