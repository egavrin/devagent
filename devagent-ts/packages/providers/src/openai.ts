/**
 * OpenAI provider — uses Vercel AI SDK with @ai-sdk/openai.
 * Supports streaming, tool calling, and ChatGPT subscriptions.
 */

import { createOpenAI } from "@ai-sdk/openai";
import { streamText, tool as aiTool, jsonSchema, type CoreMessage } from "ai";
import type { LLMProvider, ProviderConfig, Message, ToolSpec, StreamChunk } from "@devagent/core";
import { MessageRole, ProviderError } from "@devagent/core";

export function createOpenAIProvider(config: ProviderConfig): LLMProvider {
  if (!config.apiKey) {
    throw new ProviderError("OpenAI provider requires an API key");
  }

  const openai = createOpenAI({
    apiKey: config.apiKey,
    baseURL: config.baseUrl,
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
      const aiTools = tools ? convertTools(tools) : undefined;

      try {
        const result = streamText({
          model: openai(config.model),
          messages: aiMessages,
          tools: aiTools,
          maxTokens: config.maxTokens ?? 4096,
          temperature: config.temperature ?? 0,
          abortSignal: abortController.signal,
        });

        for await (const part of result.fullStream) {
          switch (part.type) {
            case "text-delta":
              yield {
                type: "text",
                content: part.textDelta,
              };
              break;

            case "tool-call":
              yield {
                type: "tool_call",
                content: JSON.stringify(part.args),
                toolCallId: part.toolCallId,
                toolName: part.toolName,
              };
              break;

            case "error":
              yield {
                type: "error",
                content: String(part.error),
              };
              break;

            case "finish":
              yield {
                type: "done",
                content: "",
              };
              break;
          }
        }
      } catch (err) {
        if (abortController.signal.aborted) {
          yield { type: "done", content: "" };
          return;
        }
        const msg = err instanceof Error ? err.message : String(err);
        throw new ProviderError(`OpenAI API error: ${msg}`);
      } finally {
        abortController = null;
      }
    },

    abort(): void {
      abortController?.abort();
    },
  };
}

// ─── Message Conversion ──────────────────────────────────────

function convertMessages(messages: ReadonlyArray<Message>): CoreMessage[] {
  const result: CoreMessage[] = [];

  for (const msg of messages) {
    switch (msg.role) {
      case MessageRole.SYSTEM:
        result.push({ role: "system", content: msg.content ?? "" });
        break;
      case MessageRole.USER:
        result.push({ role: "user", content: msg.content ?? "" });
        break;
      case MessageRole.ASSISTANT:
        if (msg.toolCalls && msg.toolCalls.length > 0) {
          // Assistant message with tool calls — use content parts format
          const parts: Array<
            | { type: "text"; text: string }
            | { type: "tool-call"; toolCallId: string; toolName: string; args: Record<string, unknown> }
          > = [];
          if (msg.content) {
            parts.push({ type: "text" as const, text: msg.content });
          }
          for (const tc of msg.toolCalls) {
            parts.push({
              type: "tool-call" as const,
              toolCallId: tc.callId,
              toolName: tc.name,
              args: tc.arguments,
            });
          }
          result.push({ role: "assistant", content: parts });
        } else {
          result.push({ role: "assistant", content: msg.content ?? "" });
        }
        break;
      case MessageRole.TOOL:
        result.push({
          role: "tool",
          content: [
            {
              type: "tool-result" as const,
              toolCallId: msg.toolCallId ?? "",
              toolName: "", // Name resolved by SDK via toolCallId
              result: msg.content ?? "",
            },
          ],
        });
        break;
    }
  }

  return result;
}

// ─── Tool Conversion ────────────────────────────────────────

function convertTools(
  tools: ReadonlyArray<ToolSpec>,
): Record<string, ReturnType<typeof aiTool>> {
  const result: Record<string, ReturnType<typeof aiTool>> = {};

  for (const t of tools) {
    result[t.name] = aiTool({
      description: t.description,
      parameters: jsonSchema({
        type: t.paramSchema.type as "object",
        properties: (t.paramSchema.properties ?? {}) as Record<string, Record<string, unknown>>,
        required: [...(t.paramSchema.required ?? [])],
      }),
    });
  }

  return result;
}
