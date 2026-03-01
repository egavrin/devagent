/**
 * OpenAI provider — uses Vercel AI SDK with @ai-sdk/openai.
 * Supports streaming, tool calling, and ChatGPT subscriptions.
 *
 * Model capabilities (Responses API, reasoning, temperature, maxTokens)
 * are driven by config.capabilities. No heuristics — configure explicitly.
 */

import { createOpenAI } from "@ai-sdk/openai";
import { streamText, tool as aiTool, jsonSchema, type CoreMessage } from "ai";
import type { LLMProvider, ProviderConfig, ModelCapabilities, Message, ToolSpec, StreamChunk } from "@devagent/core";
import { MessageRole, ProviderError } from "@devagent/core";

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

/**
 * Resolve model capabilities from explicit config.
 * No heuristics — if capabilities aren't configured, safe defaults apply.
 * Configure via TOML or ProviderConfig.capabilities.
 */
export function resolveCapabilities(explicit: ModelCapabilities | undefined): Required<ModelCapabilities> {
  return {
    useResponsesApi: explicit?.useResponsesApi ?? false,
    reasoning: explicit?.reasoning ?? false,
    supportsTemperature: explicit?.supportsTemperature ?? true,
    defaultMaxTokens: explicit?.defaultMaxTokens ?? 4096,
  };
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
  const needsCustomFetch = codexOpts || (fieldsToStrip && fieldsToStrip.length > 0);
  const customFetch = needsCustomFetch
    ? async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        if (init?.body && typeof init.body === "string") {
          try {
            const body = JSON.parse(init.body) as Record<string, unknown>;
            // ChatGPT Codex: inject required fields
            if (codexOpts) {
              if (codexOpts.store !== undefined) body["store"] = codexOpts.store;
              if (codexOpts.include) body["include"] = [...codexOpts.include];
              delete body["max_output_tokens"];
            }
            // Strip fields the endpoint rejects
            if (fieldsToStrip) {
              for (const field of fieldsToStrip) {
                delete body[field];
              }
            }
            init = { ...init, body: JSON.stringify(body) };
          } catch {
            // Not JSON — pass through
          }
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
      const aiTools = tools ? convertTools(tools) : undefined;

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

        const result = streamText({
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
                // Strip null-valued keys: our convertTools makes optional params
                // nullable (type: ["T", "null"]) + required for strict mode.
                // The model sends null for unused optionals — strip them so
                // downstream tool handlers see undefined (not provided), not null.
                content: JSON.stringify(stripNullArgs(part.args as Record<string, unknown>)),
                toolCallId: part.toolCallId,
                toolName: part.toolName,
              };
              break;

            case "error":
              throw new ProviderError(`OpenAI stream error: ${String(part.error)}`);

            case "finish":
              yield {
                type: "done",
                content: "",
                usage: part.usage
                  ? {
                      promptTokens: part.usage.promptTokens,
                      completionTokens: part.usage.completionTokens,
                    }
                  : undefined,
              };
              break;
          }
        }
      } catch (err) {
        if (abortController.signal.aborted) {
          yield { type: "done", content: "" };
          return;
        }
        if (err instanceof ProviderError) throw err;
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
    // OpenAI strict mode requires all properties in `required` and `additionalProperties: false`.
    // Optional properties must be made nullable (type: ["string", "null"]) instead.
    const rawProps = (t.paramSchema.properties ?? {}) as Record<string, Record<string, unknown>>;
    const requiredSet = new Set(t.paramSchema.required ?? []);
    const allPropertyNames = Object.keys(rawProps);

    // Make non-required properties nullable
    const strictProps: Record<string, Record<string, unknown>> = {};
    for (const [key, schema] of Object.entries(rawProps)) {
      if (!requiredSet.has(key)) {
        // Convert to nullable: type becomes ["originalType", "null"]
        const origType = schema["type"] as string | undefined;
        strictProps[key] = {
          ...schema,
          type: origType ? [origType, "null"] : ["string", "null"],
        };
      } else {
        strictProps[key] = schema;
      }
    }

    result[t.name] = aiTool({
      description: t.description,
      parameters: jsonSchema({
        type: t.paramSchema.type as "object",
        properties: strictProps,
        required: allPropertyNames,
        additionalProperties: false,
      }),
    });
  }

  return result;
}

/**
 * Strip null-valued keys from tool call arguments.
 * Our convertTools transforms optional params into nullable + required for
 * OpenAI strict mode. The model responds with null for unused optionals.
 * Strip them so tool handlers see undefined (absent), not null.
 */
export function stripNullArgs(args: Record<string, unknown>): Record<string, unknown> {
  const cleaned: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(args)) {
    if (v !== null) cleaned[k] = v;
  }
  return cleaned;
}
