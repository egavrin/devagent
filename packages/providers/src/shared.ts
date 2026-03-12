/**
 * Shared utilities for LLM providers.
 *
 * processProviderStream — async generator that maps a Vercel AI SDK fullStream to StreamChunk[].
 * convertMessages  — converts internal Message[] to Vercel AI SDK CoreMessage[].
 * convertTools     — converts ToolSpec[] to Vercel AI SDK tool records (with optional strict mode).
 * resolveCapabilities — fills in safe defaults for ModelCapabilities.
 * stripNullArgs — removes null-valued keys from tool call arguments.
 */

import { tool as aiTool, jsonSchema, type CoreMessage, type TextStreamPart, type ToolSet } from "ai";
import type { Message, ModelCapabilities, StreamChunk, ToolSpec } from "@devagent/runtime";
import { MessageRole, ProviderError, extractErrorMessage } from "@devagent/runtime";

// ─── Stream Processing ───────────────────────────────────────

export interface ProcessStreamOptions {
  /** Provider name used in error messages (e.g., "Anthropic", "OpenAI"). */
  readonly providerName: string;
  /** The fullStream from Vercel AI SDK streamText(). */
  readonly fullStream: AsyncIterable<TextStreamPart<ToolSet>>;
  /** AbortController whose signal was passed to streamText(). */
  readonly abortController: AbortController;
  /**
   * Optional transform applied to tool-call arguments before JSON.stringify.
   * OpenAI uses stripNullArgs here; Anthropic passes no transform.
   */
  readonly transformArgs?: (args: Record<string, unknown>) => Record<string, unknown>;
}

/**
 * Shared async generator that maps a Vercel AI SDK fullStream into StreamChunk[].
 *
 * Handles text-delta, tool-call, error, and finish events uniformly.
 * Provider-specific differences are parameterized:
 * - providerName controls error message prefixes
 * - transformArgs allows post-processing tool-call arguments (e.g., stripNullArgs)
 */
export async function* processProviderStream(
  options: ProcessStreamOptions,
): AsyncGenerator<StreamChunk> {
  const { providerName, fullStream, abortController, transformArgs } = options;

  try {
    for await (const part of fullStream) {
      switch (part.type) {
        case "text-delta":
          yield {
            type: "text",
            content: part.textDelta,
          };
          break;

        case "tool-call": {
          const args = transformArgs
            ? transformArgs(part.args as Record<string, unknown>)
            : part.args;
          yield {
            type: "tool_call",
            content: JSON.stringify(args),
            toolCallId: part.toolCallId,
            toolName: part.toolName,
          };
          break;
        }

        case "error":
          throw new ProviderError(`${providerName} stream error: ${String(part.error)}`);

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
    const msg = extractErrorMessage(err);
    throw new ProviderError(`${providerName} API error: ${msg}`);
  }
}

// ─── Message Conversion ──────────────────────────────────────

/**
 * Convert DevAgent Message[] to Vercel AI SDK CoreMessage[].
 * Shared by all providers that use the Vercel AI SDK.
 */
export function convertMessages(messages: ReadonlyArray<Message>): CoreMessage[] {
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

// ─── Capability Resolution ───────────────────────────────────

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

// ─── Null Argument Stripping ─────────────────────────────────

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

// ─── Tool Conversion ─────────────────────────────────────────

export interface ConvertToolsOptions {
  /**
   * When true, enables OpenAI strict mode:
   * - All properties are marked as required
   * - Non-required properties are made nullable (type: ["T", "null"])
   * - additionalProperties is set to false
   *
   * Pair with stripNullArgs() on the response side to convert null back to
   * absent for downstream tool handlers.
   */
  readonly strict?: boolean;
}

/**
 * Convert ToolSpec[] into a Vercel AI SDK tool record.
 *
 * Base mode (strict: false | omitted):
 *   Passes properties and required arrays through as-is.
 *
 * Strict mode (strict: true):
 *   OpenAI structured outputs require every property in `required` and
 *   `additionalProperties: false`. Optional properties are made nullable
 *   (type: ["originalType", "null"]) so the model can send null for unused
 *   optionals — pair with stripNullArgs() on the response side.
 */
export function convertTools(
  tools: ReadonlyArray<ToolSpec>,
  options?: ConvertToolsOptions,
): Record<string, ReturnType<typeof aiTool>> {
  const strict = options?.strict ?? false;
  const result: Record<string, ReturnType<typeof aiTool>> = {};

  for (const t of tools) {
    const rawProps = (t.paramSchema.properties ?? {}) as Record<string, Record<string, unknown>>;

    if (strict) {
      // OpenAI strict mode: all properties required, non-required become nullable
      const requiredSet = new Set(t.paramSchema.required ?? []);
      const allPropertyNames = Object.keys(rawProps);

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
    } else {
      // Base mode: pass through as-is
      result[t.name] = aiTool({
        description: t.description,
        parameters: jsonSchema({
          type: t.paramSchema.type as "object",
          properties: rawProps,
          required: [...(t.paramSchema.required ?? [])],
        }),
      });
    }
  }

  return result;
}
