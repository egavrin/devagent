/**
 * Shared utilities for LLM providers.
 *
 * processProviderStream — async generator that maps a Vercel AI SDK fullStream to StreamChunk[].
 * convertMessages  — converts internal Message[] to Vercel AI SDK CoreMessage[].
 * convertTools     — converts ToolSpec[] to Vercel AI SDK tool records (with optional strict mode).
 * resolveCapabilities — fills in safe defaults for ModelCapabilities.
 * stripNullArgs — removes null-valued keys from tool call arguments.
 */

import {
  MessageRole,
  ProviderError,
  RateLimitError,
  ProviderConnectionError,
  ProviderTlsCertificateError,
  OverloadedError,
  extractErrorMessage,
} from "@devagent/runtime";
import { tool as aiTool, jsonSchema, type CoreMessage, type TextStreamPart } from "ai";

import type { Message, ModelCapabilities, StreamChunk, ToolSpec } from "@devagent/runtime";

// ─── Stream Processing ───────────────────────────────────────

interface ProcessStreamOptions {
  /** Provider name used in error messages (e.g., "Anthropic", "OpenAI"). */
  readonly providerName: string;
  /** The fullStream from Vercel AI SDK streamText(). */
  readonly fullStream: AsyncIterable<TextStreamPart<Record<string, never>>>;
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
      const chunk = mapStreamPart(part, providerName, transformArgs);
      if (chunk) {
        yield chunk;
      }
    }
  } catch (err) {
    if (abortController.signal.aborted) {
      yield { type: "done", content: "" };
      return;
    }
    if (err instanceof ProviderError) throw err;
    throw classifyProviderError(err, providerName);
  }
}

function mapStreamPart(
  part: TextStreamPart<Record<string, never>>,
  providerName: string,
  transformArgs: ProcessStreamOptions["transformArgs"],
): StreamChunk | null {
  if (part.type === "text-delta") {
    return { type: "text", content: part.text };
  }
  if (part.type === "tool-call") {
    return {
      type: "tool_call",
      content: JSON.stringify(mapToolInput(part.input, transformArgs)),
      toolCallId: part.toolCallId,
      toolName: part.toolName,
    };
  }
  if (part.type === "reasoning-delta") {
    return { type: "thinking", content: typeof part.text === "string" ? part.text : "" };
  }
  if (part.type === "error") {
    throw new ProviderError(`${providerName} stream error: ${String(part.error)}`);
  }
  if (part.type === "finish") {
    return {
      type: "done",
      content: "",
      usage: part.totalUsage
        ? {
            promptTokens: part.totalUsage.inputTokens ?? 0,
            completionTokens: part.totalUsage.outputTokens ?? 0,
          }
        : undefined,
    };
  }
  return null;
}

function mapToolInput(
  input: unknown,
  transformArgs: ProcessStreamOptions["transformArgs"],
): unknown {
  if (!transformArgs) return input;
  return transformArgs(input as Record<string, unknown>);
}

/**
 * Classify provider errors by HTTP status code.
 * Extracts status from Vercel AI SDK error shapes and wraps in typed errors.
 */
export function classifyProviderError(err: unknown, providerName: string): ProviderError {
  if (err instanceof ProviderError) return err;

  const msg = extractErrorMessage(err);
  const status = extractHttpStatus(err);
  const certificateDetail = extractCertificateVerificationDetail(err);

  if (status === 401 || status === 403) {
    return new ProviderError(`${providerName} authentication failed (${status}): ${msg}. Check your provider, model, and credentials with 'devagent doctor'.`);
  }
  if (status === 429) {
    const retryAfter = extractRetryAfter(err);
    return new RateLimitError(`${providerName} rate limited: ${msg}`, retryAfter);
  }
  if (status === 529) {
    return new OverloadedError(`${providerName} overloaded (529): ${msg}`);
  }
  if (certificateDetail) {
    return new ProviderTlsCertificateError(providerName, certificateDetail);
  }
  if (isConnectionError(msg)) {
    return new ProviderConnectionError(`${providerName} connection error: ${msg}`);
  }

  // Catch non-iterable errors from AI SDK response parsing (e.g. invalid API key returning HTML/JSON error)
  if (msg.includes("is not iterable") || msg.includes("not a function")) {
    return new ProviderError(`${providerName} returned an unexpected response. Check your API key and model with 'devagent doctor'.`);
  }

  return new ProviderError(`${providerName} API error: ${msg}`);
}
function extractHttpStatus(err: unknown): number | null {
  if (!err || typeof err !== "object") return null;
  const e = err as Record<string, unknown>;
  const direct = firstNumber(e["statusCode"], e["status"]);
  if (direct !== null) return direct;
  const nested = firstNestedStatus(e["cause"], e["data"]);
  if (nested !== null) return nested;
  const msgStr = typeof e["message"] === "string" ? e["message"] : "";
  const match = msgStr.match(/\b(429|529|502|503|504)\b/);
  return match ? parseInt(match[1]!, 10) : null;
}

function firstNumber(...values: unknown[]): number | null {
  const found = values.find((value): value is number => typeof value === "number");
  return found ?? null;
}

function firstNestedStatus(...values: unknown[]): number | null {
  for (const value of values) {
    if (value && typeof value === "object") {
      const status = extractHttpStatus(value);
      if (status !== null) return status;
    }
  }
  return null;
}

function extractRetryAfter(err: unknown): number | null {
  if (!err || typeof err !== "object") return null;
  const e = err as Record<string, unknown>;
  return firstRetryAfter(e["headers"], e["responseHeaders"]);
}

function firstRetryAfter(...headerValues: unknown[]): number | null {
  for (const value of headerValues) {
    const retryAfter = readRetryAfter(value);
    if (retryAfter !== null) return retryAfter;
  }
  return null;
}

function readRetryAfter(value: unknown): number | null {
  if (!value || typeof value !== "object") return null;
  const headers = value as Record<string, string>;
  const retryAfter = headers["retry-after"] ?? headers["Retry-After"];
  if (!retryAfter) return null;
  const seconds = parseFloat(retryAfter);
  return Number.isNaN(seconds) ? null : seconds * 1000;
}

const CONNECTION_ERROR_PATTERNS = [
  "econnreset", "econnrefused", "epipe", "etimedout",
  "socket hang up", "network error", "fetch failed",
  "connection reset", "connection refused",
];

const CERTIFICATE_MESSAGE_PATTERNS = [
  "self-signed certificate",
  "self signed certificate",
  "unable to verify the first certificate",
  "unable to verify leaf signature",
  "unable to get local issuer certificate",
  "unable to get issuer certificate",
  "certificate verify failed",
  "certificate verification failed",
  "unknown certificate verification error",
  "unable to verify",
];

const CERTIFICATE_ERROR_CODES = new Set([
  "CERT_HAS_EXPIRED",
  "CERT_NOT_YET_VALID",
  "CERT_REJECTED",
  "CERT_SIGNATURE_FAILURE",
  "CERT_UNTRUSTED",
  "DEPTH_ZERO_SELF_SIGNED_CERT",
  "ERR_TLS_CERT_ALTNAME_INVALID",
  "ERR_TLS_CERT_SIGNATURE_ALGORITHM_UNSUPPORTED",
  "SELF_SIGNED_CERT_IN_CHAIN",
  "UNABLE_TO_GET_ISSUER_CERT",
  "UNABLE_TO_GET_ISSUER_CERT_LOCALLY",
  "UNABLE_TO_VERIFY_LEAF_SIGNATURE",
]);

function isConnectionError(message: string): boolean {
  const lower = message.toLowerCase();
  return CONNECTION_ERROR_PATTERNS.some((p) => lower.includes(p));
}

function extractCertificateVerificationDetail(err: unknown): string | null {
  const messages = collectErrorStrings(err, new Set<unknown>(), "message");
  for (const message of messages) {
    const lower = message.toLowerCase();
    if (CERTIFICATE_MESSAGE_PATTERNS.some((pattern) => lower.includes(pattern))) {
      return message;
    }
  }

  const codes = collectErrorStrings(err, new Set<unknown>(), "code");
  for (const code of codes) {
    const upper = code.toUpperCase();
    if (CERTIFICATE_ERROR_CODES.has(upper)) {
      return code;
    }
  }

  return null;
}

function collectErrorStrings(
  value: unknown,
  seen: Set<unknown>,
  key: "message" | "code",
): string[] {
  if (!value || (typeof value !== "object" && typeof value !== "function")) {
    return [];
  }
  if (seen.has(value)) {
    return [];
  }
  seen.add(value);

  const collected: string[] = [];
  const record = value as Record<string, unknown>;
  const direct = record[key];
  if (typeof direct === "string" && direct.trim().length > 0) {
    collected.push(direct.trim());
  }

  if (record["cause"] !== undefined) {
    collected.push(...collectErrorStrings(record["cause"], seen, key));
  }
  if (record["data"] !== undefined) {
    collected.push(...collectErrorStrings(record["data"], seen, key));
  }

  return collected;
}

// ─── Message Conversion ──────────────────────────────────────

/**
 * Convert DevAgent Message[] to Vercel AI SDK CoreMessage[].
 * Shared by all providers that use the Vercel AI SDK.
 */
export function convertMessages(messages: ReadonlyArray<Message>): CoreMessage[] {
  return messages.map(convertMessage);
}

type AssistantPart =
  | { type: "text"; text: string }
  | { type: "reasoning"; text: string }
  | { type: "tool-call"; toolCallId: string; toolName: string; input: Record<string, unknown> };

function convertMessage(msg: Message): CoreMessage {
  if (msg.role === MessageRole.SYSTEM) return { role: "system", content: msg.content ?? "" };
  if (msg.role === MessageRole.USER) return { role: "user", content: msg.content ?? "" };
  if (msg.role === MessageRole.TOOL) return convertToolMessage(msg);
  return convertAssistantMessage(msg);
}

function convertAssistantMessage(msg: Message): CoreMessage {
  const hasToolCalls = Boolean(msg.toolCalls?.length);
  if (!hasToolCalls && !msg.thinking) {
    return { role: "assistant", content: msg.content ?? "" };
  }
  return { role: "assistant", content: buildAssistantParts(msg) };
}

function buildAssistantParts(msg: Message): AssistantPart[] {
  const parts: AssistantPart[] = [];
  if (msg.thinking) parts.push({ type: "reasoning", text: msg.thinking });
  if (msg.content) parts.push({ type: "text", text: msg.content });
  for (const tc of msg.toolCalls ?? []) {
    parts.push({
      type: "tool-call",
      toolCallId: tc.callId,
      toolName: tc.name,
      input: tc.arguments,
    });
  }
  return parts;
}

function convertToolMessage(msg: Message): CoreMessage {
  return {
    role: "tool",
    content: [
      {
        type: "tool-result",
        toolCallId: msg.toolCallId ?? "",
        toolName: "",
        output: { type: "text", value: msg.content ?? "" },
      },
    ],
  };
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

interface ConvertToolsOptions {
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
    result[t.name] = convertTool(t, strict);
  }

  return result;
}

function convertTool(t: ToolSpec, strict: boolean): ReturnType<typeof aiTool> {
  const rawProps = (t.paramSchema.properties ?? {}) as Record<string, Record<string, unknown>>;
  const schema = strict ? buildStrictToolSchema(t, rawProps) : buildBaseToolSchema(t, rawProps);
  return aiTool({ description: t.description, inputSchema: jsonSchema(schema) });
}

function buildBaseToolSchema(
  t: ToolSpec,
  rawProps: Record<string, Record<string, unknown>>,
): Parameters<typeof jsonSchema>[0] {
  return {
    type: t.paramSchema.type as "object",
    properties: rawProps,
    required: Array.isArray(t.paramSchema.required) ? [...t.paramSchema.required] : [],
  };
}

function buildStrictToolSchema(
  t: ToolSpec,
  rawProps: Record<string, Record<string, unknown>>,
): Parameters<typeof jsonSchema>[0] {
  const requiredSet = new Set(Array.isArray(t.paramSchema.required) ? t.paramSchema.required : []);
  return {
    type: t.paramSchema.type as "object",
    properties: buildStrictProperties(rawProps, requiredSet),
    required: Object.keys(rawProps),
    additionalProperties: false,
  };
}

function buildStrictProperties(
  rawProps: Record<string, Record<string, unknown>>,
  requiredSet: ReadonlySet<string>,
): Record<string, Record<string, unknown>> {
  return Object.fromEntries(
    Object.entries(rawProps).map(([key, schema]) => [
      key,
      requiredSet.has(key) ? schema : nullableSchema(schema),
    ]),
  );
}

function nullableSchema(schema: Record<string, unknown>): Record<string, unknown> {
  const origType = schema["type"] as string | undefined;
  return {
    ...schema,
    type: origType ? [origType, "null"] : ["string", "null"],
  };
}
