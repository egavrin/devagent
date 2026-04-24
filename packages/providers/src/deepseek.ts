import {
  MessageRole,
  ProviderError,
  createProxyAwareFetch,
} from "@devagent/runtime";

import {
  classifyProviderError,
  resolveCapabilities,
} from "./shared.js";
import type { LLMProvider, Message, ProviderConfig, StreamChunk, ToolSpec } from "@devagent/runtime";

interface DeepSeekToolCallDelta {
  readonly index: number;
  readonly id?: string | null;
  readonly type?: string | null;
  readonly function?: {
    readonly name?: string | null;
    readonly arguments?: string | null;
  } | null;
}

interface PendingDeepSeekToolCall {
  id: string;
  name: string;
  arguments: string;
}

interface DeepSeekUsage {
  readonly prompt_tokens?: number | null;
  readonly completion_tokens?: number | null;
}

interface PreparedDeepSeekHistory {
  readonly messages: Array<Record<string, unknown>>;
  readonly validToolCallIds: ReadonlySet<string>;
}

interface NormalizedDeepSeekTurn {
  readonly messages: Array<Record<string, unknown>>;
  readonly validToolCallIds: ReadonlySet<string>;
}

/**
 * DeepSeek's thinking-mode Chat Completions protocol has one non-OpenAI quirk:
 * assistant tool-call messages must be replayed with reasoning_content.
 */
export function createDeepSeekProvider(config: ProviderConfig): LLMProvider {
  const baseUrl = config.baseUrl ?? "https://api.deepseek.com/v1";
  const transportFetch = createProxyAwareFetch(globalThis.fetch);
  let abortController: AbortController | null = null;

  return {
    id: "deepseek",
    async *chat(
      messages: ReadonlyArray<Message>,
      tools?: ReadonlyArray<ToolSpec>,
    ): AsyncIterable<StreamChunk> {
      abortController = new AbortController();
      try {
        const response = await transportFetch(`${baseUrl}/chat/completions`, {
          method: "POST",
          headers: buildHeaders(config),
          body: JSON.stringify(buildRequestBody(config, messages, tools)),
          signal: abortController.signal,
        });

        if (!response.ok) {
          throw await buildDeepSeekError(response);
        }

        yield* streamDeepSeekResponse(response, abortController);
      } catch (err) {
        if (abortController.signal.aborted) {
          yield { type: "done", content: "" };
          return;
        }
        if (err instanceof ProviderError) throw err;
        throw classifyProviderError(err, "DeepSeek");
      } finally {
        abortController = null;
      }
    },
    abort(): void {
      abortController?.abort();
    },
  };
}

function buildHeaders(config: ProviderConfig): Record<string, string> {
  return {
    "content-type": "application/json",
    ...(config.apiKey ? { authorization: `Bearer ${config.apiKey}` } : {}),
    ...(config.customHeaders ?? {}),
  };
}

function buildRequestBody(
  config: ProviderConfig,
  messages: ReadonlyArray<Message>,
  tools: ReadonlyArray<ToolSpec> | undefined,
): Record<string, unknown> {
  const caps = resolveCapabilities(config.capabilities);
  const body: Record<string, unknown> = {
    model: config.model,
    messages: prepareDeepSeekHistory(messages),
    stream: true,
    stream_options: { include_usage: true },
    max_tokens: config.maxTokens ?? caps.defaultMaxTokens,
  };

  if (config.reasoningEffort) {
    body["reasoning_effort"] = config.reasoningEffort === "xhigh" ? "max" : config.reasoningEffort;
  }
  if (caps.reasoning) {
    body["thinking"] = { type: "enabled" };
  }
  if (caps.supportsTemperature) {
    body["temperature"] = config.temperature ?? 0;
  }
  if (tools && tools.length > 0) {
    body["tools"] = tools.map(convertDeepSeekTool);
    body["tool_choice"] = "auto";
  }

  return body;
}

function prepareDeepSeekHistory(messages: ReadonlyArray<Message>): Array<Record<string, unknown>> {
  const prepared = normalizeDeepSeekHistory(messages);
  validateDeepSeekHistory(prepared);
  return prepared.messages;
}

function normalizeDeepSeekHistory(messages: ReadonlyArray<Message>): PreparedDeepSeekHistory {
  const validToolCallIds = new Set<string>();
  const converted: Array<Record<string, unknown>> = [];
  let turn: Message[] = [];

  for (const message of messages) {
    if (message.role === MessageRole.SYSTEM) {
      flushDeepSeekTurn(turn, converted, validToolCallIds);
      turn = [];
      converted.push({ role: "system", content: message.content ?? "" });
      continue;
    }
    if (message.role === MessageRole.USER) {
      flushDeepSeekTurn(turn, converted, validToolCallIds);
      turn = [];
      converted.push({ role: "user", content: message.content ?? "" });
      continue;
    }
    turn.push(message);
  }
  flushDeepSeekTurn(turn, converted, validToolCallIds);

  return { messages: converted, validToolCallIds };
}

function flushDeepSeekTurn(
  turn: ReadonlyArray<Message>,
  converted: Array<Record<string, unknown>>,
  validToolCallIds: Set<string>,
): void {
  if (turn.length === 0) return;
  const normalized = normalizeDeepSeekTurn(turn);
  for (const id of normalized.validToolCallIds) validToolCallIds.add(id);
  converted.push(...normalized.messages);
}

function normalizeDeepSeekTurn(turn: ReadonlyArray<Message>): NormalizedDeepSeekTurn {
  const droppedToolCallIds = collectUnreplayableDeepSeekToolCallIds(turn);
  const validToolCallIds = collectReplayableDeepSeekToolCallIds(turn);
  const hasToolUse = validToolCallIds.size > 0;
  const messages: Array<Record<string, unknown>> = [];

  for (const message of turn) {
    if (shouldDropDeepSeekToolResult(message, droppedToolCallIds, validToolCallIds)) continue;
    const normalized = convertDeepSeekTurnMessage(message, droppedToolCallIds, hasToolUse);
    if (normalized) messages.push(normalized);
  }

  return { messages, validToolCallIds };
}

function collectUnreplayableDeepSeekToolCallIds(messages: ReadonlyArray<Message>): Set<string> {
  const dropped = new Set<string>();
  for (const message of messages) {
    if (message.role !== MessageRole.ASSISTANT || message.thinking || !message.toolCalls?.length) continue;
    for (const toolCall of message.toolCalls) dropped.add(toolCall.callId);
  }
  return dropped;
}

function collectReplayableDeepSeekToolCallIds(messages: ReadonlyArray<Message>): Set<string> {
  const valid = new Set<string>();
  for (const message of messages) {
    if (message.role !== MessageRole.ASSISTANT || !message.thinking || !message.toolCalls?.length) continue;
    for (const toolCall of message.toolCalls) valid.add(toolCall.callId);
  }
  return valid;
}

function shouldDropDeepSeekToolResult(
  message: Message,
  droppedToolCallIds: ReadonlySet<string>,
  validToolCallIds: ReadonlySet<string>,
): boolean {
  if (message.role !== MessageRole.TOOL) return false;
  if (!message.toolCallId) return true;
  return droppedToolCallIds.has(message.toolCallId) || !validToolCallIds.has(message.toolCallId);
}

function convertDeepSeekTurnMessage(
  message: Message,
  droppedToolCallIds: ReadonlySet<string>,
  hasToolUse: boolean,
): Record<string, unknown> | null {
  if (message.role === MessageRole.TOOL) return convertDeepSeekToolResultMessage(message);
  return convertDeepSeekAssistantTurnMessage(message, droppedToolCallIds, hasToolUse);
}

function convertDeepSeekToolResultMessage(message: Message): Record<string, unknown> {
  return {
    role: "tool",
    tool_call_id: message.toolCallId ?? "",
    content: message.content ?? "",
  };
}

function convertDeepSeekAssistantTurnMessage(
  message: Message,
  droppedToolCallIds: ReadonlySet<string>,
  hasToolUse: boolean,
): Record<string, unknown> | null {
  const converted: Record<string, unknown> = {
    role: "assistant",
    content: message.content ?? "",
  };

  if (!message.toolCalls?.length) return convertDeepSeekAssistantNoToolCallMessage(message, converted, hasToolUse);
  if (!message.thinking) {
    return convertDeepSeekAssistantNoThinkingMessage(message, converted, hasToolUse);
  }

  const toolCalls = message.toolCalls.filter((toolCall) => !droppedToolCallIds.has(toolCall.callId));
  if (toolCalls.length === 0) return convertDeepSeekAssistantWithoutToolCalls(message, converted);

  converted["reasoning_content"] = message.thinking;
  converted["tool_calls"] = toolCalls.map(convertDeepSeekToolCall);
  return converted;
}

function convertDeepSeekAssistantNoToolCallMessage(
  message: Message,
  converted: Record<string, unknown>,
  hasToolUse: boolean,
): Record<string, unknown> | null {
  if (message.thinking && hasToolUse) converted["reasoning_content"] = message.thinking;
  return hasToolUse && !message.thinking ? null : converted;
}

function convertDeepSeekAssistantNoThinkingMessage(
  message: Message,
  converted: Record<string, unknown>,
  hasToolUse: boolean,
): Record<string, unknown> | null {
  if (hasToolUse) return null;
  return convertDeepSeekAssistantWithoutToolCalls(message, converted);
}

function convertDeepSeekAssistantWithoutToolCalls(
  message: Message,
  converted: Record<string, unknown>,
): Record<string, unknown> | null {
  return message.content?.trim() ? converted : null;
}

function convertDeepSeekToolCall(toolCall: NonNullable<Message["toolCalls"]>[number]): Record<string, unknown> {
  return {
    id: toolCall.callId,
    type: "function",
    function: {
      name: toolCall.name,
      arguments: JSON.stringify(toolCall.arguments),
    },
  };
}

function validateDeepSeekHistory(prepared: PreparedDeepSeekHistory): void {
  for (const turn of splitDeepSeekHistoryTurns(prepared.messages)) {
    const hasToolUse = turn.some((message) => readDeepSeekToolCalls(message).length > 0);
    for (const message of turn) {
      validateDeepSeekAssistantMessage(message, hasToolUse);
      validateDeepSeekToolMessage(message, prepared.validToolCallIds);
    }
  }
}

function splitDeepSeekHistoryTurns(messages: ReadonlyArray<Record<string, unknown>>): Array<Array<Record<string, unknown>>> {
  const turns: Array<Array<Record<string, unknown>>> = [];
  let current: Array<Record<string, unknown>> = [];
  for (const message of messages) {
    if (message["role"] === "user" || message["role"] === "system") {
      if (current.length > 0) turns.push(current);
      current = [message];
    } else {
      current.push(message);
    }
  }
  if (current.length > 0) turns.push(current);
  return turns;
}

function validateDeepSeekAssistantMessage(message: Record<string, unknown>, hasToolUse: boolean): void {
  if (message["role"] !== "assistant") return;
  const toolCalls = readDeepSeekToolCalls(message);
  for (const toolCall of toolCalls) {
    if (typeof toolCall.id !== "string" || toolCall.id.length === 0) {
      throw new ProviderError("DeepSeek history error: assistant tool calls require non-empty ids");
    }
  }
  if (toolCalls.length > 0 && typeof message["reasoning_content"] !== "string") {
    throw new ProviderError("DeepSeek history error: assistant tool calls require reasoning_content");
  }
  if (toolCalls.length === 0 && "reasoning_content" in message && !hasToolUse) {
    throw new ProviderError("DeepSeek history error: final assistant messages must not include reasoning_content");
  }
}

function validateDeepSeekToolMessage(
  message: Record<string, unknown>,
  validToolCallIds: ReadonlySet<string>,
): void {
  if (message["role"] !== "tool") return;
  const toolCallId = message["tool_call_id"];
  if (typeof toolCallId !== "string" || !validToolCallIds.has(toolCallId)) {
    throw new ProviderError("DeepSeek history error: tool result does not match an assistant tool call");
  }
}

function readDeepSeekToolCalls(message: Record<string, unknown>): Array<{ id?: unknown }> {
  const toolCalls = message["tool_calls"];
  return Array.isArray(toolCalls) ? toolCalls as Array<{ id?: unknown }> : [];
}

function convertDeepSeekTool(tool: ToolSpec): Record<string, unknown> {
  return {
    type: "function",
    function: {
      name: tool.name,
      description: tool.description,
      parameters: tool.paramSchema,
    },
  };
}

async function buildDeepSeekError(response: Response): Promise<ProviderError> {
  const responseBody = await response.text();
  const message = extractDeepSeekErrorMessage(responseBody) ?? responseBody;
  const error = new Error(message) as Error & {
    status: number;
    statusCode: number;
    responseHeaders: Record<string, string>;
  };
  error.status = response.status;
  error.statusCode = response.status;
  error.responseHeaders = Object.fromEntries(response.headers.entries());
  return classifyProviderError(error, "DeepSeek");
}

function extractDeepSeekErrorMessage(responseBody: string): string | null {
  try {
    const parsed = JSON.parse(responseBody) as { error?: { message?: unknown } };
    return typeof parsed.error?.message === "string" ? parsed.error.message : null;
  } catch {
    return null;
  }
}

async function* streamDeepSeekResponse(
  response: Response,
  abortController: AbortController,
): AsyncGenerator<StreamChunk> {
  if (!response.body) {
    throw new ProviderError("DeepSeek stream error: missing response body");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const toolCalls = new Map<number, PendingDeepSeekToolCall>();
  const streamState = { doneSent: false };
  let buffer = "";

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const events = splitCompleteSseEvents(buffer);
      buffer = events.remainder;
      for (const event of events.complete) {
        const finished = yield* emitDeepSeekSseEvent(event, toolCalls, streamState);
        if (finished) return;
      }
    }

    const tail = parseSseData(buffer);
    if (tail && tail !== "[DONE]") {
      yield* emitDeepSeekChunk(tail, toolCalls, streamState);
    }
    yield* finishDeepSeekStream(toolCalls, streamState);
  } catch (err) {
    if (abortController.signal.aborted) {
      yield { type: "done", content: "" };
      return;
    }
    throw err;
  } finally {
    reader.releaseLock();
  }
}

function* emitDeepSeekSseEvent(
  event: string,
  toolCalls: Map<number, PendingDeepSeekToolCall>,
  streamState: { doneSent: boolean },
): Generator<StreamChunk, boolean> {
  const chunk = parseSseData(event);
  if (!chunk) return false;
  if (chunk === "[DONE]") {
    yield* finishDeepSeekStream(toolCalls, streamState);
    return true;
  }
  yield* emitDeepSeekChunk(chunk, toolCalls, streamState);
  return false;
}

function* emitDeepSeekChunk(
  raw: string,
  toolCalls: Map<number, PendingDeepSeekToolCall>,
  streamState: { doneSent: boolean },
): Generator<StreamChunk> {
  const emitted = mapDeepSeekChunk(raw, toolCalls);
  for (const item of emitted) {
    if (item.type === "done") streamState.doneSent = true;
    yield item;
  }
}

function* finishDeepSeekStream(
  toolCalls: Map<number, PendingDeepSeekToolCall>,
  streamState: { doneSent: boolean },
): Generator<StreamChunk> {
  yield* flushDeepSeekToolCalls(toolCalls);
  if (streamState.doneSent) return;
  streamState.doneSent = true;
  yield { type: "done", content: "" };
}

function splitCompleteSseEvents(buffer: string): { readonly complete: string[]; readonly remainder: string } {
  const normalized = buffer.replaceAll("\r\n", "\n");
  const parts = normalized.split("\n\n");
  return {
    complete: parts.slice(0, -1),
    remainder: parts.at(-1) ?? "",
  };
}

function parseSseData(event: string): string | null {
  const data = event
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice("data:".length).trimStart())
    .join("\n")
    .trim();
  return data.length > 0 ? data : null;
}

function mapDeepSeekChunk(
  raw: string,
  toolCalls: Map<number, PendingDeepSeekToolCall>,
): StreamChunk[] {
  const parsed = parseDeepSeekChunk(raw);
  const chunks = (parsed.choices ?? []).flatMap((choice) => mapDeepSeekChoice(choice, toolCalls));
  if (parsed.usage) chunks.push(mapDeepSeekUsage(parsed.usage));
  return chunks;
}

function parseDeepSeekChunk(raw: string): {
  readonly choices?: Array<{
    readonly delta?: {
      readonly content?: string | null;
      readonly reasoning_content?: string | null;
      readonly tool_calls?: DeepSeekToolCallDelta[] | null;
    } | null;
    readonly finish_reason?: string | null;
  }>;
  readonly usage?: DeepSeekUsage | null;
} {
  return JSON.parse(raw) as {
    choices?: Array<{
      delta?: {
        content?: string | null;
        reasoning_content?: string | null;
        tool_calls?: DeepSeekToolCallDelta[] | null;
      } | null;
      finish_reason?: string | null;
    }>;
    usage?: DeepSeekUsage | null;
  };
}

function mapDeepSeekChoice(
  choice: NonNullable<ReturnType<typeof parseDeepSeekChunk>["choices"]>[number],
  toolCalls: Map<number, PendingDeepSeekToolCall>,
): StreamChunk[] {
  const chunks: StreamChunk[] = [];
  const delta = choice.delta;
  if (delta?.reasoning_content) {
    chunks.push({ type: "thinking", content: delta.reasoning_content });
  }
  if (delta?.content) {
    chunks.push({ type: "text", content: delta.content });
  }
  for (const toolCall of delta?.tool_calls ?? []) {
    mergeToolCallDelta(toolCalls, toolCall);
  }
  if (choice.finish_reason === "tool_calls") chunks.push(...flushDeepSeekToolCalls(toolCalls));
  return chunks;
}

function mapDeepSeekUsage(usage: DeepSeekUsage): StreamChunk {
  return {
    type: "done",
    content: "",
    usage: {
      promptTokens: usage.prompt_tokens ?? 0,
      completionTokens: usage.completion_tokens ?? 0,
    },
  };
}

function mergeToolCallDelta(
  toolCalls: Map<number, PendingDeepSeekToolCall>,
  delta: DeepSeekToolCallDelta,
): void {
  const existing = toolCalls.get(delta.index) ?? {
    id: delta.id ?? `call_${delta.index}`,
    name: "",
    arguments: "",
  };
  if (delta.id) existing.id = delta.id;
  if (delta.function?.name) existing.name += delta.function.name;
  if (delta.function?.arguments) existing.arguments += delta.function.arguments;
  toolCalls.set(delta.index, existing);
}

function flushDeepSeekToolCalls(
  toolCalls: Map<number, PendingDeepSeekToolCall>,
): StreamChunk[] {
  const chunks = Array.from(toolCalls.entries())
    .sort(([a], [b]) => a - b)
    .map(([, toolCall]) => ({
      type: "tool_call" as const,
      content: toolCall.arguments || "{}",
      toolCallId: toolCall.id,
      toolName: toolCall.name,
    }));
  toolCalls.clear();
  return chunks;
}
