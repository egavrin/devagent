/**
 * fetch_url — Fetch content from an explicit http/https URL.
 * Category: external.
 */

import { NodeHtmlMarkdown } from "node-html-markdown";
import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { buildSavedFilename, sanitizePathSegment } from "./fetch-url-filename.js";
import { ToolError, extractErrorMessage } from "../../core/errors.js";
import { createProxyAwareFetch as createCoreProxyAwareFetch } from "../../core/proxy-fetch.js";
import type { FetchFn, ProxyAwareFetchOptions } from "../../core/proxy-fetch.js";
import type { ToolResult, ToolSpec } from "../../core/types.js";

const TOOL_NAME = "fetch_url";
const DEFAULT_TIMEOUT_MS = 30_000;
const MAX_TIMEOUT_MS = 120_000;
const MAX_RESPONSE_BYTES = 5 * 1024 * 1024;
const MAX_OUTPUT_CHARS = 200_000;
const DOWNLOAD_ROOT_DIR = join(tmpdir(), "devagent-fetch-url");
const TEXTUAL_MIME_TYPES = new Set([
  "application/json",
  "application/ld+json",
  "application/xml",
  "application/xhtml+xml",
  "application/javascript",
  "application/ecmascript",
  "application/sql",
  "application/x-sh",
  "application/x-yaml",
]);
const HTML_TO_MARKDOWN = new NodeHtmlMarkdown({
  textReplace: [
    [/\u00a0/g, " "],
  ],
  ignore: ["script", "style", "noscript", "iframe", "svg", "canvas"],
});

interface FetchUrlRequest {
  readonly url: string;
  readonly saveBinary: boolean;
  readonly timeoutMs: number;
}

export const fetchUrlTool: ToolSpec = {
  name: TOOL_NAME,
  description:
    "Fetch the contents of a specific http:// or https:// URL. By default it returns normalized inline text; " +
    "set save_binary=true to save binary or attachment responses to disk and return the saved file path. " +
    "Use for a user-provided link you need to read or analyze. This fetches the exact URL only; it does not search the web.",
  category: "external",
  paramSchema: {
    type: "object",
    properties: {
      url: { type: "string", description: "The explicit http:// or https:// URL to fetch" },
      timeout_ms: {
        type: "number",
        description: "Request timeout in milliseconds (default: 30000, max: 120000)",
        maximum: MAX_TIMEOUT_MS,
      },
      save_binary: {
        type: "boolean",
        description: "When true, save binary or attachment responses to disk instead of rejecting them",
      },
    },
    required: ["url"],
  },
  resultSchema: {
    type: "object",
    properties: {
      content: { type: "string", description: "Normalized fetched content or a saved download summary" },
      final_url: { type: "string", description: "Final URL after redirects" },
      content_type: { type: "string", description: "Response content type" },
      status: { type: "number", description: "HTTP status code" },
      downloaded: { type: "boolean", description: "Whether the response was saved to disk" },
      saved_path: { type: "string", description: "Absolute path for a saved binary or attachment response" },
      saved_size_bytes: { type: "number", description: "Saved response size in bytes" },
      sha256: { type: "string", description: "SHA-256 hash of a saved binary or attachment response" },
    },
  },
  errorGuidance: {
    common:
      "Provide a direct http:// or https:// URL. This tool fetches only the exact URL content. " +
      "Text responses are returned inline by default; binary downloads require save_binary=true.",
    patterns: [
      { match: "URL must use", hint: "Pass an explicit http:// or https:// URL." },
      { match: "Unsupported content type", hint: "The target returned non-text content. Retry with save_binary=true if you want the file saved for later processing." },
      { match: "attachment download", hint: "Attachment-style responses require save_binary=true so the file can be saved instead of inlined." },
      { match: "Response too large", hint: "The page is larger than the fetch_url cap. Try a more specific page instead of a large download endpoint." },
      { match: "Request timed out", hint: "The site did not respond within the timeout. Retry with a larger timeout_ms or use a smaller/faster URL." },
    ],

  },
  handler: async (params, context) => {
    const request = parseFetchUrlRequest(params);
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(new Error("timeout")), request.timeoutMs);

    try {
      const response = await fetchResponse(request, controller);
      return await buildFetchUrlResult(response, request, context.sessionId, controller);
    } catch (error) {
      if (error instanceof ToolError) {
        throw error;
      }
      if (controller.signal.aborted && isTimeoutAbort(controller.signal.reason)) {
        throw new ToolError(TOOL_NAME, `Request timed out after ${request.timeoutMs}ms`);
      }
      throw new ToolError(TOOL_NAME, extractErrorMessage(error));
    } finally {
      clearTimeout(timeout);
    }
  },
};

function parseFetchUrlRequest(params: Record<string, unknown>): FetchUrlRequest {
  const url = String(params["url"] ?? "");
  assertHttpUrl(url);
  return {
    url,
    saveBinary: params["save_binary"] === true,
    timeoutMs: Math.min(Number(params["timeout_ms"] ?? DEFAULT_TIMEOUT_MS), MAX_TIMEOUT_MS),
  };
}

async function fetchResponse(
  request: FetchUrlRequest,
  controller: AbortController,
): Promise<Response> {
  const response = await createProxyAwareFetch()(request.url, {
    method: "GET",
    redirect: "follow",
    signal: controller.signal,
    headers: {
      Accept: "text/html, text/plain, application/json, application/xml, text/xml;q=0.9, */*;q=0.1",
      "Accept-Language": "en-US,en;q=0.8",
      "User-Agent": "DevAgent/0.2 (+https://github.com/egavrin/devagent)",
    },
  });

  if (!response.ok) {
    throw new ToolError(TOOL_NAME, `Request failed with status ${response.status}`);
  }
  return response;
}

async function buildFetchUrlResult(
  response: Response,
  request: FetchUrlRequest,
  sessionId: string,
  controller: AbortController,
): Promise<ToolResult> {
  const responseInfo = await readFetchResponseInfo(response, request.url, controller);
  if (responseInfo.isAttachment && !request.saveBinary) {
    throw new ToolError(TOOL_NAME, "Unsupported content type: attachment download");
  }
  if (!responseInfo.isTextResponse && !request.saveBinary) {
    throw new ToolError(TOOL_NAME, `Unsupported content type: ${responseInfo.mime || "unknown"}`);
  }
  return request.saveBinary && (responseInfo.isAttachment || !responseInfo.isTextResponse)
    ? await buildBinaryFetchResult(responseInfo, request.url, sessionId, response.status)
    : buildTextFetchResult(responseInfo, request.url, response.status);
}

interface FetchResponseInfo {
  readonly finalUrl: string;
  readonly contentType: string;
  readonly disposition: string;
  readonly mime: string;
  readonly isAttachment: boolean;
  readonly isTextResponse: boolean;
  readonly body: Uint8Array;
}

async function readFetchResponseInfo(
  response: Response,
  requestedUrl: string,
  controller: AbortController,
): Promise<FetchResponseInfo> {
  const rawContentType = response.headers.get("content-type") ?? "";
  assertDeclaredResponseSize(response.headers.get("content-length"));

  const body = await readResponseBody(response, controller, MAX_RESPONSE_BYTES);
  const mime = extractMimeType(rawContentType);
  const disposition = response.headers.get("content-disposition") ?? "";

  return {
    finalUrl: response.url || requestedUrl,
    contentType: rawContentType || "application/octet-stream",
    disposition,
    mime,
    isAttachment: disposition.toLowerCase().includes("attachment"),
    isTextResponse: isConfidentlyTextualResponse(mime, body),
    body,
  };
}

function assertDeclaredResponseSize(contentLength: string | null): void {
  if (!contentLength) return;
  const declaredSize = Number(contentLength);
  if (Number.isFinite(declaredSize) && declaredSize > MAX_RESPONSE_BYTES) {
    throw new ToolError(TOOL_NAME, `Response too large (${declaredSize} bytes)`);
  }
}

async function buildBinaryFetchResult(
  response: FetchResponseInfo,
  requestedUrl: string,
  sessionId: string,
  status: number,
): Promise<ToolResult> {
  const download = await saveBinaryResponse({
    sessionId,
    finalUrl: response.finalUrl,
    contentType: response.contentType,
    disposition: response.disposition,
    body: response.body,
  });

  return {
    success: true,
    output: buildDownloadOutput({
      requestedUrl,
      finalUrl: response.finalUrl,
      contentType: response.contentType,
      savedPath: download.savedPath,
      savedSizeBytes: download.savedSizeBytes,
      sha256: download.sha256,
    }),
    error: null,
    artifacts: [download.savedPath],
    metadata: {
      requestedUrl,
      finalUrl: response.finalUrl,
      contentType: response.contentType,
      status,
      savedPath: download.savedPath,
      savedSizeBytes: download.savedSizeBytes,
      sha256: download.sha256,
      downloaded: true,
    },
  };
}

function buildTextFetchResult(
  response: FetchResponseInfo,
  requestedUrl: string,
  status: number,
): ToolResult {
  const content = normalizeResponseBody(response.body, response.mime);
  return {
    success: true,
    output: buildOutput({
      requestedUrl,
      finalUrl: response.finalUrl,
      contentType: response.contentType,
      content,
    }),
    error: null,
    artifacts: [],
    metadata: {
      requestedUrl,
      finalUrl: response.finalUrl,
      contentType: response.contentType,
      status,
      downloaded: false,
    },
  };
}

function assertHttpUrl(url: string): void {
  if (!url.startsWith("http://") && !url.startsWith("https://")) {
    throw new ToolError(TOOL_NAME, "URL must use http:// or https://");
  }
}

function extractMimeType(contentType: string): string {
  return contentType.split(";")[0]?.trim().toLowerCase() ?? "";
}

function isConfidentlyTextualResponse(mime: string, body: Uint8Array): boolean {
  if (!mime) {
    return passesUnknownMimeTextSniff(body);
  }
  if (mime.startsWith("text/")) {
    return true;
  }
  if (TEXTUAL_MIME_TYPES.has(mime)) {
    return true;
  }
  if (mime.endsWith("+json") || mime.endsWith("+xml")) {
    return true;
  }
  return false;
}

function passesUnknownMimeTextSniff(body: Uint8Array): boolean {
  const sample = body.subarray(0, Math.min(body.length, 2048));
  if (sample.length === 0) {
    return true;
  }
  if (hasNullByte(sample)) {
    return false;
  }
  if (hasKnownBinarySignature(sample)) {
    return false;
  }

  const decoded = decodeUtf8SampleWithValidatedBoundary(body, sample);
  if (decoded === null) {
    return false;
  }

  return hasPrintableTextRatio(decoded);
}

function decodeUtf8SampleWithValidatedBoundary(
  body: Uint8Array,
  sample: Uint8Array,
): string | null {
  const decoder = new TextDecoder("utf-8", { fatal: true });
  try {
    return decoder.decode(sample);
  } catch {
    if (body.length <= sample.length) {
      return null;
    }

    const trailingSequence = getIncompleteTrailingUtf8Sequence(sample);
    if (trailingSequence === null) {
      return null;
    }

    try {
      const decodedPrefix = decoder.decode(sample.subarray(0, trailingSequence.sequenceStart));
      const sequenceEnd = trailingSequence.sequenceStart + trailingSequence.expectedLength;
      if (body.length < sequenceEnd) {
        return null;
      }
      const decodedSequence = decoder.decode(
        body.subarray(trailingSequence.sequenceStart, sequenceEnd),
      );
      return decodedPrefix + decodedSequence;
    } catch {
      return null;
    }
  }
}

function getIncompleteTrailingUtf8Sequence(sample: Uint8Array): {
  readonly sequenceStart: number;
  readonly expectedLength: number;
} | null {
  let continuationCount = 0;
  let index = sample.length - 1;
  while (index >= 0 && continuationCount < 3 && isContinuationByte(sample[index]!)) {
    continuationCount++;
    index--;
  }

  if (index < 0) {
    return null;
  }

  const leadByte = sample[index]!;
  const expectedLength = utf8SequenceLength(leadByte);
  if (expectedLength === null || expectedLength === 1) {
    return null;
  }

  const actualLength = sample.length - index;
  if (actualLength >= expectedLength) {
    return null;
  }

  for (let offset = index + 1; offset < sample.length; offset++) {
    if (!isContinuationByte(sample[offset]!)) {
      return null;
    }
  }

  return {
    sequenceStart: index,
    expectedLength,
  };
}

function isContinuationByte(byte: number): boolean {
  return byte >= 0x80 && byte <= 0xbf;
}

function utf8SequenceLength(byte: number): number | null {
  if (byte <= 0x7f) {
    return 1;
  }
  if (byte >= 0xc2 && byte <= 0xdf) {
    return 2;
  }
  if (byte >= 0xe0 && byte <= 0xef) {
    return 3;
  }
  if (byte >= 0xf0 && byte <= 0xf4) {
    return 4;
  }
  return null;
}

function normalizeResponseBody(body: Uint8Array, mime: string): string {
  const text = new TextDecoder("utf-8").decode(body).trim();
  if (!text) {
    return "(empty response body)";
  }

  if (mime === "text/html" || mime === "application/xhtml+xml") {
    return normalizeHtml(text);
  }
  if (mime === "application/json" || mime.endsWith("+json")) {
    try {
      return JSON.stringify(JSON.parse(text) as unknown, null, 2);
    } catch {
      return text;
    }
  }
  return text;
}

function normalizeHtml(html: string): string {
  const titleMatch = html.match(/<title[^>]*>([\s\S]*?)<\/title>/i);
  const title = titleMatch ? collapseWhitespace(decodeBasicEntities(titleMatch[1] ?? "")) : "";
  const markdown = collapseWhitespacePreservingParagraphs(
    HTML_TO_MARKDOWN.translate(html).trim(),
  );
  if (title && !markdown.includes(title)) {
    return `# ${title}\n\n${markdown}`.trim();
  }
  return markdown || (title ? `# ${title}` : "(empty HTML response)");
}

function buildOutput(input: {
  readonly requestedUrl: string;
  readonly finalUrl: string;
  readonly contentType: string;
  readonly content: string;
}): string {
  const prefix = [
    `Requested URL: ${input.requestedUrl}`,
    `Final URL: ${input.finalUrl}`,
    `Content-Type: ${input.contentType}`,
    "",
  ].join("\n");

  const content = input.content.length > MAX_OUTPUT_CHARS
    ? `${input.content.slice(0, MAX_OUTPUT_CHARS)}\n\n[... output truncated ...]`
    : input.content;

  return `${prefix}${content}`;
}

function buildDownloadOutput(input: {
  readonly requestedUrl: string;
  readonly finalUrl: string;
  readonly contentType: string;
  readonly savedPath: string;
  readonly savedSizeBytes: number;
  readonly sha256: string;
}): string {
  return [
    `Requested URL: ${input.requestedUrl}`,
    `Final URL: ${input.finalUrl}`,
    `Content-Type: ${input.contentType}`,
    `Saved Path: ${input.savedPath}`,
    `Saved Size: ${input.savedSizeBytes} bytes`,
    `SHA-256: ${input.sha256}`,
    "",
    "Binary or attachment response saved to disk for later inspection.",
  ].join("\n");
}

async function readResponseBody(
  response: Response,
  controller: AbortController,
  maxBytes: number,
): Promise<Uint8Array> {
  if (!response.body) {
    return new Uint8Array();
  }

  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let totalBytes = 0;

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      totalBytes += value.byteLength;
      if (totalBytes > maxBytes) {
        controller.abort(new Error("response-too-large"));
        throw new ToolError(TOOL_NAME, `Response too large (${totalBytes} bytes)`);
      }
      chunks.push(value);
    }
  } finally {
    reader.releaseLock();
  }

  const output = new Uint8Array(totalBytes);
  let offset = 0;
  for (const chunk of chunks) {
    output.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return output;
}

function collapseWhitespace(value: string): string {
  return value.replace(/\s+/g, " ").trim();
}

function collapseWhitespacePreservingParagraphs(value: string): string {
  return value
    .replace(/\r\n/g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .replace(/[ \t]+\n/g, "\n")
    .replace(/\n[ \t]+/g, "\n")
    .trim();
}

function decodeBasicEntities(value: string): string {
  return value
    .replace(/&amp;/g, "&")
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&quot;/g, "\"")
    .replace(/&#39;/g, "'");
}

function hasNullByte(body: Uint8Array): boolean {
  for (const byte of body) {
    if (byte === 0) {
      return true;
    }
  }
  return false;
}

function hasKnownBinarySignature(sample: Uint8Array): boolean {
  return startsWithBytes(sample, [0x25, 0x50, 0x44, 0x46, 0x2d]) // %PDF-
    || startsWithBytes(sample, [0x50, 0x4b, 0x03, 0x04]) // zip
    || startsWithBytes(sample, [0x89, 0x50, 0x4e, 0x47]) // png
    || startsWithBytes(sample, [0xff, 0xd8, 0xff]) // jpeg
    || startsWithBytes(sample, [0x47, 0x49, 0x46, 0x38]) // GIF8
    || startsWithBytes(sample, [0x1f, 0x8b, 0x08]) // gzip
    || isWebP(sample);
}

function startsWithBytes(sample: Uint8Array, expected: ReadonlyArray<number>): boolean {
  if (sample.length < expected.length) {
    return false;
  }
  for (let index = 0; index < expected.length; index++) {
    if (sample[index] !== expected[index]) {
      return false;
    }
  }
  return true;
}

function isWebP(sample: Uint8Array): boolean {
  if (sample.length < 12) {
    return false;
  }
  return startsWithBytes(sample, [0x52, 0x49, 0x46, 0x46]) // RIFF
    && sample[8] === 0x57 // W
    && sample[9] === 0x45 // E
    && sample[10] === 0x42 // B
    && sample[11] === 0x50; // P
}

function hasPrintableTextRatio(decoded: string): boolean {
  if (decoded.length === 0) {
    return true;
  }

  let printableChars = 0;
  for (const char of decoded) {
    const code = char.codePointAt(0) ?? 0;
    const isWhitespace = code === 0x09 || code === 0x0a || code === 0x0d;
    const isPrintable = isWhitespace
      || (code >= 0x20 && code < 0x7f)
      || code >= 0x00a0;
    if (isPrintable) {
      printableChars++;
    }
  }

  return printableChars / decoded.length >= 0.85;
}

function isTimeoutAbort(reason: unknown): boolean {
  return reason instanceof Error && reason.message === "timeout";
}

async function saveBinaryResponse(input: {
  readonly sessionId: string;
  readonly finalUrl: string;
  readonly contentType: string;
  readonly disposition: string;
  readonly body: Uint8Array;
}): Promise<{
  savedPath: string;
  savedSizeBytes: number;
  sha256: string;
}> {
  const sha256 = createHash("sha256").update(input.body).digest("hex");
  const sessionDir = join(DOWNLOAD_ROOT_DIR, sanitizePathSegment(input.sessionId));
  await mkdir(sessionDir, { recursive: true });

  const filename = buildSavedFilename({
    disposition: input.disposition,
    finalUrl: input.finalUrl,
    contentType: input.contentType,
    sha256,
  });
  const savedPath = join(sessionDir, filename);
  await writeFile(savedPath, input.body);

  return {
    savedPath,
    savedSizeBytes: input.body.byteLength,
    sha256,
  };
}

export function createProxyAwareFetch(
  baseFetch: FetchFn = globalThis.fetch.bind(globalThis),
  options?: ProxyAwareFetchOptions,
): FetchFn {
  return createCoreProxyAwareFetch(baseFetch, {
    ...options,
    createUnsupportedProxyError: options?.createUnsupportedProxyError
      ?? ((message) => new ToolError(TOOL_NAME, message)),
  });
}
