import { createHash } from "node:crypto";
import { readFile, rm } from "node:fs/promises";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { afterEach, describe, expect, it, vi } from "vitest";

import { createProxyAwareFetch, fetchUrlTool } from "./fetch-url.js";
import type { ToolContext } from "../../core/types.js";
import type { AddressInfo } from "node:net";

const TEST_SESSION_ID = "fetch-url-test";
const DOWNLOAD_DIR = join(tmpdir(), "devagent-fetch-url", TEST_SESSION_ID);
const PROXY_ENV_VARS = ["HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"] as const;

function makeCtx(): ToolContext {
  return {
    repoRoot: "/tmp",
    config: {} as ToolContext["config"],
    sessionId: TEST_SESSION_ID,
  };
}

async function withServer(
  handler: (req: IncomingMessage, res: ServerResponse<IncomingMessage>) => void,
): Promise<{ url: string; close: () => Promise<void> }> {
  const server = createServer(handler);
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(0, "127.0.0.1", () => resolve());
  });

  const address = server.address();
  if (!address || typeof address === "string") {
    throw new Error("Failed to bind test server");
  }

  return {
    url: `http://127.0.0.1:${(address as AddressInfo).port}`,
    close: async () => {
      await new Promise<void>((resolve, reject) => {
        server.close((error) => error ? reject(error) : resolve());
      });
    },
  };
}

async function runUnknownMimeRequest(
  closers: Array<() => Promise<void>>,
  body: string | Buffer,
  path: string,
  saveBinary = false,
) {
  const server = await withServer((_req, res) => {
    res.writeHead(200);
    res.end(body);
  });
  closers.push(server.close);

  return fetchUrlTool.handler(
    saveBinary ? { url: `${server.url}/${path}`, save_binary: true } : { url: `${server.url}/${path}` },
    makeCtx(),
  );
}

async function expectUnknownMimeInline(input: {
  readonly closers: Array<() => Promise<void>>;
  readonly body: string | Buffer;
  readonly path: string;
  readonly expectedOutput: string;
  readonly saveBinary?: boolean;
}): Promise<void> {
  const result = await runUnknownMimeRequest(
    input.closers,
    input.body,
    input.path,
    input.saveBinary,
  );

  expect(result.success).toBe(true);
  expect(result.output).toContain(input.expectedOutput);
  expect(result.artifacts).toEqual([]);
  expect(result.metadata).toMatchObject({
    downloaded: false,
    contentType: "application/octet-stream",
  });
}

async function expectUnknownMimeRejected(
  closers: Array<() => Promise<void>>,
  body: string | Buffer,
  path: string,
): Promise<void> {
  await expect(runUnknownMimeRequest(closers, body, path)).rejects.toThrow("Unsupported content type");
}

async function expectUnknownMimeSaved(
  closers: Array<() => Promise<void>>,
  body: Buffer,
  path: string,
): Promise<void> {
  const result = await runUnknownMimeRequest(closers, body, path, true);

  expect(result.success).toBe(true);
  expect(result.output).toContain("Saved Path:");
  expect(result.artifacts).toHaveLength(1);
  const savedContent = await readFile(result.artifacts[0]!);
  expect(Buffer.compare(savedContent, body)).toBe(0);
  expect(result.metadata).toMatchObject({
    downloaded: true,
    contentType: "application/octet-stream",
  });
}

const repeatedPrefix = "a".repeat(2047);

async function withClearedProxyEnv(fn: () => Promise<void>): Promise<void> {
  const original = new Map<string, string | undefined>();
  for (const key of PROXY_ENV_VARS) {
    original.set(key, process.env[key]);
    delete process.env[key];
  }

  try {
    await fn();
  } finally {
    for (const key of PROXY_ENV_VARS) {
      const value = original.get(key);
      if (value === undefined) delete process.env[key];
      else process.env[key] = value;
    }
  }
}

describe("fetch_url", () => {
  const closers: Array<() => Promise<void>> = [];

  afterEach(async () => {
    while (closers.length > 0) {
      const close = closers.pop();
      if (close) {
        await close();
      }
    }
    vi.restoreAllMocks();
    await rm(DOWNLOAD_DIR, { recursive: true, force: true });
  });

  it("does not load undici when no proxy env vars are set", async () => {
    await withClearedProxyEnv(async () => {
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const loadUndici = vi.fn();
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch, { loadUndici });

      await proxyFetch("https://example.com/docs");

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0]?.[1]).toBeUndefined();
      expect(loadUndici).not.toHaveBeenCalled();
    });
  });

  it("attaches a proxy dispatcher on Node when proxy env vars are set", async () => {
    await withClearedProxyEnv(async () => {
      process.env["HTTPS_PROXY"] = "https://proxy.example.com:8443";
      const dispatcher = { dispatch: vi.fn() };
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch, {
        runtime: "node",
        loadUndici: async () => ({
          EnvHttpProxyAgent: class {
            constructor() {
              return dispatcher;
            }
          },
        }),
      });

      await proxyFetch("https://example.com/docs", { headers: { accept: "text/plain" } });

      const init = fetchMock.mock.calls[0]?.[1] as RequestInit & { dispatcher?: unknown };
      expect(init.headers).toEqual({ accept: "text/plain" });
      expect(init.dispatcher).toBe(dispatcher);
    });
  });

  it("fails clearly under Bun when proxy env vars are set", async () => {
    await withClearedProxyEnv(async () => {
      process.env["HTTPS_PROXY"] = "https://proxy.example.com:8443";
      const fetchMock = vi.fn().mockResolvedValue(new Response("ok"));
      const proxyFetch = createProxyAwareFetch(fetchMock as typeof globalThis.fetch, { runtime: "bun" });

      await expect(proxyFetch("https://example.com/docs")).rejects.toThrow(
        "proxy dispatchers require Node.js",
      );
      expect(fetchMock).not.toHaveBeenCalled();
    });
  });

  it("fetches plain text content", async () => {
    const server = await withServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("hello from devagent");
    });
    closers.push(server.close);

    const result = await fetchUrlTool.handler({ url: `${server.url}/plain.txt` }, makeCtx());

    expect(result.success).toBe(true);
    expect(result.output).toContain("hello from devagent");
    expect(result.artifacts).toEqual([]);
    expect(result.metadata).toMatchObject({
      finalUrl: `${server.url}/plain.txt`,
      contentType: "text/plain; charset=utf-8",
      downloaded: false,
    });
  });

  it("normalizes html into readable markdown-like text", async () => {
    const server = await withServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "text/html; charset=utf-8" });
      res.end(`<!doctype html>
<html>
  <head>
    <title>Example Page</title>
    <style>body { color: red; }</style>
  </head>
  <body>
    <main>
      <h1>Heading</h1>
      <p>Hello <strong>world</strong>.</p>
      <script>window.__ignored = true;</script>
    </main>
  </body>
</html>`);
    });
    closers.push(server.close);

    const result = await fetchUrlTool.handler({ url: `${server.url}/page` }, makeCtx());

    expect(result.success).toBe(true);
    expect(result.output).toContain("Example Page");
    expect(result.output).toContain("Heading");
    expect(result.output).toContain("Hello");
    expect(result.output).toContain("world");
    expect(result.output).not.toContain("__ignored");
  });

  it("keeps textual responses inline even when save_binary is enabled", async () => {
    const server = await withServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "application/json" });
      res.end(JSON.stringify({ ok: true, mode: "inline" }));
    });
    closers.push(server.close);

    const result = await fetchUrlTool.handler(
      { url: `${server.url}/data.json`, save_binary: true },
      makeCtx(),
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain('"mode": "inline"');
    expect(result.artifacts).toEqual([]);
    expect(result.metadata).toMatchObject({
      downloaded: false,
      finalUrl: `${server.url}/data.json`,
    });
  });

  it("rejects unsupported URL schemes", async () => {
    await expect(
      fetchUrlTool.handler({ url: "file:///tmp/nope.txt" }, makeCtx()),
    ).rejects.toThrow("URL must use http:// or https://");
  });

  it("rejects oversized responses using content-length", async () => {
    const server = await withServer((_req, res) => {
      res.writeHead(200, {
        "Content-Type": "text/plain",
        "Content-Length": String(6 * 1024 * 1024),
      });
      res.end("too big");
    });
    closers.push(server.close);

    await expect(
      fetchUrlTool.handler({ url: `${server.url}/large` }, makeCtx()),
    ).rejects.toThrow("Response too large");
  });

  it("rejects oversized binary responses in save mode", async () => {
    const server = await withServer((_req, res) => {
      res.writeHead(200, {
        "Content-Type": "application/octet-stream",
        "Content-Length": String(6 * 1024 * 1024),
      });
      res.end("too big");
    });
    closers.push(server.close);

    await expect(
      fetchUrlTool.handler({ url: `${server.url}/large.bin`, save_binary: true }, makeCtx()),
    ).rejects.toThrow("Response too large");
  });

  it("rejects unsupported binary content by default", async () => {
    const body = Buffer.from([0x89, 0x50, 0x4e, 0x47]);
    const server = await withServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "image/png" });
      res.end(body);
    });
    closers.push(server.close);

    await expect(
      fetchUrlTool.handler({ url: `${server.url}/image.png` }, makeCtx()),
    ).rejects.toThrow("Unsupported content type");
  });

  it("rejects attachment responses by default", async () => {
    const server = await withServer((_req, res) => {
      res.writeHead(200, {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Disposition": "attachment; filename=\"report.txt\"",
      });
      res.end("attached report");
    });
    closers.push(server.close);

    await expect(
      fetchUrlTool.handler({ url: `${server.url}/download` }, makeCtx()),
    ).rejects.toThrow("attachment download");
  });

  it("saves binary content when save_binary is enabled", async () => {
    const body = Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x00, 0x01, 0x02, 0x03]);
    const expectedSha = createHash("sha256").update(body).digest("hex");
    const server = await withServer((_req, res) => {
      res.writeHead(200, { "Content-Type": "image/png" });
      res.end(body);
    });
    closers.push(server.close);

    const result = await fetchUrlTool.handler(
      { url: `${server.url}/image.png`, save_binary: true },
      makeCtx(),
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("Saved Path:");
    expect(result.output).toContain(expectedSha);
    expect(result.artifacts).toHaveLength(1);

    const savedPath = result.artifacts[0]!;
    expect(savedPath.startsWith(DOWNLOAD_DIR)).toBe(true);
    expect(basename(savedPath)).toMatch(/image\.png$/);

    const savedContent = await readFile(savedPath);
    expect(Buffer.compare(savedContent, body)).toBe(0);
    expect(result.metadata).toMatchObject({
      downloaded: true,
      savedPath,
      savedSizeBytes: body.length,
      sha256: expectedSha,
      contentType: "image/png",
      finalUrl: `${server.url}/image.png`,
    });
  });

  it("saves attachment responses and sanitizes the filename", async () => {
    const body = Buffer.from("attached report");
    const expectedSha = createHash("sha256").update(body).digest("hex");
    const server = await withServer((_req, res) => {
      res.writeHead(200, {
        "Content-Type": "text/plain; charset=utf-8",
        "Content-Disposition": "attachment; filename=\"../../Quarterly Report?.txt\"",
      });
      res.end(body);
    });
    closers.push(server.close);

    const result = await fetchUrlTool.handler(
      { url: `${server.url}/download`, save_binary: true },
      makeCtx(),
    );

    expect(result.success).toBe(true);
    expect(result.artifacts).toHaveLength(1);

    const savedPath = result.artifacts[0]!;
    expect(savedPath.startsWith(DOWNLOAD_DIR)).toBe(true);
    expect(basename(savedPath)).toMatch(/Quarterly-Report-.txt$/);

    const savedContent = await readFile(savedPath, "utf-8");
    expect(savedContent).toBe("attached report");
    expect(result.metadata).toMatchObject({
      downloaded: true,
      savedPath,
      savedSizeBytes: body.length,
      sha256: expectedSha,
    });
  });

  it("saves PDF-like bytes with no content-type when save_binary is enabled", async () => {
    const body = Buffer.from("%PDF-1.7\n1 0 obj\n");
    const server = await withServer((_req, res) => {
      res.writeHead(200);
      res.end(body);
    });
    closers.push(server.close);

    const result = await fetchUrlTool.handler(
      { url: `${server.url}/unknown-pdf`, save_binary: true },
      makeCtx(),
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("Saved Path:");
    expect(result.output).toContain("application/octet-stream");
    expect(result.output).not.toContain("%PDF-1.7");
    expect(result.artifacts).toHaveLength(1);
    expect(basename(result.artifacts[0]!)).toMatch(/unknown-pdf$/);
    const savedContent = await readFile(result.artifacts[0]!);
    expect(Buffer.compare(savedContent, body)).toBe(0);
    expect(result.metadata).toMatchObject({
      downloaded: true,
      contentType: "application/octet-stream",
    });
  });

  it("saves ZIP-like bytes with no content-type when save_binary is enabled", async () => {
    const body = Buffer.from([0x50, 0x4b, 0x03, 0x04, 0x14, 0x00, 0x08, 0x00]);
    const server = await withServer((_req, res) => {
      res.writeHead(200);
      res.end(body);
    });
    closers.push(server.close);

    const result = await fetchUrlTool.handler(
      { url: `${server.url}/archive`, save_binary: true },
      makeCtx(),
    );

    expect(result.success).toBe(true);
    expect(result.artifacts).toHaveLength(1);
    expect(result.output).toContain("Saved Path:");
    const savedContent = await readFile(result.artifacts[0]!);
    expect(Buffer.compare(savedContent, body)).toBe(0);
  });

  it("keeps unknown-mime UTF-8 text inline when save_binary is enabled", async () => {
    await expectUnknownMimeInline({
      closers,
      body: "plain text without a content type",
      path: "unknown-text",
      expectedOutput: "plain text without a content type",
      saveBinary: true,
    });
  });

  for (const testCase of [
    {
      name: "keeps long unknown-mime UTF-8 text inline when the sample ends mid-character",
      body: `${repeatedPrefix}é kept inline after boundary`,
      path: "unknown-text-boundary",
      expectedOutput: "kept inline after boundary",
    },
    {
      name: "keeps long unknown-mime UTF-8 text inline with save_binary when the sample ends mid-character",
      body: `${repeatedPrefix}é kept inline after boundary`,
      path: "unknown-text-boundary-save",
      expectedOutput: "kept inline after boundary",
      saveBinary: true,
    },
    {
      name: "keeps long unknown-mime UTF-8 text inline when the sample ends inside a 3-byte character",
      body: Buffer.concat([
        Buffer.from(repeatedPrefix),
        Buffer.from([0xe2, 0x82, 0xac]),
        Buffer.from(" completed after boundary"),
      ]),
      path: "unknown-text-boundary-3-byte",
      expectedOutput: "€ completed after boundary",
    },
    {
      name: "keeps long unknown-mime UTF-8 text inline with save_binary when the sample ends inside a 3-byte character",
      body: Buffer.concat([
        Buffer.from(repeatedPrefix),
        Buffer.from([0xe2, 0x82, 0xac]),
        Buffer.from(" completed after boundary"),
      ]),
      path: "unknown-text-boundary-3-byte-save",
      expectedOutput: "€ completed after boundary",
      saveBinary: true,
    },
    {
      name: "keeps long unknown-mime UTF-8 text inline when the sample ends inside a 4-byte character",
      body: Buffer.concat([
        Buffer.from(repeatedPrefix),
        Buffer.from([0xf0, 0x9f, 0x98, 0x80]),
        Buffer.from(" emoji completed after boundary"),
      ]),
      path: "unknown-text-boundary-4-byte",
      expectedOutput: "😀 emoji completed after boundary",
    },
    {
      name: "keeps long unknown-mime UTF-8 text inline with save_binary when the sample ends inside a 4-byte character",
      body: Buffer.concat([
        Buffer.from(repeatedPrefix),
        Buffer.from([0xf0, 0x9f, 0x98, 0x80]),
        Buffer.from(" emoji completed after boundary"),
      ]),
      path: "unknown-text-boundary-4-byte-save",
      expectedOutput: "😀 emoji completed after boundary",
      saveBinary: true,
    },
  ]) {
    it(testCase.name, async () => {
      await expectUnknownMimeInline({
        closers,
        body: testCase.body,
        path: testCase.path,
        expectedOutput: testCase.expectedOutput,
        saveBinary: testCase.saveBinary,
      });
    });
  }

  for (const testCase of [
    {
      name: "rejects a short unknown-mime body with an incomplete trailing UTF-8 sequence by default",
      body: Buffer.from([0x68, 0x69, 0xc3]),
      path: "malformed-short",
    },
    {
      name: "rejects a longer unknown-mime body when the next byte after a trailing lead byte is not a UTF-8 continuation",
      body: Buffer.concat([Buffer.from(repeatedPrefix), Buffer.from([0xc3, 0x78])]),
      path: "malformed-boundary",
    },
    {
      name: "rejects a longer unknown-mime body when the next byte is a continuation byte but the completed 3-byte sequence is invalid",
      body: Buffer.concat([Buffer.from(repeatedPrefix), Buffer.from([0xe0, 0x80, 0x61])]),
      path: "malformed-boundary-invalid-sequence",
    },
    {
      name: "rejects a longer unknown-mime body when the completed 4-byte sequence is invalid",
      body: Buffer.concat([Buffer.from(repeatedPrefix), Buffer.from([0xf0, 0x80, 0x80, 0x80])]),
      path: "malformed-boundary-invalid-4-byte",
    },
    {
      name: "rejects a truncated-but-still-incomplete unknown-mime UTF-8 sequence by default",
      body: Buffer.concat([Buffer.from(repeatedPrefix), Buffer.from([0xf0, 0x9f])]),
      path: "malformed-boundary-still-incomplete",
    },
  ]) {
    it(testCase.name, async () => {
      await expectUnknownMimeRejected(closers, testCase.body, testCase.path);
    });
  }

  for (const testCase of [
    {
      name: "saves a short unknown-mime body with an incomplete trailing UTF-8 sequence when save_binary is enabled",
      body: Buffer.from([0x68, 0x69, 0xc3]),
      path: "malformed-short-save",
    },
    {
      name: "saves a longer unknown-mime body when the next byte after a trailing lead byte is not a UTF-8 continuation and save_binary is enabled",
      body: Buffer.concat([Buffer.from(repeatedPrefix), Buffer.from([0xc3, 0x78])]),
      path: "malformed-boundary-save",
    },
    {
      name: "saves a longer unknown-mime body when the next byte is a continuation byte but the completed 3-byte sequence is invalid and save_binary is enabled",
      body: Buffer.concat([Buffer.from(repeatedPrefix), Buffer.from([0xe0, 0x80, 0x61])]),
      path: "malformed-boundary-invalid-sequence-save",
    },
    {
      name: "saves a longer unknown-mime body when the completed 4-byte sequence is invalid and save_binary is enabled",
      body: Buffer.concat([Buffer.from(repeatedPrefix), Buffer.from([0xf0, 0x80, 0x80, 0x80])]),
      path: "malformed-boundary-invalid-4-byte-save",
    },
    {
      name: "saves a truncated-but-still-incomplete unknown-mime UTF-8 sequence when save_binary is enabled",
      body: Buffer.concat([Buffer.from(repeatedPrefix), Buffer.from([0xf0, 0x9f])]),
      path: "malformed-boundary-still-incomplete-save",
    },
  ]) {
    it(testCase.name, async () => {
      await expectUnknownMimeSaved(closers, testCase.body, testCase.path);
    });
  }

  it("uses URL and MIME fallbacks when no filename is available", async () => {
    const body = Buffer.from("%PDF-1.7");
    const server = await withServer((_req, res) => {
      res.writeHead(200, {
        "Content-Type": "application/pdf",
      });
      res.end(body);
    });
    closers.push(server.close);

    const result = await fetchUrlTool.handler(
      { url: `${server.url}/`, save_binary: true },
      makeCtx(),
    );

    expect(result.success).toBe(true);
    expect(result.artifacts).toHaveLength(1);
    expect(basename(result.artifacts[0]!)).toMatch(/download\.pdf$/);
  });

  it("fails when the request times out", async () => {
    const server = await withServer((_req, _res) => {
      // Intentionally leave the response hanging to trigger a timeout.
    });
    closers.push(server.close);

    await expect(
      fetchUrlTool.handler({ url: `${server.url}/slow`, timeout_ms: 25 }, makeCtx()),
    ).rejects.toThrow("Request timed out");
  });

  it("fails when a binary download request times out", async () => {
    const server = await withServer((_req, _res) => {
      // Intentionally leave the response hanging to trigger a timeout.
    });
    closers.push(server.close);

    await expect(
      fetchUrlTool.handler(
        { url: `${server.url}/slow.bin`, timeout_ms: 25, save_binary: true },
        makeCtx(),
      ),
    ).rejects.toThrow("Request timed out");
  });
});
