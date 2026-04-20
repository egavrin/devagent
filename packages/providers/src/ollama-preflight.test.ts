import { ProviderError } from "@devagent/runtime";
import { afterEach, describe, expect, it, vi } from "vitest";

import { validateOllamaModel } from "./ollama-preflight.js";

describe("validateOllamaModel", () => {
  const originalFetch = globalThis.fetch;

  afterEach(() => {
    globalThis.fetch = originalFetch;
    vi.restoreAllMocks();
  });

  it("resolves when model is available (exact name:tag match)", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [
          { name: "qwen3-coder:30b", model: "qwen3-coder:30b" },
          { name: "llama3:latest", model: "llama3:latest" },
        ],
      }),
    });

    await expect(
      validateOllamaModel("qwen3-coder:30b", "http://localhost:11434/v1"),
    ).resolves.toBeUndefined();
  });

  it("resolves when model matches by base name (without tag)", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [
          { name: "llama3:latest", model: "llama3:latest" },
        ],
      }),
    });

    await expect(
      validateOllamaModel("llama3", "http://localhost:11434/v1"),
    ).resolves.toBeUndefined();
  });

  it("throws ProviderError with 'ollama pull' suggestion when model not found", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [
          { name: "llama3:latest", model: "llama3:latest" },
          { name: "mistral:latest", model: "mistral:latest" },
        ],
      }),
    });

    await expect(
      validateOllamaModel("qwen3-coder:30b", "http://localhost:11434/v1"),
    ).rejects.toThrow(ProviderError);

    await expect(
      validateOllamaModel("qwen3-coder:30b", "http://localhost:11434/v1"),
    ).rejects.toThrow("ollama pull qwen3-coder:30b");

    await expect(
      validateOllamaModel("qwen3-coder:30b", "http://localhost:11434/v1"),
    ).rejects.toThrow("Available models:");
  });

  it("throws ProviderError when Ollama is unreachable", async () => {
    globalThis.fetch = vi.fn().mockRejectedValue(new Error("fetch failed"));

    await expect(
      validateOllamaModel("llama3", "http://localhost:11434/v1"),
    ).rejects.toThrow(ProviderError);

    await expect(
      validateOllamaModel("llama3", "http://localhost:11434/v1"),
    ).rejects.toThrow("Cannot connect to Ollama");

    await expect(
      validateOllamaModel("llama3", "http://localhost:11434/v1"),
    ).rejects.toThrow("ollama serve");
  });

  it("throws ProviderError when no models are available", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({ models: [] }),
    });

    await expect(
      validateOllamaModel("llama3", "http://localhost:11434/v1"),
    ).rejects.toThrow(ProviderError);

    await expect(
      validateOllamaModel("llama3", "http://localhost:11434/v1"),
    ).rejects.toThrow("No models found");
  });

  it("throws ProviderError on non-OK HTTP response", async () => {
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: false,
      status: 500,
      statusText: "Internal Server Error",
    });

    await expect(
      validateOllamaModel("llama3", "http://localhost:11434/v1"),
    ).rejects.toThrow(ProviderError);

    await expect(
      validateOllamaModel("llama3", "http://localhost:11434/v1"),
    ).rejects.toThrow("500");
  });

  it("strips /v1 suffix from baseUrl when calling Ollama API", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [{ name: "llama3:latest", model: "llama3:latest" }],
      }),
    });
    globalThis.fetch = mockFetch;

    await validateOllamaModel("llama3", "http://localhost:11434/v1");

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:11434/api/tags",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });

  it("uses default baseUrl when none provided", async () => {
    const mockFetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        models: [{ name: "llama3:latest", model: "llama3:latest" }],
      }),
    });
    globalThis.fetch = mockFetch;

    await validateOllamaModel("llama3");

    expect(mockFetch).toHaveBeenCalledWith(
      "http://localhost:11434/api/tags",
      expect.objectContaining({ signal: expect.any(AbortSignal) }),
    );
  });
});
