/**
 * Ollama pre-flight validation — checks model availability before starting a session.
 * Queries Ollama's /api/tags endpoint to list locally pulled models.
 * Fails fast with actionable error messages.
 */

import { ProviderError } from "@devagent/runtime";

interface OllamaModel {
  name: string;
  model: string;
}

interface OllamaTagsResponse {
  models: OllamaModel[];
}

/**
 * Pre-flight check: query Ollama /api/tags to verify the requested model is available.
 * Throws ProviderError with actionable message if:
 * - Ollama is unreachable (suggests `ollama serve`)
 * - Model isn't pulled locally (suggests `ollama pull <model>`, lists available models)
 */
export async function validateOllamaModel(
  model: string,
  baseUrl: string = "http://localhost:11434",
): Promise<void> {
  // Strip /v1 suffix — Ollama native API doesn't use it
  const ollamaBase = baseUrl.replace(/\/v1\/?$/, "");

  let response: Response;
  try {
    response = await fetch(`${ollamaBase}/api/tags`, {
      signal: AbortSignal.timeout(5000),
    });
  } catch {
    throw new ProviderError(
      `Cannot connect to Ollama at ${ollamaBase}. ` +
        `Is Ollama running? Start it with: ollama serve`,
    );
  }

  if (!response.ok) {
    throw new ProviderError(
      `Ollama API error (${response.status}): ${response.statusText}`,
    );
  }

  const data = (await response.json()) as OllamaTagsResponse;
  const availableModels = data.models ?? [];

  // Normalize model names for matching — Ollama uses "name:tag" format
  // User might pass "qwen3-coder:30b" or just "qwen3-coder" (implies any tag)
  const requestedBase = model.split(":")[0]!;
  const requestedTag = model.includes(":") ? model.split(":")[1]! : null;

  const match = availableModels.find((m) => {
    const mBase = m.name.split(":")[0]!;
    const mTag = m.name.split(":")[1] ?? "latest";
    if (requestedTag) {
      return mBase === requestedBase && mTag === requestedTag;
    }
    return mBase === requestedBase;
  });

  if (!match) {
    const modelNames = availableModels.map((m) => m.name).join(", ");
    throw new ProviderError(
      `Model '${model}' is not available in Ollama. ` +
        `Pull it first: ollama pull ${model}\n` +
        (modelNames
          ? `Available models: ${modelNames}`
          : `No models found. Pull one with: ollama pull <model>`),
    );
  }
}
