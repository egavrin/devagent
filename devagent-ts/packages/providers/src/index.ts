/**
 * @devagent/providers — LLM provider abstraction.
 * Registry-based system with Vercel AI SDK implementations.
 */

export { ProviderRegistry } from "./registry.js";
export type { ProviderFactory } from "./registry.js";
export { createAnthropicProvider } from "./anthropic.js";
export { createOpenAIProvider, resolveCapabilities } from "./openai.js";

import type { LLMProvider, ProviderConfig } from "@devagent/core";
import { ProviderRegistry } from "./registry.js";
import { createAnthropicProvider } from "./anthropic.js";
import { createOpenAIProvider } from "./openai.js";

/**
 * Ollama provider — wraps the OpenAI provider with local defaults.
 * Ollama exposes an OpenAI-compatible API at /v1.
 */
function createOllamaProvider(config: ProviderConfig): LLMProvider {
  return createOpenAIProvider({
    ...config,
    apiKey: config.apiKey || "ollama", // Ollama ignores this but SDK wants non-empty
    baseUrl: config.baseUrl ?? "http://localhost:11434/v1",
  });
}

/**
 * Create a registry with all built-in providers registered.
 */
export function createDefaultRegistry(): ProviderRegistry {
  const registry = new ProviderRegistry();
  registry.register("anthropic", createAnthropicProvider);
  registry.register("openai", createOpenAIProvider);
  registry.register("ollama", createOllamaProvider);
  return registry;
}
