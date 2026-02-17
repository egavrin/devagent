/**
 * @devagent/providers — LLM provider abstraction.
 * Registry-based system with Vercel AI SDK implementations.
 */

export { ProviderRegistry } from "./registry.js";
export type { ProviderFactory } from "./registry.js";
export { createAnthropicProvider } from "./anthropic.js";
export { createOpenAIProvider, resolveCapabilities } from "./openai.js";

import { ProviderRegistry } from "./registry.js";
import { createAnthropicProvider } from "./anthropic.js";
import { createOpenAIProvider } from "./openai.js";

/**
 * Create a registry with all built-in providers registered.
 */
export function createDefaultRegistry(): ProviderRegistry {
  const registry = new ProviderRegistry();
  registry.register("anthropic", createAnthropicProvider);
  registry.register("openai", createOpenAIProvider);
  return registry;
}
