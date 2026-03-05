/**
 * @devagent/providers — LLM provider abstraction.
 * Registry-based system with Vercel AI SDK implementations.
 */

export { validateOllamaModel } from "./ollama-preflight.js";

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
 * ChatGPT provider — wraps OpenAI provider for ChatGPT subscription accounts.
 * Uses OAuth bearer token instead of API key, routes to Codex endpoint.
 * Forces Responses API (Codex endpoint only supports /responses, not /chat/completions).
 *
 * The Codex endpoint requires specific request body fields:
 * - `store: false` — stateless, don't persist responses
 * - `include: ["reasoning.encrypted_content"]` — reasoning context preservation
 * - `instructions` — required top-level field (system prompt)
 * - NO `max_output_tokens` — endpoint rejects this field
 */
function createChatGPTProvider(config: ProviderConfig): LLMProvider {
  return createOpenAIProvider({
    ...config,
    apiKey: config.oauthToken ? "unused" : config.apiKey,
    baseUrl: config.baseUrl ?? "https://chatgpt.com/backend-api/codex",
    capabilities: {
      ...config.capabilities,
      useResponsesApi: true,
      reasoning: true,
      supportsTemperature: false,
    },
    codexOptions: {
      store: false,
      include: ["reasoning.encrypted_content"],
      instructions: "You are a helpful assistant.",
    },
  });
}

/**
 * GitHub Copilot provider — wraps OpenAI provider for Copilot accounts.
 * Uses a short-lived Copilot session JWT (exchanged from GitHub OAuth token
 * in resolveProviderCredentials) with Copilot-specific editor headers.
 *
 * The Copilot API is standard OpenAI Chat Completions compatible — uses
 * /chat/completions (NOT Responses API). Auth is via session JWT in
 * customHeaders, NOT oauthToken (to avoid ChatGPT-specific openai-beta header).
 *
 * The Copilot endpoint rejects several fields that the standard OpenAI API
 * accepts: store, metadata, prediction, stream_options, logprobs, top_logprobs.
 * A stripFields list removes them via the existing custom fetch mechanism.
 *
 * Required extra headers: openai-intent, x-request-id (for request tracing).
 */
function createGitHubCopilotProvider(config: ProviderConfig): LLMProvider {
  // Build Copilot-specific headers using the session JWT from oauthToken
  const copilotHeaders: Record<string, string> = {};
  if (config.oauthToken) {
    copilotHeaders["Authorization"] = `Bearer ${config.oauthToken}`;
    copilotHeaders["Copilot-Integration-Id"] = "vscode-chat";
    copilotHeaders["Editor-Version"] = "vscode/1.104.1";
    copilotHeaders["Editor-Plugin-Version"] = "copilot-chat/0.26.7";
    copilotHeaders["User-Agent"] = "GitHubCopilotChat/0.26.7";
    copilotHeaders["openai-intent"] = "conversation-panel";
  }

  return createOpenAIProvider({
    ...config,
    // Use "unused" as apiKey — auth is via customHeaders Authorization
    apiKey: config.oauthToken ? "unused" : (config.apiKey ?? ""),
    baseUrl: config.baseUrl ?? "https://api.githubcopilot.com",
    // Clear oauthToken to avoid ChatGPT-specific header injection
    oauthToken: undefined,
    oauthAccountId: undefined,
    // Pass Copilot headers via customHeaders instead
    customHeaders: Object.keys(copilotHeaders).length > 0 ? copilotHeaders : undefined,
    // Strip fields the Copilot endpoint rejects
    stripFields: [
      "store", "metadata", "prediction",
      "stream_options", "logprobs", "top_logprobs",
    ],
  });
}

/**
 * DeepSeek provider — wraps the OpenAI provider for DeepSeek's API.
 * DeepSeek exposes an OpenAI-compatible API at api.deepseek.com/v1.
 */
function createDeepSeekProvider(config: ProviderConfig): LLMProvider {
  return createOpenAIProvider({
    ...config,
    baseUrl: config.baseUrl ?? "https://api.deepseek.com/v1",
  });
}

/**
 * OpenRouter provider — wraps the OpenAI provider for OpenRouter's API.
 * OpenRouter exposes an OpenAI-compatible API at openrouter.ai/api/v1.
 */
function createOpenRouterProvider(config: ProviderConfig): LLMProvider {
  return createOpenAIProvider({
    ...config,
    baseUrl: config.baseUrl ?? "https://openrouter.ai/api/v1",
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
  registry.register("deepseek", createDeepSeekProvider);
  registry.register("openrouter", createOpenRouterProvider);
  registry.register("chatgpt", createChatGPTProvider);
  registry.register("github-copilot", createGitHubCopilotProvider);
  return registry;
}
