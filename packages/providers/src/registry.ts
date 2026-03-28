/**
 * Provider registry — manages LLM provider instances.
 * Fail fast: throws on unknown provider or missing config.
 */

import type { LLMProvider, ProviderConfig } from "@devagent/runtime";
import { ProviderError } from "@devagent/runtime";

export type ProviderFactory = (config: ProviderConfig) => LLMProvider;

export class ProviderRegistry {
  private readonly factories = new Map<string, ProviderFactory>();
  private readonly instances = new Map<string, LLMProvider>();

  register(name: string, factory: ProviderFactory): void {
    this.factories.set(name, factory);
  }

  get(name: string, config: ProviderConfig): LLMProvider {
    // Return cached instance if available
    const cacheKey = `${name}:${JSON.stringify(config)}`;
    const cached = this.instances.get(cacheKey);
    if (cached) return cached;

    const factory = this.factories.get(name);
    if (!factory) {
      const available = Array.from(this.factories.keys()).join(", ");
      throw new ProviderError(
        `Unknown provider "${name}". Available: ${available}`,
      );
    }

    const instance = factory(config);
    this.instances.set(cacheKey, instance);
    return instance;
  }

  has(name: string): boolean {
    return this.factories.has(name);
  }

  list(): ReadonlyArray<string> {
    return Array.from(this.factories.keys());
  }

  clear(): void {
    this.instances.clear();
  }
}
