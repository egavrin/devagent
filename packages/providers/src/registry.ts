/**
 * Provider registry — manages LLM provider instances.
 * Fail fast: throws on unknown provider or missing config.
 */

import type { LLMProvider, ProviderConfig } from "@devagent/core";
import { ProviderError } from "@devagent/core";

export type ProviderFactory = (config: ProviderConfig) => LLMProvider;

export class ProviderRegistry {
  private readonly factories = new Map<string, ProviderFactory>();
  private readonly instances = new Map<string, LLMProvider>();

  register(name: string, factory: ProviderFactory): void {
    this.factories.set(name, factory);
  }

  get(name: string, config: ProviderConfig): LLMProvider {
    // Return cached instance if available
    const cached = this.instances.get(name);
    if (cached) return cached;

    const factory = this.factories.get(name);
    if (!factory) {
      const available = Array.from(this.factories.keys()).join(", ");
      throw new ProviderError(
        `Unknown provider "${name}". Available: ${available}`,
      );
    }

    const instance = factory(config);
    this.instances.set(name, instance);
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
