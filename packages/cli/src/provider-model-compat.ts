import { getProvidersForModel, isModelRegisteredForProvider } from "@devagent/runtime";

export interface ProviderModelCompatibilityIssue {
  readonly model: string;
  readonly configuredProvider: string;
  readonly supportedProviders: ReadonlyArray<string>;
}

export function getProviderModelCompatibilityIssue(
  provider: string,
  model: string,
): ProviderModelCompatibilityIssue | undefined {
  const supportedProviders = getProvidersForModel(model);
  if (supportedProviders.length === 0 || isModelRegisteredForProvider(provider, model)) {
    return undefined;
  }

  return {
    model,
    configuredProvider: provider,
    supportedProviders,
  };
}

export function formatProviderModelCompatibilityError(
  issue: ProviderModelCompatibilityIssue,
): string {
  const registeredProviders = issue.supportedProviders.map((provider) => `"${provider}"`).join(", ");
  return `Configured model "${issue.model}" is not registered for provider "${issue.configuredProvider}". It is registered for ${registeredProviders}. Switch provider or choose a model registered for "${issue.configuredProvider}".`;
}

export function formatProviderModelCompatibilityHint(
  issue: ProviderModelCompatibilityIssue,
): string | undefined {
  if (issue.model === "cortex" || issue.supportedProviders.includes("devagent-api")) {
    return 'Try "--provider devagent-api --model cortex" for the deployed Devagent API gateway.';
  }
  return undefined;
}
