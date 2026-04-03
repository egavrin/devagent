import { lookupModelEntry } from "@devagent/runtime";

export interface ProviderModelCompatibilityIssue {
  readonly model: string;
  readonly configuredProvider: string;
  readonly expectedProvider: string;
}

export function getProviderModelCompatibilityIssue(
  provider: string,
  model: string,
): ProviderModelCompatibilityIssue | undefined {
  const registryEntry = lookupModelEntry(model);
  if (!registryEntry || registryEntry.provider === provider) {
    return undefined;
  }

  return {
    model,
    configuredProvider: provider,
    expectedProvider: registryEntry.provider,
  };
}

export function formatProviderModelCompatibilityError(
  issue: ProviderModelCompatibilityIssue,
): string {
  return `Configured model "${issue.model}" belongs to provider "${issue.expectedProvider}"; current provider is "${issue.configuredProvider}". Switch provider or choose a model registered for "${issue.configuredProvider}".`;
}

export function formatProviderModelCompatibilityHint(
  issue: ProviderModelCompatibilityIssue,
): string | undefined {
  if (issue.model === "cortex" || issue.expectedProvider === "devagent-api") {
    return 'Try "--provider devagent-api --model cortex" for the deployed Devagent API gateway.';
  }
  return undefined;
}
