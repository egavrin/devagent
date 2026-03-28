import type {
  AgentType,
  DevAgentConfig,
  ModelCapabilities,
  ProviderConfig,
  ReasoningEffort,
} from "@devagent/runtime";
import { lookupModelCapabilities } from "@devagent/runtime";

export function buildProviderConfig(
  config: DevAgentConfig,
  reasoningEffort?: ReasoningEffort,
  agentType?: AgentType,
): ProviderConfig {
  const baseProviderConfig = config.providers[config.provider] ?? {
    model: config.model,
    apiKey: process.env["DEVAGENT_API_KEY"],
  };
  const registryCaps = lookupModelCapabilities(config.model);
  const resolvedReasoningEffort = agentType
    ? config.agentReasoningOverrides?.[agentType] ?? reasoningEffort
    : reasoningEffort;

  return {
    ...baseProviderConfig,
    model: config.model,
    ...(resolvedReasoningEffort ? { reasoningEffort: resolvedReasoningEffort } : {}),
    ...(!baseProviderConfig.capabilities && registryCaps
      ? { capabilities: registryCaps as ModelCapabilities }
      : {}),
  };
}
