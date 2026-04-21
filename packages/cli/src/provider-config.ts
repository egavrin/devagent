import { lookupModelCapabilities } from "@devagent/runtime";

import type {
  AgentType,
  DevAgentConfig,
  ProviderConfig,
  ReasoningEffort,
} from "@devagent/runtime";
export function buildProviderConfig(
  config: DevAgentConfig,
  reasoningEffort?: ReasoningEffort,
  agentType?: AgentType,
): ProviderConfig {
  const baseProviderConfig = config.providers[config.provider] ?? {
    model: config.model,
  };
  const registryCaps = lookupModelCapabilities(config.model, config.provider);
  const resolvedReasoningEffort = resolveReasoningEffort(config, reasoningEffort, agentType);
  const shouldUseDevagentApiMainDefault =
    config.provider === "devagent-api" &&
    agentType === undefined &&
    resolvedReasoningEffort === undefined &&
    baseProviderConfig.reasoningEffort === undefined;
  const reasoningConfig = resolvedReasoningEffort
    ? { reasoningEffort: resolvedReasoningEffort }
    : getDefaultReasoningConfig(shouldUseDevagentApiMainDefault);
  const capabilityConfig = !baseProviderConfig.capabilities && registryCaps
    ? { capabilities: registryCaps }
    : {};

  return {
    ...baseProviderConfig,
    model: config.model,
    ...reasoningConfig,
    ...capabilityConfig,
  };
}

function getDefaultReasoningConfig(enabled: boolean): Partial<ProviderConfig> {
  return enabled ? { reasoningEffort: "high" } : {};
}

function resolveReasoningEffort(
  config: DevAgentConfig,
  reasoningEffort: ReasoningEffort | undefined,
  agentType: AgentType | undefined,
): ReasoningEffort | undefined {
  return agentType ? config.agentReasoningOverrides?.[agentType] ?? reasoningEffort : reasoningEffort;
}
