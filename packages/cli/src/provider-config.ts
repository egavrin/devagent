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
  };
  const registryCaps = lookupModelCapabilities(config.model, config.provider);
  const resolvedReasoningEffort = agentType
    ? config.agentReasoningOverrides?.[agentType] ?? reasoningEffort
    : reasoningEffort;
  const shouldUseDevagentApiMainDefault =
    config.provider === "devagent-api" &&
    agentType === undefined &&
    resolvedReasoningEffort === undefined &&
    baseProviderConfig.reasoningEffort === undefined;

  return {
    ...baseProviderConfig,
    model: config.model,
    ...(resolvedReasoningEffort
      ? { reasoningEffort: resolvedReasoningEffort }
      : shouldUseDevagentApiMainDefault
        ? { reasoningEffort: "high" }
        : {}),
    ...(!baseProviderConfig.capabilities && registryCaps
      ? { capabilities: registryCaps as ModelCapabilities }
      : {}),
  };
}
