import type { ReasoningEffort } from "./types.js";
import { AgentType } from "./types.js";

const OPENAI_FAMILY_SUBAGENT_MODEL_DEFAULTS: Partial<Record<AgentType, string>> = {
  [AgentType.EXPLORE]: "gpt-5.4-mini",
  [AgentType.REVIEWER]: "gpt-5.4",
  [AgentType.ARCHITECT]: "gpt-5.4",
};

const OPENAI_FAMILY_SUBAGENT_REASONING_DEFAULTS: Partial<Record<AgentType, ReasoningEffort>> = {
  [AgentType.EXPLORE]: "low",
  [AgentType.REVIEWER]: "high",
  [AgentType.ARCHITECT]: "high",
};

const DEVAGENT_API_SUBAGENT_REASONING_DEFAULTS: Partial<Record<AgentType, ReasoningEffort>> = {
  [AgentType.EXPLORE]: "low",
  [AgentType.REVIEWER]: "high",
  [AgentType.ARCHITECT]: "high",
};

export function getDefaultSubagentProfiles(
  provider: string,
): {
  readonly agentModelOverrides?: Partial<Record<AgentType, string>>;
  readonly agentReasoningOverrides?: Partial<Record<AgentType, ReasoningEffort>>;
} {
  if (provider === "devagent-api") {
    return { agentReasoningOverrides: DEVAGENT_API_SUBAGENT_REASONING_DEFAULTS };
  }
  if (provider !== "openai" && provider !== "chatgpt") return {};

  return {
    agentModelOverrides: OPENAI_FAMILY_SUBAGENT_MODEL_DEFAULTS,
    agentReasoningOverrides: OPENAI_FAMILY_SUBAGENT_REASONING_DEFAULTS,
  };
}
