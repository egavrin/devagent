import { AgentType } from "../core/index.js";

export function parseAgentType(
  value: AgentType | string | null | undefined,
): AgentType | undefined {
  if (!value) return undefined;

  switch (String(value).toLowerCase()) {
    case AgentType.GENERAL:
      return AgentType.GENERAL;
    case AgentType.REVIEWER:
      return AgentType.REVIEWER;
    case AgentType.ARCHITECT:
      return AgentType.ARCHITECT;
    case AgentType.EXPLORE:
      return AgentType.EXPLORE;
    default:
      return undefined;
  }
}
