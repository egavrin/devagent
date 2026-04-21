import { SafetyMode } from "@devagent/runtime";

import type { DevAgentConfig } from "@devagent/runtime";

export function getSafetyPreset(
  mode: SafetyMode,
): Pick<DevAgentConfig["approval"], "mode" | "approvalPolicy" | "sandboxMode" | "networkAccess"> {
  switch (mode) {
    case SafetyMode.AUTOPILOT:
      return {
        mode,
        approvalPolicy: "never",
        sandboxMode: "danger-full-access",
        networkAccess: "on",
      };
    case SafetyMode.DEFAULT:
    default:
      return {
        mode,
        approvalPolicy: "on-request",
        sandboxMode: "workspace-write",
        networkAccess: "off",
      };
  }
}

export function withInteractiveSafetyMode(
  config: DevAgentConfig,
  mode: SafetyMode,
): DevAgentConfig {
  const safety = getSafetyPreset(mode);
  return {
    ...config,
    approval: {
      ...config.approval,
      ...safety,
    },
  };
}

export function getInteractiveSafetyMode(config: DevAgentConfig): SafetyMode {
  return config.approval.mode === SafetyMode.AUTOPILOT
    ? SafetyMode.AUTOPILOT
    : SafetyMode.DEFAULT;
}
