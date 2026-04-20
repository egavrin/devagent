/**
 * PlanView — renders the current plan with status indicators.
 * Uses left border only (opencode BlockTool pattern) to avoid full-width stretching.
 */

import { Box, Text } from "ink";
import React from "react";

export interface PlanStep {
  readonly description: string;
  readonly status: string;
}

export function PlanView({ steps }: { steps: ReadonlyArray<PlanStep> }): React.ReactElement {
  return (
    <Box flexDirection="column" paddingLeft={1} marginTop={1} borderLeft borderColor="yellow">
      <Text bold dimColor>── Plan ──</Text>
      {steps.map((step, i) => {
        const icon = step.status === "completed" ? "[x]"
          : step.status === "in_progress" ? "[>]"
          : "[ ]";
        const iconColor = step.status === "completed" ? "green"
          : step.status === "in_progress" ? "yellow"
          : "gray";
        const textDim = step.status !== "in_progress";

        return (
          <Box key={i}>
            <Text color={iconColor}>  {icon} </Text>
            <Text dimColor={textDim}>{step.description}</Text>
          </Box>
        );
      })}
    </Box>
  );
}
