/**
 * TurnSummary — shown after each query completes.
 */

import React from "react";
import { Box, Text } from "ink";
import { formatDuration } from "@devagent/runtime";

export interface TurnSummaryProps {
  readonly iterations: number;
  readonly toolCalls: number;
  readonly cost: number;
  readonly elapsedMs: number;
}

export function TurnSummary({ iterations, toolCalls, cost, elapsedMs }: TurnSummaryProps): React.ReactElement {
  const parts: string[] = [];
  if (iterations > 0) parts.push(`${iterations} iterations`);
  if (toolCalls > 0) parts.push(`${toolCalls} tool calls`);
  if (cost > 0) parts.push(`$${cost.toFixed(4)}`);
  parts.push(formatDuration(elapsedMs));

  return (
    <Box marginTop={1}>
      <Text color="green">✓ </Text>
      <Text bold>Done</Text>
      <Text dimColor> ({parts.join(", ")})</Text>
    </Box>
  );
}
