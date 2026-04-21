import { Box, Text } from "ink";
import React from "react";

import { tokenProgressBar } from "./shared.js";

export interface StatusBarProps {
  readonly model: string;
  readonly cost: number;
  readonly inputTokens: number;
  readonly maxContextTokens: number;
  readonly iteration: number;
  readonly maxIterations: number;
  readonly cwd?: string;
  readonly running?: boolean;
  readonly hasApproval?: boolean;
}
export function StatusBar(props: StatusBarProps): React.ReactElement {
  const { model, cost, inputTokens, maxContextTokens, iteration, maxIterations, cwd, running, hasApproval } = props;

  const shortModel = model.length > 20 ? model.slice(0, 20) : model;
  const pct = maxContextTokens > 0 ? Math.round((inputTokens / maxContextTokens) * 100) : 0;
  const pctColor = getTokenColor(pct);
  const iterLabel = getIterationLabel(iteration, maxIterations);
  const shortcuts = getShortcutLabel(Boolean(running), Boolean(hasApproval));

  const dirName = cwd ? (cwd.split("/").pop() ?? cwd) : "";

  return (
    <Box>
      <Text dimColor>{shortcuts} │ </Text>
      <Text bold>{shortModel}</Text>
      {cost > 0 && <><Text color="gray"> │ </Text><Text color="green">${cost.toFixed(4)}</Text></>}
      {maxContextTokens > 0 && <><Text color="gray"> │ </Text><Text color={pctColor}>{tokenProgressBar(inputTokens, maxContextTokens)}</Text></>}
      {iterLabel && <><Text color="gray"> │ </Text><Text color="gray">{iterLabel}</Text></>}
      {dirName && <><Text color="gray"> │ </Text><Text dimColor>{dirName}</Text></>}
    </Box>
  );
}

function getTokenColor(pct: number): "red" | "yellow" | "gray" {
  if (pct > 80) return "red";
  if (pct > 60) return "yellow";
  return "gray";
}

function getIterationLabel(iteration: number, maxIterations: number): string {
  return maxIterations > 0 ? `iter ${iteration}/${maxIterations}` : "";
}

function getShortcutLabel(running: boolean, hasApproval: boolean): string {
  if (hasApproval) return "[y]once [n]deny [s]ession";
  if (running) return "Ctrl+C cancel";
  return "Shift+Tab safety │ Ctrl+K cmds │ Ctrl+C exit";
}
