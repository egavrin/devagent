import React from "react";
import { Box, Text } from "ink";
import { getApprovalModeColor } from "./shared.js";

export interface StatusBarProps {
  readonly model: string;
  readonly cost: number;
  readonly inputTokens: number;
  readonly maxContextTokens: number;
  readonly iteration: number;
  readonly maxIterations: number;
  readonly approvalMode: string;
  readonly cwd?: string;
  readonly running?: boolean;
  readonly hasApproval?: boolean;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${Math.round(n / 1_000)}k`;
  return String(n);
}

export function StatusBar(props: StatusBarProps): React.ReactElement {
  const { model, cost, inputTokens, maxContextTokens, iteration, maxIterations, approvalMode, cwd, running, hasApproval } = props;

  const shortModel = model.length > 20 ? model.slice(0, 20) : model;
  const pct = maxContextTokens > 0 ? Math.round((inputTokens / maxContextTokens) * 100) : 0;
  const pctColor = pct > 80 ? "red" : pct > 60 ? "yellow" : "gray";
  const modeColor = getApprovalModeColor(approvalMode);

  const iterLabel = maxIterations > 0
    ? `iter ${iteration}/${maxIterations}`
    : iteration > 0 ? `iter ${iteration}` : "";

  // Shortcuts section — context-aware
  let shortcuts = "";
  if (hasApproval) {
    shortcuts = "[y]es [n]o [a]lways";
  } else if (running) {
    shortcuts = "Ctrl+C cancel";
  } else {
    shortcuts = "Shift+Tab mode │ Ctrl+K cmds │ Ctrl+C exit";
  }

  const dirName = cwd ? (cwd.split("/").pop() ?? cwd) : "";

  return (
    <Box>
      <Text dimColor>{shortcuts} │ </Text>
      <Text bold>{shortModel}</Text>
      {cost > 0 && <><Text color="gray"> │ </Text><Text color="green">${cost.toFixed(4)}</Text></>}
      {maxContextTokens > 0 && <><Text color="gray"> │ </Text><Text color={pctColor}>{formatTokens(inputTokens)}/{formatTokens(maxContextTokens)} ({pct}%)</Text></>}
      {iterLabel && <><Text color="gray"> │ </Text><Text color="gray">{iterLabel}</Text></>}
      <Text color="gray"> │ </Text><Text color={modeColor} bold>{approvalMode}</Text>
      {dirName && <><Text color="gray"> │ </Text><Text dimColor>{dirName}</Text></>}
    </Box>
  );
}
