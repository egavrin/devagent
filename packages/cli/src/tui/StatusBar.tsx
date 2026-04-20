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
  const pctColor = pct > 80 ? "red" : pct > 60 ? "yellow" : "gray";

  const iterLabel = maxIterations > 0
    ? `iter ${iteration}/${maxIterations}`
    : "";

  // Shortcuts section — context-aware
  let shortcuts = "";
  if (hasApproval) {
    shortcuts = "[y]once [n]deny [s]ession";
  } else if (running) {
    shortcuts = "Ctrl+C cancel";
  } else {
    shortcuts = "Shift+Tab safety │ Ctrl+K cmds │ Ctrl+C exit";
  }

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
