import { Box, Text, useStdout } from "ink";
import React from "react";

import { framedBodyWidth, wrapAnsiTextByWidth } from "./shared.js";

export function ErrorView({ message, code }: { message: string; code: string }): React.ReactElement {
  const { stdout } = useStdout();
  const body = code && code !== "LSP_INFO" ? `${message} (${code})` : message;
  const rows = wrapAnsiTextByWidth(body, framedBodyWidth(stdout.columns));

  return (
    <Box flexDirection="column">
      <Text>
        <Text dimColor>  ╭─ </Text>
        <Text color="red">error</Text>
      </Text>
      {rows.map((row, index) => (
        <Text key={`error-line-${index}`}>
          <Text dimColor>  │ </Text>
          <Text color="red">{row}</Text>
        </Text>
      ))}
      <Text dimColor>  ╰─</Text>
    </Box>
  );
}
