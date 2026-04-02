import React from "react";
import { Box, Text } from "ink";

export function ThinkingDuration({ durationMs }: { durationMs: number }): React.ReactElement | null {
  if (durationMs < 500) return null;
  return <Text dimColor>  ℹ Thought for {(durationMs / 1000).toFixed(1)}s</Text>;
}

export function ErrorView({ message, code }: { message: string; code: string }): React.ReactElement {
  return (
    <Box borderLeft borderColor="red" paddingLeft={1}>
      <Text color="red">✗ {message}</Text>
      {code && code !== "LSP_INFO" && <Text dimColor> ({code})</Text>}
    </Box>
  );
}
