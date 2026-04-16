import React from "react";
import { Box, Text } from "ink";

function ThinkingDuration({ durationMs }: { durationMs: number }): React.ReactElement | null {
  if (durationMs < 500) return null;
  return (
    <Text>
      <Text dimColor>  · </Text>
      <Text dimColor>thought for {(durationMs / 1000).toFixed(1)}s</Text>
    </Text>
  );
}

export function ErrorView({ message, code }: { message: string; code: string }): React.ReactElement {
  return (
    <Box flexDirection="column">
      <Text>
        <Text dimColor>  ╭─ </Text>
        <Text color="red">error</Text>
      </Text>
      <Text>
        <Text dimColor>  │ </Text>
        <Text color="red">{message}</Text>
        {code && code !== "LSP_INFO" && <Text dimColor> ({code})</Text>}
      </Text>
      <Text dimColor>  ╰─</Text>
    </Box>
  );
}
