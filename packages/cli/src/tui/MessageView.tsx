import { Box, Text } from "ink";
import React from "react";

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
