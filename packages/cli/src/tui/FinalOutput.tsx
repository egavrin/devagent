/**
 * FinalOutput — renders the agent's final response inside the transcript card system.
 */

import { Box, Text } from "ink";
import React from "react";

import { renderMarkdown } from "../markdown-render.js";

interface FinalOutputProps {
  readonly text: string;
}

export function FinalOutput({ text }: FinalOutputProps): React.ReactElement {
  const rendered = renderMarkdown(text);
  const lines = rendered.split("\n");

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text>
        <Text dimColor>  ╭─ </Text>
        <Text color="green">devagent</Text>
      </Text>
      {lines.map((line, index) => (
        <Text key={`devagent-line-${index}`}>
          <Text dimColor>  │ </Text>
          <Text>{line}</Text>
        </Text>
      ))}
      <Text dimColor>  ╰─</Text>
    </Box>
  );
}
