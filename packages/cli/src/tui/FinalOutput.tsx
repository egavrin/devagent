/**
 * FinalOutput — renders the agent's final response with markdown formatting.
 * No border — just renders the markdown directly to avoid overflow issues.
 */

import React from "react";
import { Box, Text } from "ink";
import { renderMarkdown } from "../markdown-render.js";

export interface FinalOutputProps {
  readonly text: string;
}

export function FinalOutput({ text }: FinalOutputProps): React.ReactElement {
  const rendered = renderMarkdown(text);

  return (
    <Box flexDirection="column" marginTop={1} borderLeft borderColor="green" paddingLeft={1}>
      <Text>{rendered}</Text>
    </Box>
  );
}
