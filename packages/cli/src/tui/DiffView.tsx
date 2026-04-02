/**
 * DiffView — renders a colored unified diff.
 * Green for additions, red for deletions, cyan for hunk headers.
 */

import React from "react";
import { Box, Text } from "ink";

export interface DiffViewProps {
  /** Raw diff text (unified format). */
  readonly diff: string;
  /** Max lines to show. */
  readonly maxLines?: number;
}

export function DiffView({ diff, maxLines = 10 }: DiffViewProps): React.ReactElement | null {
  if (!diff.trim()) return null;

  const lines = diff.split("\n");
  const visible = lines.slice(0, maxLines);
  const overflow = lines.length > maxLines ? lines.length - maxLines : 0;

  return (
    <Box flexDirection="column" marginLeft={4}>
      {visible.map((line, i) => (
        <DiffLine key={i} line={line} />
      ))}
      {overflow > 0 && (
        <Text dimColor>  ... +{overflow} more lines</Text>
      )}
    </Box>
  );
}

function DiffLine({ line }: { line: string }): React.ReactElement {
  if (line.startsWith("+") && !line.startsWith("+++")) {
    return <Text color="green">{line}</Text>;
  }
  if (line.startsWith("-") && !line.startsWith("---")) {
    return <Text color="red">{line}</Text>;
  }
  if (line.startsWith("@@")) {
    return <Text color="cyan">{line}</Text>;
  }
  if (line.startsWith("diff ") || line.startsWith("index ") || line.startsWith("---") || line.startsWith("+++")) {
    return <Text bold>{line}</Text>;
  }
  return <Text dimColor>{line}</Text>;
}
