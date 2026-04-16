/**
 * DiffView — renders structured file diffs shared with the plain CLI formatter.
 */

import React from "react";
import { Box, Text, useStdout } from "ink";
import type { ToolFileChangePreview, ToolFileChangeLine } from "@devagent/runtime";
import {
  buildHighlightedFileEdit,
  getPresentedDiffGutterWidth,
  type HighlightedDiffLine,
  takeVisibleHighlightedDiffItems,
} from "../file-edit-presentation.js";

interface DiffViewProps {
  readonly fileEdit: ToolFileChangePreview;
  readonly maxLines?: number;
}

const DEFAULT_MAX_LINES = 10;

export function DiffView({
  fileEdit,
  maxLines = DEFAULT_MAX_LINES,
}: DiffViewProps): React.ReactElement | null {
  const { stdout } = useStdout();
  const bodyWidth = Math.max(24, (stdout?.columns ?? 80) - 18);
  const highlighted = React.useMemo(
    () => buildHighlightedFileEdit(fileEdit, { bodyWidth }),
    [fileEdit, bodyWidth],
  );

  if (highlighted.hunks.length === 0) {
    return null;
  }

  const visible = takeVisibleHighlightedDiffItems(highlighted.hunks, maxLines);
  const gutterWidth = getPresentedDiffGutterWidth(highlighted.hunks);

  return (
    <Box flexDirection="column">
      {visible.items.map((item) => (
        item.type === "separator"
          ? (
            <Text key={item.key}>
              <Text dimColor>      ...</Text>
            </Text>
          )
          : (
            <StructuredDiffLineView
              key={`${item.line.type}-${item.line.oldLine ?? "none"}-${item.line.newLine ?? "none"}-${item.line.text}`}
              line={item.line}
              gutterWidth={gutterWidth}
            />
          )
      ))}
      {(visible.hiddenLines > 0 || highlighted.truncated) && (
        <Text dimColor>
          {"      ..."}
          {visible.hiddenLines > 0 ? ` +${visible.hiddenLines} more diff lines` : " diff truncated"}
        </Text>
      )}
    </Box>
  );
}

function StructuredDiffLineView(
  {
    line,
    gutterWidth,
  }: {
    readonly line: HighlightedDiffLine;
    readonly gutterWidth: number;
  },
): React.ReactElement {
  const oldLine = formatLineNumber(line.oldLine, gutterWidth);
  const newLine = formatLineNumber(line.newLine, gutterWidth);
  const marker = line.type === "add" ? "+" : line.type === "delete" ? "-" : " ";
  const codeColor = line.type === "add" ? "green" : line.type === "delete" ? "red" : undefined;
  const content = line.text.length > 0 ? line.renderedText : "<blank>";
  const useFallbackColor = !line.syntaxHighlighted;

  return (
    <Text>
      <Text dimColor>      </Text>
      <Text dimColor>{oldLine}</Text>
      <Text dimColor> </Text>
      <Text dimColor>{newLine}</Text>
      <Text color={codeColor}> {marker} </Text>
      {useFallbackColor ? (
        <Text color={codeColor} dimColor={line.type === "context" || line.text.length === 0}>{content}</Text>
      ) : (
        <Text>{content}</Text>
      )}
    </Text>
  );
}

function formatLineNumber(line: number | null, width: number): string {
  return line === null ? "".padStart(width, " ") : String(line).padStart(width, " ");
}
