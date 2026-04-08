/**
 * ToolDisplay — compact tool call rendering (claude-code style).
 * Single line per tool: `● tool_name summary` → `✓ tool_name summary (Xms)`
 */

import React from "react";
import { Box, Text } from "ink";
import { DiffView } from "./DiffView.js";
import type {
  PresentedToolEvent,
  PresentedToolGroup,
  PresentedFileEdit,
  PresentedCommandResult,
  PresentedValidationResult,
  PresentedDiagnosticList,
} from "../transcript-presenter.js";

function cleanError(error: string): string {
  let clean = error.split("\n")[0] ?? error;
  clean = clean.replace(/\[Recovery hint\].*$/, "").trim();
  // B6: Simplify path validation errors
  if (clean.includes("Invalid path") || clean.includes("Path must stay")) {
    return "path outside repo";
  }
  if (clean.includes("All steps failed")) {
    return "all steps failed";
  }
  if (clean.length > 70) {
    const cut = clean.lastIndexOf(" ", 70);
    clean = clean.slice(0, cut > 40 ? cut : 70) + "…";
  }
  return clean;
}

export function ToolDisplay({ event }: { event: PresentedToolEvent }): React.ReactElement {
  if (event.status === "running") {
    // Color file paths in summary
    const hasPath = event.summary.includes("/") || event.summary.includes(".");
    return (
      <Text>
        <Text dimColor>  </Text>
        <Text color="cyan">● </Text>
        <Text bold>{event.name}</Text>
        {hasPath
          ? <Text color="cyan"> {event.summary}</Text>
          : <Text dimColor> {event.summary}</Text>
        }
      </Text>
    );
  }

  const icon = event.status === "success" ? "✓" : "✗";
  const iconColor = event.status === "success" ? "green" : "red";
  const duration = event.durationMs !== undefined ? ` (${event.durationMs}ms)` : "";
  const isError = event.status === "error";

  return (
    <Box flexDirection="column">
      <Text>
        <Text dimColor>  </Text>
        <Text color={iconColor}>{icon} </Text>
        <Text dimColor={!isError}>{event.name}</Text>
        <Text dimColor>{duration}{event.error ? `: ${cleanError(event.error)}` : ""}</Text>
      </Text>
      {event.preview && (
        <Text>
          <Text dimColor>    </Text>
          <Text color="cyan">{event.preview}</Text>
        </Text>
      )}
    </Box>
  );
}

export function FileEditDisplay({ data }: { data: PresentedFileEdit }): React.ReactElement {
  const { fileEdit, summary } = data;
  return (
    <Box flexDirection="column" marginTop={1}>
      <Text>
        <Text dimColor>    </Text>
        <Text bold>{fileEdit.path}</Text>
        <Text color="green"> +{fileEdit.additions}</Text>
        <Text color="red"> -{fileEdit.deletions}</Text>
      </Text>
      <Text dimColor>      {summary}</Text>
      <DiffView fileEdit={fileEdit} />
    </Box>
  );
}

export function ToolGroupDisplay({ event }: { event: PresentedToolGroup }): React.ReactElement {
  if (event.status === "running") {
    const summaryStr = event.summaries.slice(0, 3).join(", ");
    const overflow = event.count > 3 ? ` +${event.count - 3} more` : "";
    return (
      <Text>
        <Text dimColor>  </Text>
        <Text color="cyan">● </Text>
        <Text bold>{event.name}</Text>
        <Text dimColor> {event.count} calls ({summaryStr}{overflow})</Text>
      </Text>
    );
  }

  const icon = event.status === "success" ? "✓" : "✗";
  const iconColor = event.status === "success" ? "green" : "red";
  const duration = event.totalDurationMs !== undefined ? ` (${event.totalDurationMs}ms)` : "";
  const files = event.summaries.slice(0, 3).map((s) => s.split("/").pop() ?? s);
  const fileStr = files.length > 0 ? ` ${files.join(", ")}${event.count > 3 ? ", …" : ""}` : "";

  return (
    <Text>
      <Text dimColor>  </Text>
      <Text color={iconColor}>{icon} </Text>
      <Text dimColor>{event.name}</Text>
      <Text dimColor> ×{event.count}</Text>
      <Text color="cyan">{fileStr}</Text>
      <Text dimColor>{duration}</Text>
    </Text>
  );
}

export function CommandResultDisplay({ data }: { data: PresentedCommandResult }): React.ReactElement {
  const tone = data.status === "success"
    ? "green"
    : data.status === "warning"
      ? "yellow"
      : "red";

  return (
    <Box flexDirection="column" marginTop={1}>
      <Text>
        <Text dimColor>    </Text>
        <Text color="cyan">command</Text>
        <Text> </Text>
        <Text bold>{data.command}</Text>
      </Text>
      <Text dimColor>      {data.cwd === "." ? "cwd ." : `cwd ${data.cwd}`} · <Text color={tone}>{data.statusLine}</Text></Text>
      {data.stdoutPreview && (
        <Text>
          <Text dimColor>      stdout </Text>
          <Text>{data.stdoutPreview.replace(/\n/g, " ↵ ")}</Text>
          {data.stdoutTruncated ? <Text dimColor> …</Text> : null}
        </Text>
      )}
      {data.stderrPreview && (
        <Text>
          <Text dimColor>      stderr </Text>
          <Text color={data.status === "warning" ? "yellow" : "red"}>{data.stderrPreview.replace(/\n/g, " ↵ ")}</Text>
          {data.stderrTruncated ? <Text dimColor> …</Text> : null}
        </Text>
      )}
    </Box>
  );
}

export function ValidationResultDisplay({ data }: { data: PresentedValidationResult }): React.ReactElement {
  return (
    <Box flexDirection="column" marginTop={1}>
      <Text>
        <Text dimColor>    </Text>
        <Text color={data.passed ? "green" : "yellow"}>validation</Text>
        <Text> </Text>
        <Text>{data.summary}</Text>
      </Text>
      {data.testSummaryLine ? <Text dimColor>      {data.testSummaryLine}</Text> : null}
      {data.testOutputPreview ? (
        <Text>
          <Text dimColor>      tests </Text>
          <Text>{data.testOutputPreview.replace(/\n/g, " ↵ ")}</Text>
        </Text>
      ) : null}
    </Box>
  );
}

export function DiagnosticListDisplay({ data }: { data: PresentedDiagnosticList }): React.ReactElement {
  return (
    <Box flexDirection="column" marginTop={1}>
      <Text>
        <Text dimColor>    </Text>
        <Text color="yellow">{data.title}</Text>
      </Text>
      {data.diagnostics.map((diagnostic, index) => (
        <Text key={`${data.toolId}-diag-${index}`}>
          <Text dimColor>      • </Text>
          <Text>{diagnostic}</Text>
        </Text>
      ))}
      {data.hiddenCount > 0 ? <Text dimColor>      ... +{data.hiddenCount} more diagnostics</Text> : null}
    </Box>
  );
}
