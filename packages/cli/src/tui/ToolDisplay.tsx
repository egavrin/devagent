/**
 * ToolDisplay — compact tool call rendering (claude-code style).
 * Single line per tool: `● tool_name summary` → `✓ tool_name summary (Xms)`
 */

import React from "react";
import { Box, Text } from "ink";
import { DiffView } from "./DiffView.js";

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

export interface ToolEvent {
  readonly id: string;
  readonly name: string;
  readonly summary: string;
  readonly iteration: number;
  readonly maxIterations: number;
  readonly status: "running" | "success" | "error";
  readonly durationMs?: number;
  readonly error?: string;
  readonly preview?: string;
  readonly diff?: string;
}

export interface ToolGroupEvent {
  readonly name: string;
  readonly count: number;
  readonly summaries: ReadonlyArray<string>;
  readonly iteration: number;
  readonly maxIterations: number;
  readonly status: "running" | "success" | "error";
  readonly totalDurationMs?: number;
}

export function ToolDisplay({ event }: { event: ToolEvent }): React.ReactElement {
  if (event.status === "running") {
    // Color file paths in summary
    const hasPath = event.summary.includes("/") || event.summary.includes(".");
    return (
      <Text>
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
        <Text color={iconColor}>{icon} </Text>
        <Text dimColor={isError}>{event.name}</Text>
        <Text dimColor>{duration}{event.error ? `: ${cleanError(event.error)}` : ""}</Text>
      </Text>
      {event.preview && (
        <Text>
          <Text dimColor>  ⎿ </Text>
          <Text color="cyan">{event.preview}</Text>
        </Text>
      )}
      {event.diff && <DiffView diff={event.diff} />}
    </Box>
  );
}

export function ToolGroupDisplay({ event }: { event: ToolGroupEvent }): React.ReactElement {
  if (event.status === "running") {
    const summaryStr = event.summaries.slice(0, 3).join(", ");
    const overflow = event.count > 3 ? ` +${event.count - 3} more` : "";
    return (
      <Text>
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
      <Text color={iconColor}>{icon} </Text>
      <Text>{event.name}</Text>
      <Text dimColor> ×{event.count}</Text>
      <Text color="cyan">{fileStr}</Text>
      <Text dimColor>{duration}</Text>
    </Text>
  );
}
