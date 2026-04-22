/**
 * LogEntryView — shared log entry renderer for Static items.
 *
 * Used by both App.tsx and SingleShotApp.tsx.
 */

import { Box, Text, useStdout } from "ink";
import React from "react";

import { FinalOutput } from "./FinalOutput.js";
import { ErrorView } from "./MessageView.js";
import { PlanView, type PlanStep } from "./PlanView.js";
import type { LogEntry } from "./shared.js";
import {
  CommandResultDisplay,
  DiagnosticListDisplay,
  FileEditDisplay,
  ToolDisplay,
  ToolGroupDisplay,
  ValidationResultDisplay,
} from "./ToolDisplay.js";
import type { PresentedStatus } from "../transcript-presenter.js";
import { cleanTime, framedBodyWidth, wrapAnsiTextByWidth } from "./shared.js";
export const LogEntryView = React.memo(function LogEntryView({ entry }: { entry: LogEntry }): React.ReactElement | null {
  const direct = renderDirectEntry(entry);
  if (direct !== undefined) return direct;
  return renderCardEntry(entry);
});

function renderDirectEntry(entry: LogEntry): React.ReactElement | null | undefined {
  return renderToolEntry(entry) ?? renderMessageEntry(entry);
}

function renderToolEntry(entry: LogEntry): React.ReactElement | undefined {
  if (entry.part.kind === "tool") return <ToolDisplay event={entry.part.event} />;
  if (entry.part.kind === "tool-group") return <ToolGroupDisplay event={entry.part.event} />;
  if (entry.part.kind === "file-edit") return <FileEditDisplay data={entry.part.data} />;
  if (entry.part.kind === "file-edit-overflow") {
    return <Text dimColor>      ... +{entry.part.data.hiddenCount} more files</Text>;
  }
  if (entry.part.kind === "command-result") return <CommandResultDisplay data={entry.part.data} />;
  if (entry.part.kind === "validation-result") return <ValidationResultDisplay data={entry.part.data} />;
  if (entry.part.kind === "diagnostic-list") return <DiagnosticListDisplay data={entry.part.data} />;
  return undefined;
}

function renderMessageEntry(entry: LogEntry): React.ReactElement | null | undefined {
  if (entry.part.kind === "reasoning") {
    return (
      <Box borderLeft borderColor="gray" paddingLeft={1}>
        <Text dimColor>ℹ {entry.part.data.text}</Text>
      </Box>
    );
  }
  if (entry.part.kind === "plan") return <PlanView steps={entry.part.data as PlanStep[]} />;
  if (entry.part.kind === "error") return <ErrorView message={entry.part.data.message} code={entry.part.data.code} />;
  if (entry.part.kind === "final-output") return <FinalOutput text={entry.part.data.text} />;
  if (entry.part.kind === "turn-summary") return null;
  if (entry.part.kind === "info") return <InfoEntry title={entry.part.data.title} lines={entry.part.data.lines} />;
  return undefined;
}

function renderCardEntry(entry: LogEntry): React.ReactElement | null {
  if (entry.part.kind === "status") {
    return <TranscriptCard title={entry.part.data.title} color={toneToColor(entry.part.data.tone)} lines={entry.part.data.lines} />;
  }
  if (entry.part.kind === "progress") {
    return <TranscriptCard title={entry.part.data.title} color="cyan" lines={[entry.part.data.detail ?? "Working…"]} />;
  }
  if (entry.part.kind === "approval") {
    return <TranscriptCard title="approval" color="yellow" lines={[`Awaiting approval for ${entry.part.data.toolName}`, entry.part.data.details]} />;
  }
  if (entry.part.kind === "user") {
    return <TranscriptCard title="you" color="cyan" lines={[`> ${entry.part.data.text}`]} boldBody marginTop={1} />;
  }
  return null;
}

function InfoEntry(
  { title, lines }: { readonly title: string; readonly lines: ReadonlyArray<string> },
): React.ReactElement {
  const joined = lines.join("\n");
  if (!joined.includes("completed")) {
    return <TranscriptCard title={title} color="gray" lines={lines} />;
  }
  const score = extractScoreDisplay(joined);
  if (!score) {
    return <Text dimColor>{cleanTime(joined)}</Text>;
  }
  return (
    <Text>
      <Text dimColor>{cleanTime(score.before)}</Text>
      <Text color={score.color} bold>{score.value.toFixed(2)}</Text>
      <Text dimColor>{score.after}</Text>
    </Text>
  );
}

function extractScoreDisplay(joined: string): {
  readonly before: string;
  readonly after: string;
  readonly value: number;
  readonly color: "green" | "yellow" | "red";
} | null {
  const scoreMatch = joined.match(/(?:score )?(\d+\.\d+),?\s*(?:partial|complete)/);
  if (!scoreMatch) return null;
  const value = parseFloat(scoreMatch[1]!);
  const matchStart = joined.indexOf(scoreMatch[0]);
  return {
    before: joined.slice(0, matchStart),
    after: joined.slice(matchStart + scoreMatch[0].length),
    value,
    color: value >= 0.8 ? "green" : value >= 0.5 ? "yellow" : "red",
  };
}

function toneToColor(tone: PresentedStatus["tone"]): "cyan" | "green" | "yellow" | "red" | "gray" {
  switch (tone) {
    case "success":
      return "green";
    case "warning":
      return "yellow";
    case "error":
      return "red";
    case "info":
      return "cyan";
    case undefined:
      return "gray";
    default:
      return "gray";
  }
}

function TranscriptCard(
  {
    title,
    color,
    lines,
    boldBody = false,
    marginTop = 0,
  }: {
    readonly title: string;
    readonly color: "cyan" | "green" | "yellow" | "red" | "gray";
    readonly lines: ReadonlyArray<string>;
    readonly boldBody?: boolean;
    readonly marginTop?: number;
  },
): React.ReactElement {
  const { stdout } = useStdout();
  const bodyWidth = framedBodyWidth(stdout.columns);

  return (
    <Box flexDirection="column" marginTop={marginTop}>
      <Text>
        <Text dimColor>  ╭─ </Text>
        <Text color={color}>{title}</Text>
      </Text>
      {lines
        .flatMap((line) => wrapAnsiTextByWidth(line, bodyWidth))
        .map((line, index) => (
          <Text key={`${title}-${index}`}>
            <Text dimColor>  │ </Text>
            <Text bold={boldBody}>{line}</Text>
          </Text>
        ))}
      <Text dimColor>  ╰─</Text>
    </Box>
  );
}
