/**
 * LogEntryView — shared log entry renderer for Static items.
 *
 * Used by both App.tsx and SingleShotApp.tsx.
 */

import { Box, Text } from "ink";
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
import { cleanTime } from "./shared.js";

export const LogEntryView = React.memo(function LogEntryView({ entry }: { entry: LogEntry }): React.ReactElement | null {
  switch (entry.part.kind) {
    case "tool":
      return <ToolDisplay event={entry.part.event} />;
    case "tool-group":
      return <ToolGroupDisplay event={entry.part.event} />;
    case "file-edit":
      return <FileEditDisplay data={entry.part.data} />;
    case "file-edit-overflow":
      return <Text dimColor>      ... +{entry.part.data.hiddenCount} more files</Text>;
    case "command-result":
      return <CommandResultDisplay data={entry.part.data} />;
    case "validation-result":
      return <ValidationResultDisplay data={entry.part.data} />;
    case "diagnostic-list":
      return <DiagnosticListDisplay data={entry.part.data} />;
    case "reasoning":
      return (
        <Box borderLeft borderColor="gray" paddingLeft={1}>
          <Text dimColor>ℹ {entry.part.data.text}</Text>
        </Box>
      );
    case "plan":
      return <PlanView steps={entry.part.data as PlanStep[]} />;
    case "error": {
      const err = entry.part.data;
      return <ErrorView message={err.message} code={err.code} />;
    }
    case "status": {
      return (
        <TranscriptCard
          title={entry.part.data.title}
          color={toneToColor(entry.part.data.tone)}
          lines={entry.part.data.lines}
        />
      );
    }
    case "progress": {
      return (
        <TranscriptCard
          title={entry.part.data.title}
          color="cyan"
          lines={[entry.part.data.detail ?? "Working…"]}
        />
      );
    }
    case "approval":
      return (
        <TranscriptCard
          title="approval"
          color="yellow"
          lines={[`Awaiting approval for ${entry.part.data.toolName}`, entry.part.data.details]}
        />
      );
    case "final-output":
      return <FinalOutput text={entry.part.data.text} />;
    case "turn-summary":
      return null;
    case "user":
      return <TranscriptCard title="you" color="cyan" lines={[`> ${entry.part.data.text}`]} boldBody marginTop={1} />;
    case "info": {
      const lines = entry.part.data.lines;
      const joined = lines.join("\n");
      if (joined.includes("completed")) {
        const scoreMatch = joined.match(/(?:score )?(\d+\.\d+),?\s*(?:partial|complete)/);
        if (scoreMatch) {
          const score = parseFloat(scoreMatch[1]!);
          const scoreColor = score >= 0.8 ? "green" : score >= 0.5 ? "yellow" : "red";
          const matchStart = joined.indexOf(scoreMatch[0]);
          const beforeScore = joined.slice(0, matchStart);
          const afterScore = joined.slice(matchStart + scoreMatch[0].length);
          return (
            <Text>
              <Text dimColor>{cleanTime(beforeScore)}</Text>
              <Text color={scoreColor} bold>{score.toFixed(2)}</Text>
              <Text dimColor>{afterScore}</Text>
            </Text>
          );
        }
        return <Text dimColor>{cleanTime(joined)}</Text>;
      }
      return <TranscriptCard title={entry.part.data.title} color="gray" lines={lines} />;
    }
    default:
      return null;
  }
});

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
  return (
    <Box flexDirection="column" marginTop={marginTop}>
      <Text>
        <Text dimColor>  ╭─ </Text>
        <Text color={color}>{title}</Text>
      </Text>
      {lines.map((line, index) => (
        <Text key={`${title}-${index}`}>
          <Text dimColor>  │ </Text>
          <Text bold={boldBody}>{line}</Text>
        </Text>
      ))}
      <Text dimColor>  ╰─</Text>
    </Box>
  );
}
