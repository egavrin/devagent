/**
 * LogEntryView — shared log entry renderer for Static items.
 *
 * Used by both App.tsx and SingleShotApp.tsx.
 */

import React from "react";
import { Box, Text } from "ink";
import { ToolDisplay, ToolGroupDisplay, type ToolEvent, type ToolGroupEvent } from "./ToolDisplay.js";
import { PlanView, type PlanStep } from "./PlanView.js";
import { ThinkingDuration, ErrorView } from "./MessageView.js";
import { FinalOutput } from "./FinalOutput.js";
import type { LogEntry } from "./shared.js";
import { cleanTime } from "./shared.js";

export const LogEntryView = React.memo(function LogEntryView({ entry }: { entry: LogEntry }): React.ReactElement | null {
  switch (entry.type) {
    case "tool":
      return <ToolDisplay event={entry.data as ToolEvent} />;
    case "tool-group":
      return <ToolGroupDisplay event={entry.data as ToolGroupEvent} />;
    case "reasoning":
      return (
        <Box borderLeft borderColor="gray" paddingLeft={1}>
          <Text dimColor>ℹ {(entry.data as { text: string }).text}</Text>
        </Box>
      );
    case "thinking-duration":
      return <ThinkingDuration durationMs={(entry.data as { durationMs: number }).durationMs} />;
    case "plan":
      return <PlanView steps={entry.data as PlanStep[]} />;
    case "error": {
      const err = entry.data as { message: string; code: string };
      return <ErrorView message={err.message} code={err.code} />;
    }
    case "final-output":
      return <FinalOutput text={(entry.data as { text: string }).text} />;
    case "info": {
      const data = String(entry.data);
      // User query
      if (data.startsWith("> ")) {
        return (
          <Box borderLeft borderColor="cyan" paddingLeft={1} marginTop={1}>
            <Text bold>{data}</Text>
          </Box>
        );
      }
      // Cancelled
      if (data === "Cancelled." || data.includes("Cancelled")) {
        return <Text color="yellow">⚠ Cancelled</Text>;
      }
      if (data.startsWith("Iteration limit exhausted.")) {
        return <Text color="yellow">⚠ {data}</Text>;
      }
      // Subagent completion with colored score
      if (data.includes("completed")) {
        const scoreMatch = data.match(/(?:score )?(\d+\.\d+),?\s*(?:partial|complete)/);
        if (scoreMatch) {
          const score = parseFloat(scoreMatch[1]!);
          const scoreColor = score >= 0.8 ? "green" : score >= 0.5 ? "yellow" : "red";
          const matchStart = data.indexOf(scoreMatch[0]);
          const beforeScore = data.slice(0, matchStart);
          const afterScore = data.slice(matchStart + scoreMatch[0].length);
          return (
            <Text>
              <Text dimColor>{cleanTime(beforeScore)}</Text>
              <Text color={scoreColor} bold>{score.toFixed(2)}</Text>
              <Text dimColor>{afterScore}</Text>
            </Text>
          );
        }
        return <Text dimColor>{cleanTime(data)}</Text>;
      }
      return <Text dimColor>{data}</Text>;
    }
    case "compaction": {
      const evt = entry.data as { tokensBefore: number; estimatedTokens: number };
      const pct = evt.tokensBefore > 0 ? Math.round(((evt.tokensBefore - evt.estimatedTokens) / evt.tokensBefore) * 100) : 0;
      return <Text dimColor>[context] Compacted: {Math.round(evt.tokensBefore / 1000)}k → {Math.round(evt.estimatedTokens / 1000)}k tokens ({pct}% reduction)</Text>;
    }
    default:
      return null;
  }
});
