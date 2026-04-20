/**
 * SingleShotApp — Ink app for non-interactive (single query) mode.
 *
 * Executes one query immediately on mount, renders the same TUI
 * components, then writes final output to stdout and exits.
 */

import React, { useEffect, useRef, useState } from "react";
import { Static, useApp } from "ink";
import { StatusBar } from "./StatusBar.js";
import { Spinner } from "./Spinner.js";
import { SubagentPanel } from "./SubagentPanel.js";
import { useAgentLog } from "./useAgentLog.js";
import type { InteractiveQueryResult } from "./shared.js";
import type { EventBus } from "@devagent/runtime";
import { TranscriptView } from "./App.js";
import { makeErrorPart, makeTurnSummaryPart } from "../transcript-presenter.js";

// ─── Types ──────────────────────────────────────────────────

export interface SingleShotAppProps {
  readonly bus: EventBus;
  readonly query: string;
  readonly onQuery: (query: string) => Promise<InteractiveQueryResult>;
  readonly model: string;
  readonly onFinalOutput: (text: string) => void;
}

// ─── Component ──────────────────────────────────────────────

export function SingleShotApp({ bus, query, onQuery, model, onFinalOutput }: SingleShotAppProps): React.ReactElement {
  const { exit } = useApp();
  const [running, setRunning] = useState(true);
  const launchedRef = useRef(false);

  const {
    transcriptNodes, status, subagents, spinnerMessage,
    startTurn, appendTurnPart, completeTurn, flushThinking, flushGroup, nextId,
    refs,
  } = useAgentLog({ bus, model });

  // Execute query on mount
  useEffect(() => {
    if (launchedRef.current) {
      return;
    }
    launchedRef.current = true;

    let cancelled = false;
    (async () => {
      refs.turnStart.current = Date.now();
      refs.turnToolCount.current = 0;
      refs.costAccum.current = 0;
      startTurn(nextId("turn"), query, refs.turnStart.current);
      try {
        const result = await onQuery(query);
        if (cancelled) return;
        flushThinking();
        flushGroup();

        const finalText = refs.textBuffer.current.trim() || result.lastText;
        if (finalText) {
          appendTurnPart(nextId("final"), { kind: "final-output", data: { text: finalText } });
          onFinalOutput(finalText);
        }

        completeTurn(
          nextId("summary"),
          makeTurnSummaryPart({ iterations: result.iterations, toolCalls: refs.turnToolCount.current, cost: refs.costAccum.current, elapsedMs: Date.now() - refs.turnStart.current }),
          { status: result.status === "budget_exceeded" ? "budget_exceeded" : "completed", finishedAt: Date.now() },
        );
      } catch (err) {
        appendTurnPart(nextId("e"), makeErrorPart({ message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" }));
        completeTurn(
          nextId("summary"),
          makeTurnSummaryPart({ iterations: 0, toolCalls: refs.turnToolCount.current, cost: refs.costAccum.current, elapsedMs: Date.now() - refs.turnStart.current }),
          { status: "error", finishedAt: Date.now() },
        );
      }
      setRunning(false);
      setTimeout(() => { if (!cancelled) exit(); }, 100);
    })().catch((err: unknown) => {
      appendTurnPart(nextId("e"), makeErrorPart({ message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" }));
      completeTurn(
        nextId("summary"),
        makeTurnSummaryPart({
          iterations: 0,
          toolCalls: refs.turnToolCount.current,
          cost: refs.costAccum.current,
          elapsedMs: Date.now() - refs.turnStart.current,
        }),
        { status: "error", finishedAt: Date.now() },
      );
      setRunning(false);
      setTimeout(() => { if (!cancelled) exit(); }, 100);
    });
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  let hasActiveSubagents = false;
  for (const a of subagents.values()) { if (a.status === "running") { hasActiveSubagents = true; break; } }

  return (
    <>
      <TranscriptView showWelcome={false} transcriptNodes={transcriptNodes} model={model} />
      {hasActiveSubagents && <SubagentPanel agents={subagents} />}
      <Spinner active={running} message={spinnerMessage} suffix={status.cost > 0 ? `$${status.cost.toFixed(4)}` : ""} />
      <StatusBar {...status} />
    </>
  );
}
