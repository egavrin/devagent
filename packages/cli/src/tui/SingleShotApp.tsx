/**
 * SingleShotApp — Ink app for non-interactive (single query) mode.
 *
 * Executes one query immediately on mount, renders the same TUI
 * components, then writes final output to stdout and exits.
 */

import { useApp } from "ink";
import React, { useEffect, useRef, useState } from "react";

import { TranscriptView, turnStatusForQueryStatus } from "./App.js";
import type { InteractiveQueryResult } from "./shared.js";
import { Spinner } from "./Spinner.js";
import { StatusBar } from "./StatusBar.js";
import { SubagentPanel } from "./SubagentPanel.js";
import { useAgentLog } from "./useAgentLog.js";
import { makeErrorPart, makeInfoPart, makeTurnSummaryPart } from "../transcript-presenter.js";
import type { EventBus } from "@devagent/runtime";

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
  const hasStartedRef = useRef(false);
  const mountedRef = useRef(true);

  const {
    transcriptNodes, status, subagents, spinnerMessage,
    startTurn, appendTurnPart, completeTurn, flushThinking, flushGroup, nextId,
    refs,
  } = useAgentLog({ bus, model });

  useEffect(() => () => { mountedRef.current = false; }, []);

  useEffect(() => {
    if (hasStartedRef.current) return;
    hasStartedRef.current = true;
    const runQuery = async (): Promise<void> => {
      refs.turnStart.current = Date.now(); refs.turnToolCount.current = 0; refs.costAccum.current = 0;
      startTurn(nextId("turn"), query, refs.turnStart.current);
      try {
        const result = await onQuery(query);
        if (!mountedRef.current) return;
        flushThinking();
        flushGroup();

        const finalText = refs.textBuffer.current.trim() || result.lastText;
        if (finalText) {
          appendTurnPart(nextId("final"), { kind: "final-output", data: { text: finalText } });
          onFinalOutput(finalText);
        }
        const notice = noticeForSingleShotStatus(result.status);
        if (notice) appendTurnPart(nextId("status"), makeInfoPart("status", [notice]));

        completeTurn(
          nextId("summary"),
          makeTurnSummaryPart({ iterations: result.iterations, toolCalls: refs.turnToolCount.current, cost: refs.costAccum.current, elapsedMs: Date.now() - refs.turnStart.current }),
          { status: turnStatusForQueryStatus(result.status), finishedAt: Date.now() },
        );
      } catch (err) {
        if (!mountedRef.current) return;
        appendTurnPart(nextId("e"), makeErrorPart({ message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" }));
        completeTurn(
          nextId("summary"),
          makeTurnSummaryPart({ iterations: 0, toolCalls: refs.turnToolCount.current, cost: refs.costAccum.current, elapsedMs: Date.now() - refs.turnStart.current }),
          { status: "error", finishedAt: Date.now() },
        );
      }
      setRunning(false);
      scheduleExit(() => !mountedRef.current, exit);
    };

    void runQuery();
  }, [appendTurnPart, completeTurn, exit, flushGroup, flushThinking, model, nextId, onFinalOutput, onQuery, query, refs, startTurn]);

  return (
    <>
      <TranscriptView showWelcome={false} transcriptNodes={transcriptNodes} model={model} keepLatestNodeLive />
      {hasActiveSubagents(subagents) && <SubagentPanel agents={subagents} />}
      <Spinner active={running} message={spinnerMessage} suffix={status.cost > 0 ? `$${status.cost.toFixed(4)}` : ""} />
      <StatusBar {...status} />
    </>
  );
}

function scheduleExit(isCancelled: () => boolean, exit: () => void): void {
  setTimeout(() => { if (!isCancelled()) exit(); }, 100);
}

function hasActiveSubagents(subagents: ReadonlyMap<unknown, { readonly status: string }>): boolean {
  return Array.from(subagents.values()).some((agent) => agent.status === "running");
}

function noticeForSingleShotStatus(status: InteractiveQueryResult["status"]): string | null {
  if (status === "budget_exceeded") {
    return "Iteration limit exhausted before completion. Re-run with a higher iteration limit or start interactive mode to continue.";
  }
  if (status === "empty_response") {
    return "Model returned no final response. Re-run the command to retry, or switch provider/model if it repeats.";
  }
  if (status === "aborted") {
    return "Run stopped before completion. Re-run the command to retry.";
  }
  return null;
}
