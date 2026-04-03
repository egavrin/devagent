/**
 * SingleShotApp — Ink app for non-interactive (single query) mode.
 *
 * Executes one query immediately on mount, renders the same TUI
 * components, then writes final output to stdout and exits.
 */

import React, { useState, useEffect } from "react";
import { Static, useApp } from "ink";
import { StatusBar } from "./StatusBar.js";
import { Spinner } from "./Spinner.js";
import { SubagentPanel } from "./SubagentPanel.js";
import { LogEntryView } from "./LogEntryView.js";
import { useAgentLog } from "./useAgentLog.js";
import { getTurnCompletionNotice, type InteractiveQueryResult } from "./shared.js";
import type { EventBus } from "@devagent/runtime";

// ─── Types ──────────────────────────────────────────────────

export interface SingleShotAppProps {
  readonly bus: EventBus;
  readonly query: string;
  readonly onQuery: (query: string) => Promise<InteractiveQueryResult>;
  readonly model: string;
  readonly approvalMode: string;
  readonly onFinalOutput: (text: string) => void;
}

// ─── Component ──────────────────────────────────────────────

export function SingleShotApp({ bus, query, onQuery, model, approvalMode, onFinalOutput }: SingleShotAppProps): React.ReactElement {
  const { exit } = useApp();
  const [running, setRunning] = useState(true);

  const {
    log, status, subagents, spinnerMessage,
    addLog, flushThinking, flushGroup, nextId,
    refs,
  } = useAgentLog({ bus, model, approvalMode });

  // Execute query on mount
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const result = await onQuery(query);
        if (cancelled) return;
        flushThinking();
        flushGroup();

        const finalText = refs.textBuffer.current.trim() || result.lastText;
        if (finalText) onFinalOutput(finalText);

        const completionNotice = getTurnCompletionNotice(result.status);
        if (completionNotice) {
          addLog({ id: nextId("budget"), type: "info", data: completionNotice });
        }
      } catch (err) {
        addLog({ id: nextId("e"), type: "error", data: { message: err instanceof Error ? err.message : String(err), code: "QUERY_ERROR" } });
      }
      setRunning(false);
      setTimeout(() => { if (!cancelled) exit(); }, 100);
    })();
    return () => { cancelled = true; };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  let hasActiveSubagents = false;
  for (const a of subagents.values()) { if (a.status === "running") { hasActiveSubagents = true; break; } }

  return (
    <>
      <Static items={log}>
        {(entry) => <LogEntryView key={entry.id} entry={entry} />}
      </Static>
      {hasActiveSubagents && <SubagentPanel agents={subagents} />}
      <Spinner active={running} message={spinnerMessage} suffix={status.cost > 0 ? `$${status.cost.toFixed(4)}` : ""} />
      <StatusBar {...status} />
    </>
  );
}
