import { EventBus } from "@devagent/runtime";
import React, { useEffect, useMemo } from "react";
import { afterEach, describe, expect, it } from "vitest";

import { TranscriptView } from "./App.js";
import {
  cleanupRenderedInstances,
  renderForTest,
  settle,
  stripAnsi,
  waitForRenders,
} from "./App.test-utils.js";
import { SingleShotApp } from "./SingleShotApp.js";
import { useAgentLog } from "./useAgentLog.js";
import { makeTurnSummaryPart } from "../transcript-presenter.js";

afterEach(cleanupRenderedInstances);

function emitTool(bus: EventBus, name: string, callId: string, durationMs: number): void {
  bus.emit("tool:before", {
    name,
    params: { pattern: "." },
    callId,
  });
  emitToolAfter(bus, name, callId, durationMs);
}

function emitToolAfter(bus: EventBus, name: string, callId: string, durationMs: number): void {
  bus.emit("tool:after", {
    name,
    callId,
    durationMs,
    result: {
      success: true,
      output: "1 file(s) found",
      error: null,
      artifacts: [],
    },
  });
}

function VisualHarness(
  { mode, onSpinnerMessage }: {
    readonly mode: "grouped" | "sequential";
    readonly onSpinnerMessage?: (message: string | undefined) => void;
  },
): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, spinnerMessage, startTurn, completeTurn, nextId } = useAgentLog({ bus, model: "test-model" });

  useEffect(() => {
    onSpinnerMessage?.(spinnerMessage);
  }, [onSpinnerMessage, spinnerMessage]);

  useEffect(() => {
    startTurn(nextId("turn"), mode === "grouped" ? "Check LSP" : "Find twice", Date.now());
    if (mode === "grouped") {
      bus.emit("tool:before", { name: "lsp", params: { operation: "diagnostics", path: "src/tmp.ts" }, callId: "call-1" });
      bus.emit("tool:before", { name: "lsp", params: { operation: "symbols", path: "src/tmp.ts" }, callId: "call-2" });
      emitToolAfter(bus, "lsp", "call-1", 1);
      emitToolAfter(bus, "lsp", "call-2", 9);
    } else {
      emitTool(bus, "find_files", "call-1", 50);
      emitTool(bus, "find_files", "call-2", 52);
    }
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 2, cost: 0, elapsedMs: 120 }));
  }, [bus, completeTurn, mode, nextId, startTurn]);

  return React.createElement(TranscriptView, { showWelcome: false, transcriptNodes, model: "test-model" });
}

describe("TUI visual regressions", () => {
  it("flushes grouped tool rows and clears the active tool label", async () => {
    const spinnerMessages: Array<string | undefined> = [];
    const view = renderForTest(React.createElement(VisualHarness, {
      mode: "grouped",
      onSpinnerMessage: (message) => spinnerMessages.push(message),
    }));

    await settle();

    const plain = stripAnsi(view.stdout.readAll());
    expect(plain).toContain("✓ lsp ×2");
    expect(spinnerMessages.at(-1)).toBeUndefined();
  });

  it("does not add grouped summaries for sequential same-name tools", async () => {
    const view = renderForTest(React.createElement(VisualHarness, { mode: "sequential" }));

    await settle();

    const plain = stripAnsi(view.stdout.readAll());
    expect(plain).toContain("✓ find_files (50ms)");
    expect(plain).toContain("✓ find_files (52ms)");
    expect(plain).not.toContain("✓ find_files ×2");
  });

  it("finishes single-shot turns after tool-event rerenders", async () => {
    const bus = new EventBus();
    let finalOutput: string | null = null;
    const view = renderForTest(React.createElement(SingleShotApp, {
      bus,
      query: "single shot tools",
      model: "test-model",
      onFinalOutput: (text) => { finalOutput = text; },
      onQuery: () => new Promise((resolve) => {
        emitTool(bus, "lsp", "call-lsp-1", 5);
        setTimeout(() => resolve({
          iterations: 1,
          toolCalls: 1,
          lastText: "LSP finished.",
          status: "success",
        }), 0);
      }),
    }), { columns: 180 });

    await waitForRenders();

    const output = stripAnsi(view.stdout.readAll());
    expect(output).toContain("╭─ completed single shot tools");
    expect(output).toContain("LSP finished.");
    expect(finalOutput).toBe("LSP finished.");
  });
});
