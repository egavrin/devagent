import { EventBus } from "@devagent/runtime";
import React, { useEffect, useMemo, useRef } from "react";
import { afterEach, describe, expect, it } from "vitest";

import {
  App,
  ITERATION_LIMIT_NOTICE,
  TranscriptView,
  renderResumeCommandOutput,
  renderSessionsCommandOutput,
} from "./App.js";
import {
  cleanupRenderedInstances,
  countPromptPlaceholders,
  insertModifiedReturn,
  renderForTest,
  settle,
  stripAnsi,
  typeAndSubmit,
  waitForRenders,
} from "./App.test-utils.js";
import type { TranscriptNode } from "./shared.js";
import { SingleShotApp } from "./SingleShotApp.js";
import { StatusBar } from "./StatusBar.js";
import { useAgentLog } from "./useAgentLog.js";
import type { PresentedTurnStatus } from "../transcript-composer.js";
import {
  makeErrorPart,
  makeFinalOutputPart,
  makeInfoPart,
  makeTurnSummaryPart,
  type TranscriptPart,
} from "../transcript-presenter.js";

function outputLines(text: string): string[] {
  return stripAnsi(text).replace(/\r/g, "\n").split("\n");
}

function expectWrappedCardFragmentsInsideGutter(text: string, fragments: ReadonlyArray<string>): void {
  const lines = outputLines(text);
  for (const fragment of fragments) {
    const matchingLines = lines.filter((line) => line.includes(fragment));
    expect(matchingLines, `expected output to include fragment ${fragment}`).not.toEqual([]);
    for (const line of matchingLines) {
      expect(line, `fragment ${fragment} should stay inside the card gutter`).toMatch(/^  │ /);
    }
  }
}

function makeTurnNode(
  userText: string,
  entries: ReadonlyArray<{ readonly id: string; readonly part: TranscriptPart }>,
  options?: {
    readonly id?: string;
    readonly status?: PresentedTurnStatus;
    readonly toolCalls?: number;
    readonly filesChanged?: number;
    readonly validationFailed?: boolean;
    readonly iterations?: number;
    readonly cost?: number;
    readonly elapsedMs?: number;
  },
): TranscriptNode {
  const resolved = {
    id: "turn-1",
    status: "completed" as PresentedTurnStatus,
    toolCalls: 0,
    filesChanged: 0,
    validationFailed: false,
    iterations: 1,
    cost: 0,
    elapsedMs: 100,
    ...options,
  };
  return {
    id: resolved.id,
    kind: "turn",
    turn: {
      id: resolved.id,
      userText,
      startedAt: 1_000,
      finishedAt: 1_000 + resolved.elapsedMs,
      status: resolved.status,
      entries,
      summary: {
        iterations: resolved.iterations,
        toolCalls: resolved.toolCalls,
        cost: resolved.cost,
        elapsedMs: resolved.elapsedMs,
      },
      metrics: {
        toolCalls: resolved.toolCalls,
        filesChanged: resolved.filesChanged,
        validationFailed: resolved.validationFailed,
        iterations: resolved.iterations,
        cost: resolved.cost,
        elapsedMs: resolved.elapsedMs,
      },
    },
  };
}

function AgentLogHarness({ entries }: { readonly entries: ReadonlyArray<string> }): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, appendStandalonePart } = useAgentLog({
    bus,
    model: "test-model",
  });
  const emittedCount = useRef(0);

  useEffect(() => {
    for (let index = emittedCount.current; index < entries.length; index++) {
      appendStandalonePart(`entry-${index + 1}`, makeInfoPart("info", [entries[index]!]));
    }
    emittedCount.current = entries.length;
  }, [entries, appendStandalonePart]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function ToolDiffHarness(): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Create a file", Date.now());
    bus.emit("tool:before", {
      name: "write_file",
      params: { path: "src/new-file.ts" },
      callId: "call-1",
    });
    bus.emit("tool:after", {
      name: "write_file",
      callId: "call-1",
      durationMs: 12,
      fileEditHiddenCount: 2,
      fileEdits: [{
        path: "src/new-file.ts",
        kind: "create",
        additions: 1,
        deletions: 0,
        unifiedDiff: "--- /dev/null\n+++ b/src/new-file.ts\n@@ -0,0 +1,1 @@\n+export const x = 1;",
        truncated: false,
        before: "",
        after: "export const x = 1;\n",
      }],
      result: {
        success: true,
        output: "Wrote file successfully.",
        error: null,
        artifacts: ["src/new-file.ts"],
        metadata: {
          fileEdits: [{
            path: "src/new-file.ts",
            kind: "create",
            additions: 1,
            deletions: 0,
            unifiedDiff: "--- /dev/null\n+++ b/src/new-file.ts\n@@ -0,0 +1,1 @@\n+export const x = 1;",
            truncated: false,
            before: "",
            after: "export const x = 1;\n",
          }],
        },
      },
    });
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 1, cost: 0, elapsedMs: 10 }));
  }, [bus, startTurn, completeTurn, nextId]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function StatusHarness(): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Write a file", Date.now());
    bus.emit("context:compacting", {
      estimatedTokens: 96_000,
      maxTokens: 128_000,
    });
    bus.emit("approval:request", {
      id: "approval-1",
      action: "edit",
      toolName: "write_file",
      details: "Write src/new-file.ts",
    });
    bus.emit("approval:response", {
      id: "approval-1",
      approved: true,
    });
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 0, cost: 0, elapsedMs: 10 }));
  }, [bus, startTurn, completeTurn, nextId]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function ToolSpecificHarness(): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Run validation", Date.now());

    bus.emit("tool:before", {
      name: "run_command",
      params: { command: "npm test" },
      callId: "call-cmd-1",
    });
    bus.emit("tool:after", {
      name: "run_command",
      callId: "call-cmd-1",
      durationMs: 45,
      result: {
        success: false,
        output: "Exit code: 1",
        error: "Command exited with code 1",
        artifacts: [],
        metadata: {
          commandResult: {
            command: "npm test",
            cwd: ".",
            exitCode: 1,
            timedOut: false,
            warningOnly: false,
            stdoutPreview: "stdout line 1\nstdout line 2",
            stderrPreview: "stderr line 1\nstderr line 2",
            stdoutTruncated: false,
            stderrTruncated: false,
          },
        },
      },
    });

    bus.emit("tool:before", {
      name: "write_file",
      params: { path: "src/new.ts" },
      callId: "call-validate-1",
    });
    bus.emit("tool:after", {
      name: "write_file",
      callId: "call-validate-1",
      durationMs: 12,
      fileEdits: [{
        path: "src/new.ts",
        kind: "create",
        additions: 1,
        deletions: 0,
        unifiedDiff: "--- /dev/null\n+++ b/src/new.ts\n@@ -0,0 +1,1 @@\n+export const x = 1;",
        truncated: false,
        before: "",
        after: "export const x = 1;\n",
      }],
      result: {
        success: true,
        output: "Wrote file",
        error: null,
        artifacts: ["src/new.ts"],
        metadata: {
          validationResult: {
            passed: false,
            diagnosticErrors: ["src/new.ts: Unexpected token"],
            testPassed: false,
            testOutputPreview: "Packages in scope: @devagent/cli\nRunning test in 5 packages",
            testSummary: {
              framework: "vitest",
              passed: 3,
              failed: 1,
              failureMessages: ["fails"],
            },
          },
        },
      },
    });

    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 2, cost: 0, elapsedMs: 50 }));
  }, [bus, startTurn, completeTurn, nextId]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function ToolScriptHarness({ success }: { readonly success: boolean }): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
    collapseFailures: true,
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Audit with script", Date.now());
    bus.emit("tool:before", {
      name: "execute_tool_script",
      params: { script: "print('done')" },
      callId: "call-script-1",
    });
    bus.emit("tool:after", {
      name: "execute_tool_script",
      callId: "call-script-1",
      durationMs: 20,
      result: {
        success,
        output: success ? "compact answer" : "",
        error: success ? null : "No output printed: call print(...) with the synthesized final answer.",
        artifacts: [],
        metadata: {
          toolScript: {
            toolCallCount: 3,
            innerOutputChars: 12000,
            finalOutputChars: success ? 14 : 0,
            durationMs: 20,
            timedOut: false,
            truncated: false,
          },
        },
      },
    });
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 1, cost: 0, elapsedMs: 20 }));
  }, [bus, completeTurn, nextId, startTurn, success]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function LargeCreateHarness(): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Create bench.ts", Date.now());
    bus.emit("tool:before", {
      name: "write_file",
      params: { path: "bench.ts" },
      callId: "call-large-create",
    });
    bus.emit("tool:after", {
      name: "write_file",
      callId: "call-large-create",
      durationMs: 12,
      fileEdits: [{
        path: "bench.ts",
        kind: "create",
        additions: 12,
        deletions: 0,
        unifiedDiff: "--- /dev/null\n+++ b/bench.ts\n@@ -0,0 +1,12 @@\n+line 1\n+line 2\n+line 3\n+line 4\n+line 5\n+line 6\n+line 7\n+line 8\n+line 9\n+line 10\n+line 11\n+line 12",
        truncated: false,
        before: "",
        after: "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7\nline 8\nline 9\nline 10\nline 11\nline 12\n",
      }],
      result: {
        success: true,
        output: "Wrote bench.ts.",
        error: null,
        artifacts: ["bench.ts"],
      },
    });
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 1, cost: 0, elapsedMs: 12 }));
  }, [bus, startTurn, completeTurn, nextId]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function FallbackHarness(): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Update src/index.ts", Date.now());
    bus.emit("tool:before", {
      name: "replace_in_file",
      params: { path: "src/index.ts" },
      callId: "call-2",
    });
    bus.emit("tool:after", {
      name: "replace_in_file",
      callId: "call-2",
      durationMs: 18,
      fileEdits: [{
        path: "src/index.ts",
        kind: "update",
        additions: 1,
        deletions: 1,
        unifiedDiff: "--- a/src/index.ts\n+++ b/src/index.ts\n@@ -1,1 +1,1 @@\n-old\n+new",
        truncated: false,
      }],
      result: {
        success: true,
        output: "Replaced text.",
        error: null,
        artifacts: ["src/index.ts"],
      },
    });
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 1, cost: 0, elapsedMs: 18 }));
  }, [bus, startTurn, completeTurn, nextId]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function BlankLineHarness(): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Update aa.txt", Date.now());
    bus.emit("tool:before", {
      name: "replace_in_file",
      params: { path: "aa.txt" },
      callId: "call-3",
    });
    bus.emit("tool:after", {
      name: "replace_in_file",
      callId: "call-3",
      durationMs: 10,
      fileEdits: [{
        path: "aa.txt",
        kind: "update",
        additions: 1,
        deletions: 0,
        unifiedDiff: "--- a/aa.txt\n+++ b/aa.txt\n@@ -1,2 +1,3 @@\n \n line\n+tail",
        truncated: false,
        before: "\nline\n",
        after: "\nline\ntail\n",
      }],
      result: {
        success: true,
        output: "Updated aa.txt.",
        error: null,
        artifacts: ["aa.txt"],
      },
    });
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 1, cost: 0, elapsedMs: 10 }));
  }, [bus, startTurn, completeTurn, nextId]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function MultiHunkHarness(): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Update multi.txt", Date.now());
    bus.emit("tool:before", {
      name: "replace_in_file",
      params: { path: "multi.txt" },
      callId: "call-multi",
    });
    bus.emit("tool:after", {
      name: "replace_in_file",
      callId: "call-multi",
      durationMs: 15,
      fileEdits: [{
        path: "multi.txt",
        kind: "update",
        additions: 2,
        deletions: 2,
        unifiedDiff: "--- a/multi.txt\n+++ b/multi.txt\n@@ -1,11 +1,11 @@\n start\n-old one\n+new one\n keep 1\n keep 2\n keep 3\n keep 4\n keep 5\n keep 6\n keep 7\n-old two\n+new two\n end",
        truncated: false,
        before: "start\nold one\nkeep 1\nkeep 2\nkeep 3\nkeep 4\nkeep 5\nkeep 6\nkeep 7\nold two\nend\n",
        after: "start\nnew one\nkeep 1\nkeep 2\nkeep 3\nkeep 4\nkeep 5\nkeep 6\nkeep 7\nnew two\nend\n",
      }],
      result: {
        success: true,
        output: "Updated multi.txt.",
        error: null,
        artifacts: ["multi.txt"],
      },
    });
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 1, cost: 0, elapsedMs: 15 }));
  }, [bus, startTurn, completeTurn, nextId]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

function NarrowHarness(): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { transcriptNodes, startTurn, completeTurn, nextId } = useAgentLog({
    bus,
    model: "test-model",
  });

  useEffect(() => {
    startTurn(nextId("turn"), "Create wide.ts", Date.now());
    bus.emit("tool:before", {
      name: "write_file",
      params: { path: "wide.ts" },
      callId: "call-wide",
    });
    bus.emit("tool:after", {
      name: "write_file",
      callId: "call-wide",
      durationMs: 11,
      fileEdits: [{
        path: "wide.ts",
        kind: "create",
        additions: 1,
        deletions: 0,
        unifiedDiff: "--- /dev/null\n+++ b/wide.ts\n@@ -0,0 +1,1 @@\n+const line = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz';",
        truncated: false,
        before: "",
        after: "const line = 'abcdefghijklmnopqrstuvwxyzabcdefghijklmnopqrstuvwxyz';\n",
      }],
      result: {
        success: true,
        output: "Wrote wide.ts.",
        error: null,
        artifacts: ["wide.ts"],
      },
    });
    completeTurn(nextId("summary"), makeTurnSummaryPart({ iterations: 1, toolCalls: 1, cost: 0, elapsedMs: 11 }));
  }, [bus, startTurn, completeTurn, nextId]);

  return React.createElement(TranscriptView, {
    showWelcome: false,
    transcriptNodes,
    model: "test-model",
  });
}

afterEach(cleanupRenderedInstances);

describe("interactive completion notices", () => {
  it("keeps the budget-exhausted follow-up hint stable", () => {
    expect(ITERATION_LIMIT_NOTICE).toBe("Iteration limit exhausted. Type /continue to proceed.");
  });

  it("renders turns after the welcome screen disappears", async () => {
    const view = renderForTest(
      React.createElement(TranscriptView, {
        showWelcome: true,
        transcriptNodes: [],
        model: "test-model",
      }),
    );

    await settle();

    view.rerender(
      React.createElement(TranscriptView, {
        showWelcome: false,
        transcriptNodes: [
          makeTurnNode("hello world", [], { status: "completed", elapsedMs: 250 }),
        ],
        model: "test-model",
      }),
    );

    await settle();

    const output = view.stdout.readAll();
    expect(output).toContain("hello world");
    expect(output).toContain("completed");
    expect(output).toContain("╭─");
  });

  it("keeps turn headers minimal and leaves metrics in the footer summary", async () => {
    const view = renderForTest(
      React.createElement(TranscriptView, {
        showWelcome: false,
        model: "test-model",
        transcriptNodes: [
          makeTurnNode(
            "add 1 line to bench.js",
            [{ id: "final-minimal-1", part: makeFinalOutputPart("Done.") }],
            {
              status: "completed",
              toolCalls: 2,
              filesChanged: 1,
              elapsedMs: 300,
            },
          ),
        ],
      }),
    );

    await settle();

    const plain = stripAnsi(view.stdout.readAll());
    expect(plain).toContain("╭─ completed add 1 line to bench.js");
    expect(plain).not.toContain("completed add 1 line to bench.js · 2 tools");
    expect(plain).not.toContain("✓ Done");
    expect(plain).not.toContain("2 tool calls");
    expect(plain).not.toContain("1 file changed");
    expect(plain).not.toContain("1 files");
  });

  it("does not render a duplicate completion footer after final output", async () => {
    const view = renderForTest(
      React.createElement(TranscriptView, {
        showWelcome: false,
        model: "test-model",
        transcriptNodes: [
          makeTurnNode(
            "Ship it",
            [{ id: "final-no-footer-1", part: makeFinalOutputPart("Done.\nEverything passed.") }],
            { status: "completed", toolCalls: 1, filesChanged: 1, elapsedMs: 300 },
          ),
        ],
      }),
    );

    await settle();

    const plain = stripAnsi(view.stdout.readAll());
    expect(plain).toContain("╭─ completed Ship it");
    expect(plain).toContain("devagent");
    expect(plain).toContain("Done.");
    expect(plain).not.toContain("✓ Done");
    expect(plain).not.toContain("tool calls");
  });

  it("keeps rendering new transcript entries after the transcript grows past fifty items", async () => {
    const firstBatch = Array.from({ length: 50 }, (_, index) => `entry-${index + 1}`);
    const secondBatch = Array.from({ length: 51 }, (_, index) => `entry-${index + 1}`);
    const thirdBatch = Array.from({ length: 52 }, (_, index) => `entry-${index + 1}`);
    const view = renderForTest(React.createElement(AgentLogHarness, { entries: firstBatch }));

    await settle();

    view.rerender(React.createElement(AgentLogHarness, { entries: secondBatch }));

    await settle();

    view.rerender(React.createElement(AgentLogHarness, { entries: thirdBatch }));

    await settle();

    expect(view.stdout.readAll()).toContain("entry-52");
  });

  it("renders file diffs from typed tool:after previews instead of parsing tool output", async () => {
    const view = renderForTest(React.createElement(ToolDiffHarness));

    await settle();

    const output = view.stdout.readAll();
    const plain = stripAnsi(output);
    expect(output).toContain("Create a file");
    expect(output).toContain("src/new-file.ts");
    expect(output).toContain("Added 1 line");
    expect(plain).toContain("export const x = 1;");
    expect(output).not.toContain("Wrote file successfully.");
    expect(output).not.toContain("@@");
    expect(output).not.toContain("--- /dev/null");
    expect(output).not.toContain("+++ b/src/new-file.ts");
    expect(output).toContain("... +2 more files");
  });
});

describe("framed TUI wrapping", () => {
  const cases: Array<{
    readonly name: string;
    readonly columns: number;
    readonly node: TranscriptNode;
    readonly fragments: ReadonlyArray<string>;
  }> = [
    {
      name: "error card",
      columns: 72,
      node: makeTurnNode(
        "Verify image",
        [{
          id: "provider-error-1",
          part: makeErrorPart({
            message: "Provider error (attempt 1, ProviderError): OpenAI API error: terminated. Retrying in 611ms...",
            code: "PROVIDER_RETRY",
          }),
        }],
        { status: "error", elapsedMs: 300 },
      ),
      fragments: ["Provider error", "Retrying in", "611ms"],
    },
    {
      name: "info card",
      columns: 66,
      node: {
        id: "info-1",
        kind: "part",
        part: makeInfoPart("status", [
          "A very long status update is rendered as a transcript card and should wrap through the shared framed-line gutter instead of escaping to column zero.",
        ]),
      },
      fragments: ["A very long", "shared framed-line", "column zero"],
    },
    {
      name: "final output card",
      columns: 68,
      node: makeTurnNode(
        "Explain retry",
        [{
          id: "final-wrap-1",
          part: makeFinalOutputPart("**Provider retry** details stay visible while markdown formatted assistant output wraps inside the framed response gutter."),
        }],
        { status: "completed", elapsedMs: 300 },
      ),
      fragments: ["Provider retry", "markdown formatted", "response gutter"],
    },
  ];

  it.each(cases)("keeps wrapped $name text inside the left gutter", async ({ columns, fragments, node }) => {
    const view = renderForTest(
      React.createElement(TranscriptView, {
        showWelcome: false,
        model: "test-model",
        transcriptNodes: [node],
      }),
      { columns },
    );

    await settle();

    expectWrappedCardFragmentsInsideGutter(view.stdout.readAll(), fragments);
  });
});

describe("interactive transcript typed rows", () => {
  it("renders typed progress and status transcript rows for compaction and approvals", async () => {
    const view = renderForTest(React.createElement(StatusHarness));

    await settle();

    const output = view.stdout.readAll();
    expect(output).toContain("Write a file");
    expect(output).toContain("Compacting context");
    expect(output).toContain("96k / 128k tokens");
    expect(output).toContain("Awaiting approval for write_file");
    expect(output).toContain("Write src/new-file.ts");
    expect(output).toContain("Approved");
  });

  it("renders typed command, validation, and diagnostic transcript rows", async () => {
    const view = renderForTest(React.createElement(ToolSpecificHarness));

    await settle();

    const output = view.stdout.readAll();
    const plain = stripAnsi(output);
    expect(output).toContain("Run validation");
    expect(output).toContain("command");
    expect(output).toContain("npm test");
    expect(output).toContain("Exited with code 1");
    expect(plain).toContain("stdout line 1");
    expect(plain).toContain("stdout line 2");
    expect(plain).toContain("stderr line 1");
    expect(plain).toContain("stderr line 2");
    expect(output).toContain("validation");
    expect(output).toContain("Validation failed");
    expect(output).toContain("vitest: 3 passed, 1 failed");
    expect(plain).toContain("Packages in scope: @devagent/cli");
    expect(plain).toContain("Running test in 5 packages");
    expect(output).toContain("diagnostics (1)");
    expect(output).toContain("src/new.ts: Unexpected token");
    expect(plain).not.toContain("↵");
  });

  it("renders execute_tool_script telemetry in the TUI transcript", async () => {
    const view = renderForTest(React.createElement(ToolScriptHarness, { success: true }));

    await settle();

    const output = stripAnsi(view.stdout.readAll());
    expect(output).toContain("execute_tool_script");
    expect(output).toContain("3 inner call(s), 12000 hidden chars -> 14 stdout chars");
  });

  it("renders execute_tool_script failures instead of hiding them in collapsed failures", async () => {
    const view = renderForTest(React.createElement(ToolScriptHarness, { success: false }));

    await settle();

    const output = stripAnsi(view.stdout.readAll());
    expect(output).toContain("execute_tool_script");
    expect(output).toContain("No output printed");
    expect(output).not.toContain("1 calls failed");
  });

  it("caps large snapshot diffs in condensed mode", async () => {
    const view = renderForTest(React.createElement(LargeCreateHarness));

    await waitForRenders();

    const output = view.stdout.readAll();
    const plain = stripAnsi(output);
    expect(output).toContain("bench.ts");
    expect(output).toContain("Added 12 lines");
    expect(plain).toContain("line 1");
    expect(plain).toContain("line 10");
    expect(plain).not.toContain("line 11");
    expect(output).toContain("... +2 more diff lines");
  });

  it("falls back to unified diff rendering when snapshots are absent", async () => {
    const view = renderForTest(React.createElement(FallbackHarness));

    await settle();

    const output = view.stdout.readAll();
    expect(output).toContain("Added 1 line, removed 1 line");
    expect(output).toContain("new");
    expect(output).toContain("old");
    expect(output).not.toContain("+++ b/src/index.ts");
    expect(output).not.toContain("@@");
  });

  it("shows explicit placeholders for blank snapshot lines", async () => {
    const view = renderForTest(React.createElement(BlankLineHarness));

    await settle();

    const output = view.stdout.readAll();
    expect(output).toContain("<blank>");
    expect(output).toContain("Added 1 line");
    expect(output).toContain("tail");
    expect(output).not.toContain("@@");
  });

  it("renders multiple hunks with Claude-style ellipsis separators", async () => {
    const view = renderForTest(React.createElement(MultiHunkHarness));

    await settle();

    const output = view.stdout.readAll();
    const plain = stripAnsi(output);
    expect(output).toContain("Added 2 lines, removed 2 lines");
    expect(plain).toContain("new one");
    expect(output).toContain("...");
    expect(output).not.toContain("@@");
  });

  it("truncates long code lines to the available terminal width", async () => {
    const view = renderForTest(React.createElement(NarrowHarness), { columns: 48 });

    await settle();

    const output = view.stdout.readAll();
    const plain = stripAnsi(output);
    expect(plain).toContain("const line =");
    expect(output).toContain("…");
  });
});

describe("session command output", () => {
  it("renders titled session rows for /sessions", () => {
    const text = renderSessionsCommandOutput([{
      id: "12345678-aaaa-bbbb-cccc-1234567890ab",
      updatedAt: 60_000,
      title: "Fix auth retry loop",
      repoLabel: "devagent",
      cost: 0.125,
    }]);

    expect(text).toContain("Recent sessions:");
    expect(text).toContain("Fix auth retry loop");
    expect(text).toContain("12345678  devagent");
  });

  it("renders titled session rows for /resume", () => {
    const text = renderResumeCommandOutput([{
      id: "87654321-aaaa-bbbb-cccc-1234567890ab",
      updatedAt: 60_000,
      title: "Review provider config",
      repoLabel: "providers",
    }]);

    expect(text).toContain("Sessions (use --resume <id> to continue):");
    expect(text).toContain("Review provider config");
    expect(text).toContain("87654321  providers");
  });
});

describe("interactive prompt scrollback and status", () => {
  it("renders final output inside the transcript card system", async () => {
    const view = renderForTest(
      React.createElement(TranscriptView, {
        showWelcome: false,
        model: "test-model",
        transcriptNodes: [
          makeTurnNode(
            "Ship it",
            [{ id: "final-1", part: makeFinalOutputPart("Done.\nEverything passed.") }],
            { status: "completed", elapsedMs: 300 },
          ),
        ],
      }),
    );

    await settle();

    const output = view.stdout.readAll();
    expect(output).toContain("devagent");
    expect(output).toContain("Done.");
    expect(output).toContain("Everything passed.");
    expect(output).toContain("╭─");
    expect(output).toContain("╰─");
  });

  it("keeps prompt scrollback stable after a completed turn", async () => {
    const bus = new EventBus();
    const view = renderForTest(
      React.createElement(App, {
        bus,
        model: "test-model",
        approvalMode: "autopilot",
        cwd: "/tmp/devagent",
        onClear: () => {},
        onCycleApprovalMode: () => {},
        onQuery: async () => {
          bus.emit("iteration:start", {
            iteration: 1,
            maxIterations: 10,
            estimatedTokens: 100,
            maxContextTokens: 1_000,
          });
          await settle();
          return {
            iterations: 1,
            toolCalls: 0,
            lastText: "Bottom line: done",
            status: "success" as const,
          };
        },
      }),
    );

    await settle();
    await typeAndSubmit(view.stdin, "x");
    await waitForRenders();

    const output = view.stdout.readAll();
    const promptPlaceholders = countPromptPlaceholders(output);
    expect(promptPlaceholders).toBeGreaterThanOrEqual(1);
    expect(promptPlaceholders).toBeLessThanOrEqual(2);
    expect(output).toContain("Bottom line: done");
  });
});

describe("interactive incomplete query status", () => {
  it.each([
    {
      name: "empty model response",
      query: "reproduce stop",
      status: "empty_response",
      notice: "Model returned no final response. Type /continue to retry",
    },
    {
      name: "aborted run",
      query: "cancelled work",
      status: "aborted",
      notice: "Run stopped before completion. Type /continue to retry",
    },
  ] as const)("surfaces an $name as an incomplete turn", async ({ query, status, notice }) => {
    const view = renderForTest(
      React.createElement(App, {
        bus: new EventBus(),
        model: "test-model",
        approvalMode: "autopilot",
        cwd: "/tmp/devagent",
        onClear: () => {},
        onCycleApprovalMode: () => {},
        onQuery: async () => ({
          iterations: 2,
          toolCalls: 1,
          lastText: null,
          status,
        }),
      }),
    );

    await settle();
    await typeAndSubmit(view.stdin, query);
    await waitForRenders();

    const output = stripAnsi(view.stdout.readAll());
    expect(output).toContain(`╭─ error ${query}`);
    expect(output).toContain(notice);
    expect(output).not.toContain(`╭─ completed ${query}`);
  });
});

describe("single-shot incomplete query status", () => {
  it.each([
    {
      name: "budget exhausted run",
      status: "budget_exceeded",
      label: "budget",
      noticeFragments: [
        "Iteration limit exhausted before completion.",
        "Re-run with a higher iteration limit or start interactive mode to",
        "continue.",
      ],
    },
    {
      name: "empty model response",
      status: "empty_response",
      label: "error",
      noticeFragments: [
        "Model returned no final response.",
        "Re-run the command to retry, or switch provider/model if it repeats.",
      ],
    },
    {
      name: "aborted run",
      status: "aborted",
      label: "error",
      noticeFragments: ["Run stopped before completion. Re-run the command to retry."],
    },
  ] as const)("surfaces an $name as an incomplete turn", async ({ status, label, noticeFragments }) => {
    let finalOutput: string | null = null;
    const view = renderForTest(
      React.createElement(SingleShotApp, {
        bus: new EventBus(),
        query: "single shot",
        model: "test-model",
        onFinalOutput: (text) => {
          finalOutput = text;
        },
        onQuery: async () => ({
          iterations: 1,
          toolCalls: 0,
          lastText: null,
          status,
        }),
      }),
      { columns: 180 },
    );

    await waitForRenders();

    const output = stripAnsi(view.stdout.readAll());
    expect(output).toContain(`╭─ ${label} single shot`);
    for (const fragment of noticeFragments) {
      expect(output).toContain(fragment);
    }
    expect(output).not.toContain("/continue");
    expect(output).not.toContain("╭─ completed single shot");
    expect(finalOutput).toBeNull();
  });
});

describe("interactive prompt editing and status bar", () => {
  it("submits multiline prompts when modified Return inserts a newline", async () => {
    let submittedQuery: string | null = null;
    const view = renderForTest(
      React.createElement(App, {
        bus: new EventBus(),
        model: "test-model",
        approvalMode: "autopilot",
        cwd: "/tmp/devagent",
        onClear: () => {},
        onCycleApprovalMode: () => {},
        onQuery: async (query) => {
          submittedQuery = query;
          return {
            iterations: 1,
            toolCalls: 0,
            lastText: "done",
            status: "success" as const,
          };
        },
      }),
      { columns: 28 },
    );

    await settle();
    view.stdin.write("alpha");
    await settle();
    await insertModifiedReturn(view.stdin);
    view.stdin.write("beta");
    await settle();
    view.stdin.write("\r");
    await waitForRenders();

    expect(submittedQuery).toBe("alpha\nbeta");

    const output = stripAnsi(view.stdout.readAll());
    expect(output).toContain("done");
  });

  it("renders markdown tables cleanly inside the assistant card", async () => {
    const view = renderForTest(
      React.createElement(TranscriptView, {
        showWelcome: false,
        model: "test-model",
        transcriptNodes: [
          makeTurnNode(
            "Summarize renames",
            [{
              id: "final-table-1",
              part: makeFinalOutputPart([
                "| Before | After |",
                "| --- | --- |",
                "| a (param) | array |",
                "| _fn | sortFn |",
              ].join("\n")),
            }],
            { status: "completed", elapsedMs: 300 },
          ),
        ],
      }),
    );

    await settle();

    const output = view.stdout.readAll();
    expect(output).toContain("Before");
    expect(output).toContain("After");
    expect(output).toContain("a (param)");
    expect(output).toContain("sortFn");
    expect(output).toContain("─┼─");
    expect(output).not.toContain("│ │ Before");
  });

  it("renders the token bar inline in the status bar without mode text", async () => {
    const view = renderForTest(
      React.createElement(StatusBar, {
        model: "cortex",
        cost: 0,
        inputTokens: 9_000,
        maxContextTokens: 207_000,
        iteration: 6,
        maxIterations: 0,
        approvalMode: "default",
        cwd: "/tmp/devagent",
        running: false,
        hasApproval: false,
      }),
    );

    await settle();

    const plain = stripAnsi(view.stdout.readAll());
    expect(plain).toContain("▰");
    expect(plain).toContain("9k/207k (4%)");
    expect(plain).toContain("devagent");
    expect(plain).not.toContain("default");
    expect(plain).not.toContain("iter 6");
  });

  it("shows finite iteration limits inline beside the token bar", async () => {
    const view = renderForTest(
      React.createElement(StatusBar, {
        model: "cortex",
        cost: 0,
        inputTokens: 9_000,
        maxContextTokens: 207_000,
        iteration: 6,
        maxIterations: 33,
        approvalMode: "autopilot",
        cwd: "/tmp/devagent",
        running: false,
        hasApproval: false,
      }),
    );

    await settle();

    const plain = stripAnsi(view.stdout.readAll());
    expect(plain).toContain("▰");
    expect(plain).toContain("9k/207k (4%)");
    expect(plain).toContain("iter 6/33");
    expect(plain).not.toContain("autopilot");
  });
});
