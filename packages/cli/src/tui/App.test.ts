import { PassThrough, Writable } from "node:stream";
import React, { useEffect, useMemo, useRef } from "react";
import { render } from "ink";
import { EventBus } from "@devagent/runtime";
import { afterEach, describe, expect, it } from "vitest";

import {
  App,
  ITERATION_LIMIT_NOTICE,
  TranscriptView,
  renderResumeCommandOutput,
  renderSessionsCommandOutput,
} from "./App.js";
import { StatusBar } from "./StatusBar.js";
import { useAgentLog } from "./useAgentLog.js";
import type { TranscriptNode } from "./shared.js";
import type { PresentedTurnStatus } from "../transcript-composer.js";
import {
  makeFinalOutputPart,
  makeInfoPart,
  makeTurnSummaryPart,
  type TranscriptPart,
} from "../transcript-presenter.js";

class TestInput extends PassThrough {
  readonly isTTY = true;

  setRawMode(_value: boolean): void {}

  ref(): this {
    return this;
  }

  unref(): this {
    return this;
  }
}

class TestOutput extends Writable {
  readonly isTTY = true;
  readonly columns: number;
  readonly rows = 40;
  private readonly chunks: string[] = [];

  constructor(columns: number = 120) {
    super();
    this.columns = columns;
  }

  override _write(
    chunk: string | Uint8Array,
    _encoding: BufferEncoding,
    callback: (error?: Error | null) => void,
  ): void {
    this.chunks.push(typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf8"));
    callback();
  }

  readAll(): string {
    return this.chunks.join("");
  }
}

const instances: Array<{ unmount: () => void; cleanup: () => void }> = [];

async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 20));
}

async function waitForRenders(cycles: number = 6): Promise<void> {
  for (let index = 0; index < cycles; index++) {
    await settle();
  }
}

function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]*m/g, "");
}

function countPromptPlaceholders(text: string): number {
  return (stripAnsi(text).match(/Ask anything…/g) ?? []).length;
}

async function typeAndSubmit(stdin: TestInput, text: string): Promise<void> {
  stdin.write(text);
  await settle();
  stdin.write("\r");
}

async function insertModifiedReturn(stdin: TestInput): Promise<void> {
  stdin.write("\x1b\r");
  await settle();
}

function renderForTest(
  node: React.ReactElement,
  options?: { readonly columns?: number },
): {
  readonly stdout: TestOutput;
  readonly stdin: TestInput;
  readonly rerender: (tree: React.ReactElement) => void;
  readonly unmount: () => void;
  readonly cleanup: () => void;
} {
  const stdout = new TestOutput(options?.columns);
  const stdin = new TestInput();
  const stderr = new TestOutput(options?.columns);
  const instance = render(node, {
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stderr: stderr as unknown as NodeJS.WriteStream,
    debug: true,
    exitOnCtrlC: false,
    patchConsole: false,
  });
  instances.push(instance);
  return {
    stdout,
    stdin,
    rerender: instance.rerender,
    unmount: instance.unmount,
    cleanup: instance.cleanup,
  };
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
  const id = options?.id ?? "turn-1";
  const iterations = options?.iterations ?? 1;
  const cost = options?.cost ?? 0;
  const elapsedMs = options?.elapsedMs ?? 100;
  return {
    id,
    kind: "turn",
    turn: {
      id,
      userText,
      startedAt: 1_000,
      finishedAt: 1_000 + elapsedMs,
      status: options?.status ?? "completed",
      entries,
      summary: {
        iterations,
        toolCalls: options?.toolCalls ?? 0,
        cost,
        elapsedMs,
      },
      metrics: {
        toolCalls: options?.toolCalls ?? 0,
        filesChanged: options?.filesChanged ?? 0,
        validationFailed: options?.validationFailed ?? false,
        iterations,
        cost,
        elapsedMs,
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

describe("interactive completion notices", () => {
  afterEach(() => {
    while (instances.length > 0) {
      const instance = instances.pop();
      instance?.unmount();
      instance?.cleanup();
    }
  });

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
    expect(countPromptPlaceholders(output)).toBe(2);
    expect(output).toContain("Bottom line: done");
  });

  it("keeps prompt scrollback stable after idle slash-command output", async () => {
    const view = renderForTest(
      React.createElement(App, {
        bus: new EventBus(),
        model: "test-model",
        approvalMode: "autopilot",
        cwd: "/tmp/devagent",
        onClear: () => {},
        onCycleApprovalMode: () => {},
        onQuery: async () => ({
          iterations: 0,
          toolCalls: 0,
          lastText: null,
          status: "success" as const,
        }),
      }),
    );

    await settle();
    await typeAndSubmit(view.stdin, "/help");
    await waitForRenders();

    const output = view.stdout.readAll();
    expect(countPromptPlaceholders(output)).toBe(2);
    expect(output).toContain("Commands: /clear (reset)");
  });

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
