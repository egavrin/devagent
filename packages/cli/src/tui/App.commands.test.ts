import { EventBus } from "@devagent/runtime";
import { render } from "ink";
import { PassThrough, Writable } from "node:stream";
import React from "react";
import { afterEach, describe, expect, it } from "vitest";

import { App, handleCancelShortcut } from "./App.js";

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

  clear(): void {
    this.chunks.length = 0;
  }
}

const instances: Array<{ unmount: () => void; cleanup: () => void }> = [];

afterEach(() => {
  while (instances.length > 0) {
    const instance = instances.pop();
    instance?.unmount();
    instance?.cleanup();
  }
});

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

function renderForTest(node: React.ReactElement): { readonly stdout: TestOutput; readonly stdin: TestInput } {
  const stdout = new TestOutput();
  const stdin = new TestInput();
  const stderr = new TestOutput();
  instances.push(render(node, {
    stdout: stdout as unknown as NodeJS.WriteStream,
    stdin: stdin as unknown as NodeJS.ReadStream,
    stderr: stderr as unknown as NodeJS.WriteStream,
    debug: true,
    exitOnCtrlC: false,
    patchConsole: false,
  }));
  return { stdout, stdin };
}

function makeIdleAppProps(overrides?: { readonly bus?: EventBus; readonly onClear?: () => void }) {
  return {
    bus: overrides?.bus ?? new EventBus(),
    model: "test-model",
    approvalMode: "autopilot" as const,
    cwd: "/tmp/devagent",
    onClear: overrides?.onClear ?? (() => {}),
    onCycleApprovalMode: () => {},
    onQuery: async () => ({
      iterations: 0,
      toolCalls: 0,
      lastText: null,
      status: "success" as const,
    }),
  };
}

function emitBusyStatus(bus: EventBus): void {
  bus.emit("iteration:start", {
    iteration: 7,
    maxIterations: 10,
    estimatedTokens: 42_000,
    maxContextTokens: 100_000,
  });
  bus.emit("cost:update", {
    inputTokens: 42_000,
    outputTokens: 100,
    totalCost: 0.1234,
    model: "test-model",
  });
}

describe("interactive prompt commands", () => {
  it("keeps prompt scrollback stable after idle slash-command output", async () => {
    const view = renderForTest(React.createElement(App, makeIdleAppProps()));

    await settle();
    await typeAndSubmit(view.stdin, "/help");
    await waitForRenders();

    const output = view.stdout.readAll();
    const promptPlaceholders = countPromptPlaceholders(output);
    expect(promptPlaceholders).toBeGreaterThanOrEqual(1);
    expect(promptPlaceholders).toBeLessThanOrEqual(2);
    expect(output).toContain("Commands: /clear (reset)");
  });

  it("keeps the active run open while Ctrl+C requests cancellation", () => {
    let cancelCalls = 0;
    const runningStates: boolean[] = [];
    const cancelStates: boolean[] = [];
    const spinnerMessages: Array<string | undefined> = [];
    const appendIds: string[] = [];

    handleCancelShortcut({
      appendStandalonePart: (id) => {
        appendIds.push(id);
      },
      cancelPending: false,
      exit: () => {},
      handleApproval: () => {},
      nextId: (prefix) => `${prefix}-1`,
      onCancelQuery: () => {
        cancelCalls += 1;
      },
      pendingApproval: null,
      running: true,
      setCancelPending: (value) => {
        cancelStates.push(typeof value === "function" ? value(false) : value);
      },
      setRunning: (value) => {
        runningStates.push(typeof value === "function" ? value(true) : value);
      },
      setShowCommandPalette: () => {},
      setSpinnerMessage: (value) => {
        spinnerMessages.push(typeof value === "function" ? value(undefined) : value);
      },
      showCommandPalette: false,
    });

    expect(cancelCalls).toBe(1);
    expect(cancelStates).toEqual([true]);
    expect(spinnerMessages).toEqual(["Cancelling..."]);
    expect(runningStates).toEqual([]);
    expect(appendIds).toEqual([]);

    handleCancelShortcut({
      appendStandalonePart: () => {},
      cancelPending: true,
      exit: () => {},
      handleApproval: () => {},
      nextId: (prefix) => `${prefix}-1`,
      onCancelQuery: () => {
        cancelCalls += 1;
      },
      pendingApproval: null,
      running: true,
      setCancelPending: () => {},
      setRunning: () => {},
      setShowCommandPalette: () => {},
      setSpinnerMessage: () => {},
      showCommandPalette: false,
    });

    expect(cancelCalls).toBe(1);
  });

  it("uses the same clear behavior from the command palette as /clear", async () => {
    let clearCalls = 0;
    const bus = new EventBus();
    const view = renderForTest(React.createElement(App, makeIdleAppProps({
      bus,
      onClear: () => {
        clearCalls += 1;
      },
    })));

    emitBusyStatus(bus);
    await waitForRenders();
    view.stdout.clear();
    view.stdin.write("\x0b");
    await waitForRenders();
    view.stdin.write("\r");
    await waitForRenders();

    const output = stripAnsi(view.stdout.readAll());
    const clearedFrame = output.slice(output.lastIndexOf("Context cleared."));
    expect(clearCalls).toBe(1);
    expect(output).toContain("Context cleared.");
    expect(clearedFrame).not.toContain("$0.123");
    expect(clearedFrame).not.toContain("42k/100k");
    expect(clearedFrame).not.toContain("iter 7/10");
    expect(clearedFrame).not.toContain("100k");
    expect(clearedFrame).not.toContain("iter");
  });

  it("fully resets status values when clearing with /clear", async () => {
    let clearCalls = 0;
    const bus = new EventBus();
    const view = renderForTest(React.createElement(App, makeIdleAppProps({
      bus,
      onClear: () => {
        clearCalls += 1;
      },
    })));

    emitBusyStatus(bus);
    await waitForRenders();
    view.stdout.clear();
    await typeAndSubmit(view.stdin, "/clear");
    await waitForRenders();

    const output = stripAnsi(view.stdout.readAll());
    const clearedFrame = output.slice(output.lastIndexOf("Context cleared."));
    expect(clearCalls).toBe(1);
    expect(output).toContain("Context cleared.");
    expect(clearedFrame).not.toContain("$0.123");
    expect(clearedFrame).not.toContain("42k");
    expect(clearedFrame).not.toContain("100k");
    expect(clearedFrame).not.toContain("iter");
  });
});
