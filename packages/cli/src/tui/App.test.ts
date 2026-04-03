import { PassThrough, Writable } from "node:stream";
import React, { useEffect, useMemo, useRef } from "react";
import { render, Static, Text } from "ink";
import { EventBus } from "@devagent/runtime";
import { afterEach, describe, expect, it } from "vitest";

import { ITERATION_LIMIT_NOTICE, TranscriptView } from "./App.js";
import { useAgentLog } from "./useAgentLog.js";
import type { LogEntry } from "./shared.js";

class TestInput extends PassThrough {
  readonly isTTY = true;

  setRawMode(_value: boolean): void {}
}

class TestOutput extends Writable {
  readonly isTTY = true;
  readonly columns = 120;
  readonly rows = 40;
  private readonly chunks: string[] = [];

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

function renderForTest(node: React.ReactElement): {
  readonly stdout: TestOutput;
  readonly rerender: (tree: React.ReactElement) => void;
  readonly unmount: () => void;
  readonly cleanup: () => void;
} {
  const stdout = new TestOutput();
  const stdin = new TestInput();
  const stderr = new TestOutput();
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
    rerender: instance.rerender,
    unmount: instance.unmount,
    cleanup: instance.cleanup,
  };
}

function AgentLogHarness({ entries }: { readonly entries: ReadonlyArray<string> }): React.ReactElement {
  const bus = useMemo(() => new EventBus(), []);
  const { log, addLog } = useAgentLog({
    bus,
    model: "test-model",
    approvalMode: "default",
  });
  const emittedCount = useRef(0);

  useEffect(() => {
    for (let index = emittedCount.current; index < entries.length; index++) {
      addLog({
        id: `entry-${index + 1}`,
        type: "info",
        data: entries[index]!,
      });
    }
    emittedCount.current = entries.length;
  }, [entries, addLog]);

  return React.createElement(
    Static,
    { items: log },
    (entry: LogEntry) => React.createElement(Text, { key: entry.id }, String(entry.data)),
  );
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

  it("renders the first submitted user message after the welcome screen disappears", async () => {
    const view = renderForTest(
      React.createElement(TranscriptView, {
        showWelcome: true,
        log: [],
        model: "test-model",
      }),
    );

    await settle();

    view.rerender(
      React.createElement(TranscriptView, {
        showWelcome: false,
        log: [{ id: "user-1", type: "info", data: "> hello world" }],
        model: "test-model",
      }),
    );

    await settle();

    expect(view.stdout.readAll()).toContain("> hello world");
  });

  it("keeps rendering new transcript entries after the log grows past fifty items", async () => {
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
});
