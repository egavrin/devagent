import { render } from "ink";
import { PassThrough, Writable } from "node:stream";

import type React from "react";

export class TestInput extends PassThrough {
  readonly isTTY = true;

  setRawMode(_value: boolean): void {}

  ref(): this {
    return this;
  }

  unref(): this {
    return this;
  }
}

export class TestOutput extends Writable {
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
    const text = typeof chunk === "string" ? chunk : Buffer.from(chunk).toString("utf8");
    this.chunks.push(text);
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

export function cleanupRenderedInstances(): void {
  while (instances.length > 0) {
    const instance = instances.pop();
    instance?.unmount();
    instance?.cleanup();
  }
}

export async function settle(): Promise<void> {
  await new Promise((resolve) => setTimeout(resolve, 20));
}

export async function waitForRenders(cycles: number = 6): Promise<void> {
  for (let index = 0; index < cycles; index++) {
    await settle();
  }
}

export function stripAnsi(text: string): string {
  return text.replace(/\x1b\[[0-9;]*m/g, "");
}

export function countPromptPlaceholders(text: string): number {
  return (stripAnsi(text).match(/Ask anything…/g) ?? []).length;
}

export async function typeAndSubmit(stdin: TestInput, text: string): Promise<void> {
  stdin.write(text);
  await settle();
  stdin.write("\r");
}

export async function insertModifiedReturn(stdin: TestInput): Promise<void> {
  stdin.write("\x1b\r");
  await settle();
}

export function renderForTest(
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
