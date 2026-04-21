import { describe, expect, it, vi } from "vitest";

import { createCrashSessionReporter } from "./main.js";

type CrashEvent = "SIGINT" | "uncaughtException" | "unhandledRejection";

function createMockCrashProcess() {
  const write = vi.fn(() => true);
  const onceHandlers = new Map<string, (...args: any[]) => void>();
  const exit = vi.fn(() => {
    throw new Error("exit");
  }) as unknown as (code?: number) => never;
  const proc = {
    stderr: { write },
    once: (event: CrashEvent, listener: (...args: any[]) => void) => {
      onceHandlers.set(event, listener);
    },
    off: vi.fn(),
    exit,
  };
  return { exit, onceHandlers, proc, write };
}

describe("createCrashSessionReporter", () => {
  it("prints session id on SIGINT and exits 130", () => {
    const { exit, onceHandlers, proc, write } = createMockCrashProcess();

    createCrashSessionReporter("s-123", "normal", proc);

    expect(() => onceHandlers.get("SIGINT")!()).toThrow("exit");
    expect(exit).toHaveBeenCalledWith(130);
    expect(write).toHaveBeenCalledWith(expect.stringContaining("[session] s-123"));
  });

  it("prints crash error and session id on uncaughtException", () => {
    const { exit, onceHandlers, proc, write } = createMockCrashProcess();

    createCrashSessionReporter("s-456", "normal", proc);

    expect(() => onceHandlers.get("uncaughtException")!(new Error("boom"))).toThrow("exit");
    expect(exit).toHaveBeenCalledWith(1);
    expect(write).toHaveBeenCalledWith(expect.stringContaining("Uncaught exception: boom"));
    expect(write).toHaveBeenCalledWith(expect.stringContaining("[session] s-456"));
  });

  it("prints crash error and session id on unhandledRejection", () => {
    const { exit, onceHandlers, proc, write } = createMockCrashProcess();

    createCrashSessionReporter("s-789", "normal", proc);

    expect(() => onceHandlers.get("unhandledRejection")!("bad")).toThrow("exit");
    expect(exit).toHaveBeenCalledWith(1);
    expect(write).toHaveBeenCalledWith(expect.stringContaining("Unhandled rejection: bad"));
    expect(write).toHaveBeenCalledWith(expect.stringContaining("[session] s-789"));
  });

  it("does not print in quiet mode", () => {
    const { onceHandlers, proc, write } = createMockCrashProcess();

    createCrashSessionReporter("s-q", "quiet", proc);

    expect(() => onceHandlers.get("SIGINT")!()).toThrow("exit");
    expect(write).not.toHaveBeenCalledWith(expect.stringContaining("[session] s-q"));
  });

  it("printSessionId is idempotent and dispose unregisters handlers", () => {
    const write = vi.fn(() => true);
    const off = vi.fn();

    const proc = {
      stderr: { write },
      once: vi.fn(),
      off,
      exit: vi.fn(() => {
        throw new Error("exit");
      }) as unknown as (code?: number) => never,
    };

    const reporter = createCrashSessionReporter("s-once", "normal", proc);
    reporter.printSessionId();
    reporter.printSessionId();
    reporter.dispose();

    const sessionWrites = write.mock.calls.filter((args) => String(args[0]).includes("[session] s-once"));
    expect(sessionWrites).toHaveLength(1);
    expect(off).toHaveBeenCalledTimes(3);
  });

  it("suppresses destroyed-stream errors while still exiting on SIGINT", () => {
    const onceHandlers = new Map<string, (...args: any[]) => void>();
    const exit = vi.fn(() => {
      throw new Error("exit");
    }) as unknown as (code?: number) => never;
    const proc = {
      stderr: {
        destroyed: false,
        writableEnded: false,
        writableFinished: false,
        write: vi.fn(() => {
          throw Object.assign(new Error("destroyed"), { code: "ERR_STREAM_DESTROYED" });
        }),
      },
      once: (event: "SIGINT" | "uncaughtException" | "unhandledRejection", listener: (...args: any[]) => void) => {
        onceHandlers.set(event, listener);
      },
      off: vi.fn(),
      exit,
    };

    createCrashSessionReporter("s-destroyed", "normal", proc);

    expect(() => onceHandlers.get("SIGINT")!()).toThrow("exit");
    expect(exit).toHaveBeenCalledWith(130);
  });
});
