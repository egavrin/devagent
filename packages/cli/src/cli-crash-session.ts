import { extractErrorMessage } from "@devagent/runtime";

import { dim, formatError } from "./format.js";
import type { CrashSessionReporter, CrashSessionReporterProcess, Verbosity } from "./main-types.js";

function isStderrWritable(proc: CrashSessionReporterProcess): boolean {
  return !proc.stderr.destroyed && !proc.stderr.writableEnded && !proc.stderr.writableFinished;
}

function extractErrorCode(error: unknown): string {
  return typeof error === "object" && error !== null && "code" in error
    ? String((error as { code?: unknown }).code ?? "")
    : "";
}

function isIgnoredStderrWriteError(error: unknown): boolean {
  const code = extractErrorCode(error);
  return code === "ERR_STREAM_DESTROYED" || code === "EPIPE";
}

function safeWriteStderr(proc: CrashSessionReporterProcess, chunk: string): void {
  if (!isStderrWritable(proc)) return;
  try {
    proc.stderr.write(chunk);
  } catch (error) {
    if (isIgnoredStderrWriteError(error)) return;
    throw error;
  }
}

export function createCrashSessionReporter(
  sessionId: string,
  verbosity: Verbosity,
  proc: CrashSessionReporterProcess = process,
): CrashSessionReporter {
  let printed = false;

  const printSessionId = (): void => {
    if (printed || verbosity === "quiet") return;
    safeWriteStderr(proc, dim(`[session] ${sessionId}`) + "\n");
    printed = true;
  };

  const onSigint = (): void => {
    printSessionId();
    proc.exit(130);
  };

  const onUncaughtException = (err: unknown): void => {
    safeWriteStderr(proc, formatError(`Uncaught exception: ${extractErrorMessage(err)}`) + "\n");
    printSessionId();
    proc.exit(1);
  };

  const onUnhandledRejection = (reason: unknown): void => {
    safeWriteStderr(proc, formatError(`Unhandled rejection: ${extractErrorMessage(reason)}`) + "\n");
    printSessionId();
    proc.exit(1);
  };

  proc.once("SIGINT", onSigint);
  proc.once("uncaughtException", onUncaughtException);
  proc.once("unhandledRejection", onUnhandledRejection);

  return {
    printSessionId,
    dispose: (): void => {
      proc.off("SIGINT", onSigint);
      proc.off("uncaughtException", onUncaughtException);
      proc.off("unhandledRejection", onUnhandledRejection);
    },
  };
}
