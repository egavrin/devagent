import { validateTaskExecutionRequest } from "@devagent-sdk/validation";
import { execFile, type ExecFileException } from "node:child_process";
import { constants } from "node:fs";
import { access, readFile } from "node:fs/promises";
import { delimiter, join, resolve } from "node:path";

import type {
  ArtifactKind,
  ArtifactVariant,
  TaskExecutionRequest,
  TaskExecutionResult,
} from "@devagent-sdk/types";

interface ExecuteArgs {
  requestPath: string;
  artifactDir: string;
}


export interface VerifyCommandRun {
  command: string;
  status: "passed" | "failed";
  exitCode: number;
  stdout: string;
  stderr: string;
}

export type TaskLoopEnvelope = {
  result?: string;
  responseText?: string;
};

export type StructuredArtifactEnvelope = {
  structured: unknown;
  rendered: string;
};

export type PendingArtifact = {
  kind: ArtifactKind;
  fileName: string;
  content: string;
  mimeType: string;
  variant?: ArtifactVariant;
};

export function classifyFailedWorkflowResult(
  result: { outcomeReason?: TaskExecutionResult["outcomeReason"] },
): TaskExecutionResult["outcomeReason"] | undefined {
  if (result.outcomeReason) {
    return result.outcomeReason;
  }
  return undefined;
}

export function classifySuccessArtifactOutcome(content: string): Pick<TaskExecutionResult, "outcome" | "outcomeReason"> {
  if (content.trim().length === 0) {
    return {
      outcome: "no_progress",
      outcomeReason: "empty_artifact",
    };
  }
  return {
    outcome: "completed",
  };
}

const PREFERRED_VERIFY_PATHS = [
  "/opt/homebrew/bin",
  "/opt/homebrew/sbin",
  "/usr/local/bin",
  "/usr/local/sbin",
  "/usr/bin",
  "/usr/sbin",
  "/bin",
  "/sbin",
];

const LEADING_NODE_COMMAND = /^(\s*(?:[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|[^\s]+)\s+)*)((?:\/usr\/bin\/env|env)\s+)?node(?=\s|$)/;

function buildVerifyNodeSearchPath(currentPath = process.env["PATH"] ?? ""): string[] {
  return Array.from(new Set([
    ...currentPath.split(delimiter).filter(Boolean),
    ...PREFERRED_VERIFY_PATHS,
  ]));
}

function quoteShellArgument(value: string): string {
  return `'${value.replaceAll("'", `'\"'\"'`)}'`;
}

async function isExecutable(path: string): Promise<boolean> {
  try {
    await access(path, constants.X_OK);
    return true;
  } catch {
    return false;
  }
}

async function isRealNodeBinary(path: string): Promise<boolean> {
  return await new Promise((resolveCommand) => {
    execFile(
      path,
      ["-p", "process.versions && process.versions.bun ? 'bun' : ((process.release && process.release.name) || '')"],
      {
        encoding: "utf8",
        timeout: 5_000,
      },
      (error: ExecFileException | null, stdout: string) => {
        resolveCommand(!error && stdout.trim() === "node");
      },
    );
  });
}

export async function resolveVerifyNodeBinary(currentPath = process.env["PATH"] ?? ""): Promise<string | null> {
  for (const entry of buildVerifyNodeSearchPath(currentPath)) {
    const candidate = join(entry, "node");
    if (!await isExecutable(candidate)) {
      continue;
    }
    if (await isRealNodeBinary(candidate)) {
      return candidate;
    }
  }

  return null;
}

export function rewriteVerifyCommand(command: string, nodeBinary: string): string {
  if (!LEADING_NODE_COMMAND.test(command)) {
    return command;
  }

  return command.replace(LEADING_NODE_COMMAND, (_, prefix: string) => `${prefix}${quoteShellArgument(nodeBinary)}`);
}

function fakeResponseEnvKey(taskType: TaskExecutionRequest["taskType"]): string {
  return `DEVAGENT_EXECUTOR_FAKE_RESPONSE_${taskType.replace(/[^A-Za-z0-9]+/g, "_").toUpperCase()}`;
}

export function readFakeTaskResponse(taskType: TaskExecutionRequest["taskType"]): string | undefined {
  return process.env[fakeResponseEnvKey(taskType)] ?? process.env["DEVAGENT_EXECUTOR_FAKE_RESPONSE"];
}

export function parseExecuteArgs(argv: string[]): ExecuteArgs | null {
  const args = argv.slice(2);
  if (args[0] !== "execute") return null;

  let requestPath: string | null = null;
  let artifactDir: string | null = null;

  for (let i = 1; i < args.length; i++) {
    const arg = args[i]!;
    if (arg === "--request" && args[i + 1]) {
      requestPath = args[++i]!;
      continue;
    }
    if (arg === "--artifact-dir" && args[i + 1]) {
      artifactDir = args[++i]!;
      continue;
    }
  }

  if (!requestPath || !artifactDir) {
    throw new Error("Usage: devagent execute --request <file> --artifact-dir <dir>");
  }

  return {
    requestPath: resolve(requestPath),
    artifactDir: resolve(artifactDir),
  };
}

export async function loadTaskExecutionRequest(requestPath: string): Promise<TaskExecutionRequest> {
  const parsed = JSON.parse(await readFile(requestPath, "utf-8")) as unknown;
  return validateTaskExecutionRequest(parsed);
}
