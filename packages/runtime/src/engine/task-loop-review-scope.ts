import type { SessionState } from "./session-state.js";
import { normalizeRepoPath } from "./task-loop-paths.js";

interface PendingToolCall {
  readonly name: string;
  readonly arguments: Record<string, unknown>;
  readonly callId: string;
}

export function captureTaskLoopReviewScopeFiles(
  sessionState: SessionState | null,
  toolCall: PendingToolCall,
  originalOutput: string,
): void {
  if (!sessionState) return;
  if (captureGitDiffPath(sessionState, toolCall)) return;
  captureGitDiffNameOnlyOutput(sessionState, toolCall, originalOutput);
}

function captureGitDiffPath(sessionState: SessionState, toolCall: PendingToolCall): boolean {
  if (toolCall.name !== "git_diff") return false;
  const path = toolCall.arguments["path"];
  if (typeof path === "string" && path.trim().length > 0) {
    sessionState.recordModifiedFile(normalizeRepoPath(path));
  }
  return true;
}

function captureGitDiffNameOnlyOutput(
  sessionState: SessionState,
  toolCall: PendingToolCall,
  originalOutput: string,
): void {
  if (toolCall.name !== "run_command") return;
  const command = toolCall.arguments["command"];
  if (typeof command !== "string" || !isGitDiffNameOnlyCommand(command)) return;
  for (const file of parseGitNameOnlyOutput(originalOutput)) {
    sessionState.recordModifiedFile(file);
  }
}

function isGitDiffNameOnlyCommand(command: string): boolean {
  return /\bgit\s+diff\b/.test(command) && /\b--name-only\b/.test(command);
}

function parseGitNameOnlyOutput(output: string): string[] {
  return output
    .split(/\r?\n/)
    .map((line) => normalizeRepoPath(line))
    .filter((line) => line.length > 0)
    .filter((line) => !line.startsWith("fatal:"))
    .filter((line) => !line.startsWith("warning:"));
}
