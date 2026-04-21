import type { SessionState } from "./session-state.js";
import { normalizeRepoPath } from "./task-loop-paths.js";
import { parseToolScriptStepsArg } from "./tool-script.js";

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
  if (captureToolScriptPaths(sessionState, toolCall)) return;
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

function captureToolScriptPaths(sessionState: SessionState, toolCall: PendingToolCall): boolean {
  if (toolCall.name !== "execute_tool_script") return false;
  const steps = parseToolScriptStepsArg(toolCall.arguments["steps"]);
  if (!steps) return true;
  for (const step of steps) {
    const path = getReviewScopeStepPath(step.tool, step.args);
    if (path) sessionState.recordModifiedFile(path);
  }
  return true;
}

function getReviewScopeStepPath(tool: string, args: Record<string, unknown>): string | null {
  if (tool !== "git_diff" && tool !== "read_file") return null;
  const path = args["path"];
  if (typeof path !== "string" || path.trim().length === 0) return null;
  return normalizeRepoPath(path);
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
