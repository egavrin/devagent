import { existsSync, realpathSync } from "node:fs";
import { resolve, relative, isAbsolute, dirname, basename } from "node:path";
import { ToolError } from "@devagent/core";

function resolveWithSymlinkAwareness(
  targetPath: string,
  toolName: string,
  paramName: string,
  rawInput: string,
): string {
  let current = targetPath;
  const unresolvedTail: string[] = [];

  // Walk up until we find an existing ancestor, tracking missing segments.
  while (!existsSync(current)) {
    const parent = dirname(current);
    if (parent === current) {
      throw new ToolError(
        toolName,
        `Invalid ${paramName}: ${rawInput}. Path must stay within repo root.`,
      );
    }
    unresolvedTail.unshift(basename(current));
    current = parent;
  }

  let realExisting: string;
  try {
    realExisting = realpathSync(current);
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new ToolError(
      toolName,
      `Invalid ${paramName}: ${rawInput}. Failed to resolve path: ${message}`,
    );
  }

  return resolve(realExisting, ...unresolvedTail);
}

/**
 * Resolve a user-provided path and ensure it stays inside repoRoot.
 */
export function resolvePathInRepo(
  repoRoot: string,
  inputPath: string,
  toolName: string,
  paramName = "path",
): string {
  let rootReal: string;
  try {
    rootReal = realpathSync(resolve(repoRoot));
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new ToolError(
      toolName,
      `Invalid repo root: ${repoRoot}. Failed to resolve path: ${message}`,
    );
  }

  const candidatePath = resolve(rootReal, inputPath);
  const resolvedPath = resolveWithSymlinkAwareness(
    candidatePath,
    toolName,
    paramName,
    inputPath,
  );
  const rel = relative(rootReal, resolvedPath);

  if (rel === "" || (!rel.startsWith("..") && !isAbsolute(rel))) {
    return resolvedPath;
  }

  throw new ToolError(
    toolName,
    `Invalid ${paramName}: ${inputPath}. Path must stay within repo root.`,
  );
}
