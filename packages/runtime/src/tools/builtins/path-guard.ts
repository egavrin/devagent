import { existsSync, realpathSync } from "node:fs";
import { resolve, relative, isAbsolute, dirname, basename } from "node:path";
import { ToolError , extractErrorMessage } from "../../core/index.js";

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
    const message = extractErrorMessage(err);
    throw new ToolError(
      toolName,
      `Invalid ${paramName}: ${rawInput}. Failed to resolve path: ${message}`,
    );
  }

  return resolve(realExisting, ...unresolvedTail);
}

/**
 * Resolve `repoRoot` to its canonical absolute path (resolving symlinks).
 * Throws a ToolError if the path cannot be resolved.
 */
export function resolveRepoRoot(repoRoot: string): string {
  try {
    return realpathSync(resolve(repoRoot));
  } catch (err) {
    const message = extractErrorMessage(err);
    throw new ToolError(
      "resolveRepoRoot",
      `Invalid repo root: ${repoRoot}. Failed to resolve path: ${message}`,
    );
  }
}

export function normalizeRelativePath(path: string): string {
  if (!path || path === ".") {
    return ".";
  }
  return path.replaceAll("\\", "/");
}

export function resolvePathInRoot(
  rootPath: string,
  inputPath: string,
  toolName: string,
  paramName = "path",
): string {
  const rootReal = resolveRepoRoot(rootPath);

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

/**
 * Resolve a user-provided path and ensure it stays inside repoRoot.
 */
export function resolvePathInRepo(
  repoRoot: string,
  inputPath: string,
  toolName: string,
  paramName = "path",
): string {
  if (inputPath.startsWith("skill://")) {
    throw new ToolError(
      toolName,
      `Invalid ${paramName}: ${inputPath}. Skill paths are read-only and must not be used with repo-bound tools.`,
    );
  }
  return resolvePathInRoot(repoRoot, inputPath, toolName, paramName);
}
