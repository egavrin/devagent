import { existsSync } from "node:fs";
import { relative } from "node:path";
import { ToolError } from "../../core/errors.js";
import type { SkillAccessManager } from "../../core/skills/index.js";
import {
  normalizeRelativePath,
  resolvePathInRepo,
  resolvePathInRoot,
  resolveRepoRoot,
} from "./path-guard.js";

export interface ReadonlyToolOptions {
  readonly skillAccess?: SkillAccessManager;
}

export interface ResolvedReadonlyPath {
  readonly kind: "repo" | "skill";
  readonly rootPath: string;
  readonly resolvedPath: string;
  readonly displayPath: string;
}

const SKILL_URI_PREFIX = "skill://";

export function resolveReadonlyPath(
  repoRoot: string,
  rawPath: string,
  toolName: string,
  options?: ReadonlyToolOptions,
): ResolvedReadonlyPath {
  const parsed = parseSkillUri(rawPath);
  if (!parsed) {
    return {
      kind: "repo",
      rootPath: resolveRepoRoot(repoRoot),
      resolvedPath: resolvePathInRepo(repoRoot, rawPath, toolName),
      displayPath: normalizeRelativePath(rawPath),
    };
  }

  if (!options?.skillAccess) {
    throw new ToolError(
      toolName,
      `Skill paths are not enabled in this context: ${rawPath}`,
    );
  }

  let skillDir: string;
  let supportRootPath: string | undefined;
  try {
    const metadata = options.skillAccess.requireUnlocked(parsed.skillName);
    skillDir = metadata.dirPath;
    supportRootPath = metadata.supportRootPath;
  } catch (err) {
    const message = err instanceof Error ? err.message : String(err);
    throw new ToolError(toolName, message);
  }

  const relativePath = parsed.path.length > 0 ? parsed.path : ".";
  const resolved = resolveSkillPath(
    skillDir,
    supportRootPath,
    relativePath,
    toolName,
  );
  return {
    kind: "skill",
    rootPath: resolved.rootPath,
    resolvedPath: resolved.resolvedPath,
    displayPath: `skill://${parsed.skillName}/${normalizeRelativePath(parsed.path)}`,
  };
}

export function toRootRelativePath(rootPath: string, fullPath: string): string {
  return normalizeRelativePath(relative(rootPath, fullPath));
}

function parseSkillUri(rawPath: string): { skillName: string; path: string } | null {
  if (!rawPath.startsWith(SKILL_URI_PREFIX)) {
    return null;
  }

  const rest = rawPath.slice(SKILL_URI_PREFIX.length);
  const slashIndex = rest.indexOf("/");
  const skillName = slashIndex === -1 ? rest : rest.slice(0, slashIndex);
  const innerPath = slashIndex === -1 ? "" : rest.slice(slashIndex + 1);

  if (!skillName) {
    throw new ToolError(
      "skill_path",
      `Invalid skill path: ${rawPath}. Expected skill://<skill-name>/...`,
    );
  }

  return {
    skillName,
    path: innerPath,
  };
}

function resolveSkillPath(
  wrapperDirPath: string,
  supportRootPath: string | undefined,
  relativePath: string,
  toolName: string,
): { rootPath: string; resolvedPath: string } {
  const supportRootExists = Boolean(
    supportRootPath &&
    supportRootPath !== wrapperDirPath &&
    existsSync(supportRootPath),
  );

  if (relativePath === "." && supportRootExists && supportRootPath) {
    return {
      rootPath: resolveRepoRoot(supportRootPath),
      resolvedPath: resolvePathInRoot(supportRootPath, ".", toolName),
    };
  }

  const rootCandidates = [wrapperDirPath];
  if (supportRootExists && supportRootPath) {
    rootCandidates.push(supportRootPath);
  }

  let fallback: { rootPath: string; resolvedPath: string } | null = null;
  for (const rootPath of rootCandidates) {
    const resolvedPath = resolvePathInRoot(rootPath, relativePath, toolName);
    const resolvedRootPath = resolveRepoRoot(rootPath);
    if (!fallback) {
      fallback = { rootPath: resolvedRootPath, resolvedPath };
    }
    if (existsSync(resolvedPath)) {
      return { rootPath: resolvedRootPath, resolvedPath };
    }
  }

  return fallback ?? {
    rootPath: resolveRepoRoot(wrapperDirPath),
    resolvedPath: resolvePathInRoot(wrapperDirPath, relativePath, toolName),
  };
}
