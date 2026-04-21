import { existsSync } from "node:fs";
import { join, resolve } from "node:path";

export function findProjectRoot(startDir?: string): string | null {
  const start = resolve(startDir ?? process.cwd());
  const root = resolve("/");
  let gitFallback: string | null = null;
  let packageJsonFallback: string | null = null;
  let dir = start;

  while (dir !== root) {
    if (existsSync(join(dir, ".agents", "skills"))) return dir;
    packageJsonFallback ??= getPackageJsonFallback(dir, start);
    gitFallback ??= getGitFallback(dir);
    const parent = resolve(dir, "..");
    if (parent === dir) break;
    dir = parent;
  }

  return packageJsonFallback ?? gitFallback ?? null;
}

function getPackageJsonFallback(dir: string, start: string): string | null {
  return dir === start && existsSync(join(dir, "package.json")) ? dir : null;
}

function getGitFallback(dir: string): string | null {
  return existsSync(join(dir, ".git")) ? dir : null;
}
