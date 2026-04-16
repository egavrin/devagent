import { copyFile, mkdir, readlink, lstat, symlink, chmod, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { execFile } from "node:child_process";
import type { IsolationMode, IsolationWorkspace } from "./types";

interface CreateIsolationWorkspaceOptions {
  readonly mode: IsolationMode;
  readonly sourceRoot: string;
  readonly targetRoot: string;
}

interface TimedWorkspaceCreationResult {
  readonly workspace: IsolationWorkspace;
  readonly durationMs: number;
}

function execGit(
  args: ReadonlyArray<string>,
  cwd: string,
): Promise<{ stdout: string; stderr: string }> {
  return new Promise((resolve, reject) => {
    execFile("git", args, { cwd, encoding: "utf-8", maxBuffer: 64 * 1024 * 1024 }, (error, stdout, stderr) => {
      if (error) {
        reject(new Error(stderr.trim() || stdout.trim() || error.message));
        return;
      }
      resolve({ stdout: stdout.trim(), stderr: stderr.trim() });
    });
  });
}

type TrackedEntry = {
  readonly mode: string;
  readonly path: string;
};

async function listTrackedEntries(sourceRoot: string): Promise<TrackedEntry[]> {
  const { stdout } = await execGit(["ls-files", "-s", "-z"], sourceRoot);
  if (stdout.length === 0) return [];
  return stdout.split("\0")
    .filter(Boolean)
    .map((entry) => {
      const match = entry.match(/^(\d+)\s+[0-9a-f]+\s+\d+\t(.+)$/);
      if (!match) {
        throw new Error(`Unexpected git ls-files output: ${entry}`);
      }
      return {
        mode: match[1]!,
        path: match[2]!,
      };
    });
}

async function copyTrackedFiles(sourceRoot: string, targetRoot: string): Promise<void> {
  const entries = await listTrackedEntries(sourceRoot);
  await mkdir(targetRoot, { recursive: true });

  for (const entry of entries) {
    const relativePath = entry.path;
    const sourcePath = join(sourceRoot, relativePath);
    const targetPath = join(targetRoot, relativePath);
    await mkdir(dirname(targetPath), { recursive: true });
    if (entry.mode === "160000") {
      await copyTrackedFiles(sourcePath, targetPath);
      continue;
    }

    const stat = await lstat(sourcePath);
    if (stat.isSymbolicLink()) {
      const target = await readlink(sourcePath);
      await symlink(target, targetPath);
      continue;
    }
    await copyFile(sourcePath, targetPath);
    await chmod(targetPath, stat.mode);
  }
}

export async function createIsolationWorkspace(
  options: CreateIsolationWorkspaceOptions,
): Promise<IsolationWorkspace> {
  if (options.mode === "temp-copy") {
    await copyTrackedFiles(options.sourceRoot, options.targetRoot);
    return {
      mode: options.mode,
      path: options.targetRoot,
      sourceRoot: options.sourceRoot,
    };
  }

  await mkdir(dirname(options.targetRoot), { recursive: true });
  await execGit(["worktree", "add", "--detach", options.targetRoot, "HEAD"], options.sourceRoot);
  return {
    mode: options.mode,
    path: options.targetRoot,
    sourceRoot: options.sourceRoot,
  };
}

export async function createIsolationWorkspaceWithTimeout(
  options: CreateIsolationWorkspaceOptions,
  timeoutMs: number,
  createWorkspace: (options: CreateIsolationWorkspaceOptions) => Promise<IsolationWorkspace> = createIsolationWorkspace,
): Promise<TimedWorkspaceCreationResult> {
  const startedAt = Date.now();
  const workspacePromise = createWorkspace(options);
  const timeoutPromise = new Promise<never>((_, reject) => {
    const timer = setTimeout(() => {
      reject(new Error(
        `setup failed during isolation after ${timeoutMs}ms while creating a ${options.mode} workspace from ${options.sourceRoot}`,
      ));
    }, timeoutMs);
    workspacePromise.finally(() => clearTimeout(timer)).catch(() => clearTimeout(timer));
  });
  const workspace = await Promise.race([workspacePromise, timeoutPromise]);
  return {
    workspace,
    durationMs: Date.now() - startedAt,
  };
}

export async function destroyIsolationWorkspace(
  workspace: IsolationWorkspace,
): Promise<void> {
  if (workspace.mode === "worktree") {
    await execGit(["worktree", "remove", "--force", workspace.path], workspace.sourceRoot);
    return;
  }
  await rm(workspace.path, { recursive: true, force: true });
}

export async function ensureGitIdentity(repoRoot: string): Promise<void> {
  await execGit(["config", "user.email", "devagent-validation@example.com"], repoRoot);
  await execGit(["config", "user.name", "DevAgent Validation"], repoRoot);
}

export async function initializeTempCopyRepository(repoRoot: string): Promise<void> {
  await execGit(["init"], repoRoot);
  await ensureGitIdentity(repoRoot);
}

export async function commitWorkspaceState(
  repoRoot: string,
  message: string,
): Promise<void> {
  await execGit(["add", "."], repoRoot);
  try {
    await execGit(["commit", "-m", message], repoRoot);
  } catch (error) {
    const messageText = error instanceof Error ? error.message : String(error);
    if (messageText.includes("nothing to commit")) {
      return;
    }
    throw error;
  }
}

export async function captureGitOutputs(
  repoRoot: string,
): Promise<{ repoStatus: string; repoDiff: string; repoDiffCached: string }> {
  const [status, diff, diffCached] = await Promise.all([
    execGit(["status", "--short"], repoRoot),
    execGit(["diff", "--"], repoRoot).catch(() => ({ stdout: "", stderr: "" })),
    execGit(["diff", "--cached", "--"], repoRoot).catch(() => ({ stdout: "", stderr: "" })),
  ]);
  return {
    repoStatus: status.stdout,
    repoDiff: diff.stdout,
    repoDiffCached: diffCached.stdout,
  };
}
