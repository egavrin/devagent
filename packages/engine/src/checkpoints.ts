/**
 * Git-based workspace checkpoints.
 * Creates snapshots after each mutating tool execution.
 * Uses a shadow git repo separate from the user's repo.
 * Enables compare & restore to any point.
 */

import { execSync } from "node:child_process";
import { existsSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import type { EventBus } from "@devagent/core";

// ─── Types ──────────────────────────────────────────────────

export interface Checkpoint {
  readonly id: string;
  readonly commitHash: string;
  readonly description: string;
  readonly timestamp: number;
  readonly toolName: string;
}

export interface CheckpointManagerOptions {
  readonly repoRoot: string;
  readonly bus: EventBus;
  readonly enabled: boolean;
}

// ─── Checkpoint Manager ─────────────────────────────────────

export class CheckpointManager {
  private readonly repoRoot: string;
  private readonly shadowDir: string;
  private readonly bus: EventBus;
  private readonly enabled: boolean;
  private readonly checkpoints: Checkpoint[] = [];
  private initialized = false;

  constructor(options: CheckpointManagerOptions) {
    this.repoRoot = options.repoRoot;
    this.shadowDir = join(options.repoRoot, ".devagent", "checkpoints");
    this.bus = options.bus;
    this.enabled = options.enabled;
  }

  /**
   * Initialize the shadow git repo. Idempotent — safe to call multiple times.
   */
  init(): void {
    if (!this.enabled || this.initialized) return;

    try {
      // Create shadow directory
      if (!existsSync(this.shadowDir)) {
        mkdirSync(this.shadowDir, { recursive: true });
      }

      // Init shadow repo if needed
      if (!existsSync(join(this.shadowDir, ".git"))) {
        this.git("init");
        this.git("config user.name DevAgent");
        this.git("config user.email devagent@local");
      }

      // Create initial snapshot
      this.snapshot("initial", "checkpoint:init");

      this.initialized = true;
    } catch (err) {
      // Fail fast — surface the error, don't silently degrade
      const message = err instanceof Error ? err.message : String(err);
      this.bus.emit("error", {
        message: `Checkpoint init failed: ${message}`,
        code: "CHECKPOINT_INIT_ERROR",
        fatal: false,
      });
    }
  }

  /**
   * Create a snapshot of the current working directory.
   */
  create(description: string, toolName: string): Checkpoint | null {
    if (!this.enabled) return null;

    if (!this.initialized) {
      this.init();
    }

    try {
      const commitHash = this.snapshot(description, toolName);
      if (!commitHash) return null;

      const checkpoint: Checkpoint = {
        id: `cp-${this.checkpoints.length}`,
        commitHash,
        description,
        timestamp: Date.now(),
        toolName,
      };

      this.checkpoints.push(checkpoint);

      this.bus.emit("checkpoint:created", {
        id: checkpoint.id,
        description: checkpoint.description,
        timestamp: checkpoint.timestamp,
      });

      return checkpoint;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.bus.emit("error", {
        message: `Checkpoint create failed: ${message}`,
        code: "CHECKPOINT_CREATE_ERROR",
        fatal: false,
      });
      return null;
    }
  }

  /**
   * List all checkpoints in order.
   */
  list(): ReadonlyArray<Checkpoint> {
    return this.checkpoints;
  }

  /**
   * Get diff between two checkpoints (or between a checkpoint and current state).
   */
  diff(fromId: string, toId?: string): string {
    if (!this.enabled) return "";

    const from = this.checkpoints.find((cp) => cp.id === fromId);
    if (!from) {
      throw new Error(`Checkpoint not found: ${fromId}`);
    }

    try {
      if (toId) {
        const to = this.checkpoints.find((cp) => cp.id === toId);
        if (!to) {
          throw new Error(`Checkpoint not found: ${toId}`);
        }
        return this.git(`diff ${from.commitHash} ${to.commitHash}`);
      }

      // Diff from checkpoint to current working directory
      // First, create a temp snapshot to compare against
      this.syncToShadow();
      return this.git(`diff ${from.commitHash}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return `Error computing diff: ${message}`;
    }
  }

  /**
   * Restore workspace to a specific checkpoint.
   */
  restore(checkpointId: string): boolean {
    if (!this.enabled) return false;

    const checkpoint = this.checkpoints.find((cp) => cp.id === checkpointId);
    if (!checkpoint) {
      throw new Error(`Checkpoint not found: ${checkpointId}`);
    }

    try {
      // Get the file list from the checkpoint commit
      const files = this.git(`diff --name-only ${checkpoint.commitHash} HEAD`).trim();
      if (!files) return true; // No differences

      // Reset shadow work tree to the exact checkpoint snapshot.
      // `reset --hard` restores tracked files and removes tracked files that
      // were added after the target checkpoint; `clean -fd` removes leftovers.
      this.git(`reset --hard ${checkpoint.commitHash}`);
      this.git("clean -fd");

      // Copy restored files back to the repo root
      this.syncFromShadow();

      return true;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      this.bus.emit("error", {
        message: `Checkpoint restore failed: ${message}`,
        code: "CHECKPOINT_RESTORE_ERROR",
        fatal: false,
      });
      return false;
    }
  }

  // ─── Private ────────────────────────────────────────────────

  /**
   * Sync working directory files into shadow repo and commit.
   */
  private snapshot(description: string, toolName: string): string | null {
    this.syncToShadow();

    // Stage all changes
    this.git("add -A");

    // Check if there are changes to commit
    const status = this.git("status --porcelain");
    if (!status.trim()) {
      return null; // No changes
    }

    // Commit
    const message = `[${toolName}] ${description}`;
    this.git(`commit -m "${message.replace(/"/g, '\\"')}" --allow-empty`);

    // Get commit hash
    return this.git("rev-parse HEAD").trim();
  }

  /**
   * Copy tracked files from repo root to shadow directory.
   * Always excludes .devagent, .git, node_modules, dist to prevent recursion.
   */
  private syncToShadow(): void {
    try {
      execSync(
        `rsync -ac --delete --exclude='.git' --exclude='.devagent' --exclude='node_modules' --exclude='dist' "${this.repoRoot}/" "${this.shadowDir}/"`,
        { encoding: "utf-8", timeout: 30_000 },
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(`syncToShadow failed: ${message}`);
    }
  }

  /**
   * Copy files from shadow directory back to repo root.
   */
  private syncFromShadow(): void {
    try {
      execSync(
        `rsync -a --delete --exclude='.git' --exclude='.devagent' --exclude='node_modules' --exclude='dist' "${this.shadowDir}/" "${this.repoRoot}/"`,
        { encoding: "utf-8", timeout: 30_000 },
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      throw new Error(`syncFromShadow failed: ${message}`);
    }
  }

  /**
   * Run a git command in the shadow directory.
   */
  private git(command: string): string {
    return execSync(`git ${command}`, {
      cwd: this.shadowDir,
      encoding: "utf-8",
      timeout: 10_000,
    });
  }
}
