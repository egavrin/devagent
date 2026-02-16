import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { CheckpointManager } from "./checkpoints.js";
import { EventBus } from "@devagent/core";
import { mkdtempSync, writeFileSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { execSync } from "node:child_process";

// ─── Helpers ────────────────────────────────────────────────

function createTestRepo(): string {
  const dir = mkdtempSync(join(tmpdir(), "devagent-cp-test-"));
  execSync("git init", { cwd: dir });
  execSync("git config user.name Test", { cwd: dir });
  execSync("git config user.email test@test.com", { cwd: dir });
  writeFileSync(join(dir, "file.txt"), "initial content");
  execSync("git add -A && git commit -m initial", { cwd: dir });
  return dir;
}

describe("CheckpointManager", () => {
  let testDir: string;
  let bus: EventBus;

  beforeEach(() => {
    testDir = createTestRepo();
    bus = new EventBus();
  });

  afterEach(() => {
    rmSync(testDir, { recursive: true, force: true });
  });

  it("does nothing when disabled", () => {
    const mgr = new CheckpointManager({
      repoRoot: testDir,
      bus,
      enabled: false,
    });
    mgr.init();
    const cp = mgr.create("test", "write_file");
    expect(cp).toBeNull();
    expect(mgr.list().length).toBe(0);
  });

  it("initializes shadow git repo", () => {
    const mgr = new CheckpointManager({
      repoRoot: testDir,
      bus,
      enabled: true,
    });
    mgr.init();
    // Shadow dir should exist
    const shadowGit = join(testDir, ".devagent", "checkpoints", ".git");
    const { existsSync } = require("node:fs");
    expect(existsSync(shadowGit)).toBe(true);
  });

  it("creates a checkpoint after file change", () => {
    const mgr = new CheckpointManager({
      repoRoot: testDir,
      bus,
      enabled: true,
    });
    mgr.init();

    // Modify a file
    writeFileSync(join(testDir, "file.txt"), "modified content");

    const cp = mgr.create("modified file.txt", "write_file");
    expect(cp).not.toBeNull();
    expect(cp!.toolName).toBe("write_file");
    expect(cp!.description).toBe("modified file.txt");
    expect(cp!.commitHash).toBeTruthy();
  });

  it("tracks multiple checkpoints", () => {
    const mgr = new CheckpointManager({
      repoRoot: testDir,
      bus,
      enabled: true,
    });
    mgr.init();

    writeFileSync(join(testDir, "file.txt"), "version 1");
    mgr.create("first edit", "write_file");

    writeFileSync(join(testDir, "file.txt"), "version 2");
    mgr.create("second edit", "write_file");

    const checkpoints = mgr.list();
    expect(checkpoints.length).toBe(2);
    expect(checkpoints[0]!.description).toBe("first edit");
    expect(checkpoints[1]!.description).toBe("second edit");
  });

  it("emits checkpoint:created event", () => {
    const events: Array<{ id: string; description: string }> = [];
    bus.on("checkpoint:created", (e) => events.push(e));

    const mgr = new CheckpointManager({
      repoRoot: testDir,
      bus,
      enabled: true,
    });
    mgr.init();

    writeFileSync(join(testDir, "file.txt"), "changed");
    mgr.create("test checkpoint", "write_file");

    expect(events.length).toBe(1);
    expect(events[0]!.description).toBe("test checkpoint");
  });

  it("returns null when no changes to checkpoint", () => {
    const mgr = new CheckpointManager({
      repoRoot: testDir,
      bus,
      enabled: true,
    });
    mgr.init();

    // No file changes — should return null
    const cp = mgr.create("no changes", "write_file");
    expect(cp).toBeNull();
  });

  it("computes diff between checkpoints", () => {
    const mgr = new CheckpointManager({
      repoRoot: testDir,
      bus,
      enabled: true,
    });
    mgr.init();

    writeFileSync(join(testDir, "file.txt"), "version 1");
    const cp1 = mgr.create("first", "write_file");

    writeFileSync(join(testDir, "file.txt"), "version 2");
    const cp2 = mgr.create("second", "write_file");

    expect(cp1).not.toBeNull();
    expect(cp2).not.toBeNull();

    const d = mgr.diff(cp1!.id, cp2!.id);
    expect(d).toContain("version 1");
    expect(d).toContain("version 2");
  });
});
