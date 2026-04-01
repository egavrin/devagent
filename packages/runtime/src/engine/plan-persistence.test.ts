import { describe, it, expect, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { writePlanFile, readPlanFile, getPlanFilePath } from "./plan-persistence.js";
import type { PlanStep } from "./plan-tool.js";

describe("plan-persistence", () => {
  const tempDirs: string[] = [];

  function makeTempDir(): string {
    const dir = mkdtempSync(join(tmpdir(), "plan-test-"));
    tempDirs.push(dir);
    return dir;
  }

  afterEach(() => {
    for (const dir of tempDirs) {
      rmSync(dir, { recursive: true, force: true });
    }
    tempDirs.length = 0;
  });

  it("write then read round-trip preserves steps", () => {
    const repoRoot = makeTempDir();
    const sessionId = "test-session-1";
    const steps: PlanStep[] = [
      { description: "Analyze codebase", status: "completed", lastTransitionIteration: 1 },
      { description: "Implement fix", status: "in_progress", lastTransitionIteration: 3 },
      { description: "Write tests", status: "pending" },
    ];

    writePlanFile(sessionId, repoRoot, steps);
    const result = readPlanFile(sessionId, repoRoot);

    expect(result).not.toBeNull();
    expect(result).toHaveLength(3);
    expect(result![0]!.description).toBe("Analyze codebase");
    expect(result![0]!.status).toBe("completed");
    expect(result![1]!.description).toBe("Implement fix");
    expect(result![1]!.status).toBe("in_progress");
    expect(result![2]!.description).toBe("Write tests");
    expect(result![2]!.status).toBe("pending");
  });

  it("readPlanFile returns null for missing file", () => {
    const repoRoot = makeTempDir();
    const result = readPlanFile("nonexistent-session", repoRoot);
    expect(result).toBeNull();
  });

  it("all statuses handled correctly", () => {
    const repoRoot = makeTempDir();
    const sessionId = "status-test";
    const steps: PlanStep[] = [
      { description: "Done step", status: "completed" },
      { description: "Active step", status: "in_progress" },
      { description: "Waiting step", status: "pending" },
    ];

    writePlanFile(sessionId, repoRoot, steps);
    const result = readPlanFile(sessionId, repoRoot);

    expect(result).not.toBeNull();
    expect(result![0]!.status).toBe("completed");
    expect(result![1]!.status).toBe("in_progress");
    expect(result![2]!.status).toBe("pending");
  });

  it("getPlanFilePath returns expected path", () => {
    const path = getPlanFilePath("my-session", "/repo");
    expect(path).toBe(join("/repo", ".devagent", "plans", "my-session.md"));
  });
});
