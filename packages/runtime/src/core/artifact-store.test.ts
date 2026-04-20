import { mkdtempSync, rmSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { describe, it, expect, beforeEach, afterEach } from "vitest";

import { ArtifactStore } from "./artifact-store.js";

describe("ArtifactStore", () => {
  let tempDir: string;
  let store: ArtifactStore;

  beforeEach(() => {
    tempDir = mkdtempSync(join(tmpdir(), "artifact-store-test-"));
    store = new ArtifactStore(tempDir);
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it("saves and loads JSON artifacts", () => {
    const data = { status: "ok", items: [1, 2, 3] };
    const path = store.save("run-001", "result.json", data);
    expect(path).toContain("run-001");
    expect(path).toContain("result.json");

    const loaded = store.load("run-001", "result.json");
    expect(loaded).toEqual(data);
  });

  it("saves and loads string artifacts", () => {
    const text = "plain text log output\nline two";
    store.save("run-002", "log.txt", text);

    const loaded = store.load("run-002", "log.txt");
    expect(loaded).toBe(text);
  });

  it("lists runs in reverse order", () => {
    store.save("run-aaa", "a.json", {});
    store.save("run-bbb", "b.json", {});
    store.save("run-ccc", "c.json", {});

    const runs = store.listRuns();
    expect(runs).toEqual(["run-ccc", "run-bbb", "run-aaa"]);
  });

  it("lists artifacts for a run", () => {
    store.save("run-100", "input.json", { query: "test" });
    store.save("run-100", "output.json", { result: "done" });
    store.save("run-100", "events.log", "event1\nevent2");

    const artifacts = store.listArtifacts("run-100");
    expect(artifacts).toEqual(["events.log", "input.json", "output.json"]);
  });

  it("returns null for missing artifact", () => {
    const result = store.load("nonexistent-run", "missing.json");
    expect(result).toBeNull();
  });

  it("returns empty arrays for missing base dir", () => {
    const emptyStore = new ArtifactStore(join(tempDir, "does-not-exist"));
    expect(emptyStore.listRuns()).toEqual([]);
    expect(emptyStore.listArtifacts("any-run")).toEqual([]);
  });
});
