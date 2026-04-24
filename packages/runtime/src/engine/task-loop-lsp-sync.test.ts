import { describe, expect, it, vi } from "vitest";

import { syncLSPAfterToolResult } from "./task-loop-lsp-sync.js";
import type { ToolResult } from "../core/index.js";

function result(overrides: Partial<ToolResult> = {}): ToolResult {
  return {
    success: true,
    output: "",
    error: null,
    artifacts: [],
    ...overrides,
  };
}

describe("syncLSPAfterToolResult", () => {
  it("syncs read_file path without didSave after a successful read", async () => {
    const syncFile = vi.fn(async () => undefined);
    await syncLSPAfterToolResult(
      "read_file",
      { path: "src/a.ts" },
      result(),
      { syncFile },
    );

    expect(syncFile).toHaveBeenCalledWith("src/a.ts", { didSave: false });
  });

  it("does not sync failed read_file calls", async () => {
    const syncFile = vi.fn(async () => undefined);
    await syncLSPAfterToolResult(
      "read_file",
      { path: "src/a.ts" },
      result({ success: false, error: "missing" }),
      { syncFile },
    );

    expect(syncFile).not.toHaveBeenCalled();
  });

  it("does not sync skill reads through the repository LSP", async () => {
    const syncFile = vi.fn(async () => undefined);
    await syncLSPAfterToolResult(
      "read_file",
      { path: "skill://example/reference.ts" },
      result(),
      { syncFile },
    );

    expect(syncFile).not.toHaveBeenCalled();
  });

  it("syncs write_file artifacts with didSave", async () => {
    const syncFile = vi.fn(async () => undefined);
    await syncLSPAfterToolResult(
      "write_file",
      { path: "src/a.ts" },
      result({ artifacts: ["/repo/src/a.ts"] }),
      { syncFile },
    );

    expect(syncFile).toHaveBeenCalledWith("/repo/src/a.ts", { didSave: true });
  });

  it("syncs replace_in_file partial-write artifacts even when the tool failed", async () => {
    const syncFile = vi.fn(async () => undefined);
    await syncLSPAfterToolResult(
      "replace_in_file",
      { path: "src/a.ts" },
      result({ success: false, error: "Some replacements failed", artifacts: ["/repo/src/a.ts"] }),
      { syncFile },
    );

    expect(syncFile).toHaveBeenCalledWith("/repo/src/a.ts", { didSave: true });
  });

  it("does not alter flow when sync throws", async () => {
    const syncFile = vi.fn(async () => { throw new Error("server down"); });

    await expect(syncLSPAfterToolResult(
      "write_file",
      {},
      result({ artifacts: ["src/a.ts"] }),
      { syncFile },
    )).resolves.toBeUndefined();
  });
});
