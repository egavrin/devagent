import { describe, it, expect } from "vitest";

import { createFindingTool } from "./finding-tool.js";
import { SessionState } from "./session-state.js";

describe("save_finding tool", () => {
  it("saves a finding to session state", async () => {
    const ss = new SessionState();
    const tool = createFindingTool(() => ss, () => 5);

    const result = await tool.handler(
      { title: "SQL injection", detail: "User input not sanitized in login()" },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(true);
    expect(result.output).toContain("Finding saved");

    const findings = ss.getFindings();
    expect(findings.length).toBe(1);
    expect(findings[0]!.title).toBe("SQL injection");
    expect(findings[0]!.detail).toBe("User input not sanitized in login()");
    expect(findings[0]!.iteration).toBe(5);
  });

  it("deduplicates by title when saving same finding twice", async () => {
    const ss = new SessionState();
    const tool = createFindingTool(() => ss, () => 3);

    await tool.handler(
      { title: "Memory leak", detail: "Buffer not freed" },
      { repoRoot: "/tmp" },
    );
    await tool.handler(
      { title: "Memory leak", detail: "Buffer freed but reference held" },
      { repoRoot: "/tmp" },
    );

    const findings = ss.getFindings();
    expect(findings.length).toBe(1);
    expect(findings[0]!.detail).toBe("Buffer freed but reference held");
  });

  it("fails with empty title", async () => {
    const ss = new SessionState();
    const tool = createFindingTool(() => ss, () => 1);

    const result = await tool.handler(
      { title: "  ", detail: "some detail" },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("empty");
    expect(ss.getFindings().length).toBe(0);
  });

  it("fails when session state is not available", async () => {
    const tool = createFindingTool(() => undefined, () => 1);

    const result = await tool.handler(
      { title: "Test", detail: "Detail" },
      { repoRoot: "/tmp" },
    );

    expect(result.success).toBe(false);
    expect(result.error).toContain("not available");
  });

  it("trims whitespace from title and detail", async () => {
    const ss = new SessionState();
    const tool = createFindingTool(() => ss, () => 1);

    await tool.handler(
      { title: "  Bug  ", detail: "  detail  " },
      { repoRoot: "/tmp" },
    );

    const findings = ss.getFindings();
    expect(findings[0]!.title).toBe("Bug");
    expect(findings[0]!.detail).toBe("detail");
  });

  it("has correct tool metadata", () => {
    const tool = createFindingTool(() => undefined, () => 0);
    expect(tool.name).toBe("save_finding");
    expect(tool.category).toBe("state");
    expect(tool.paramSchema.required).toContain("title");
    expect(tool.paramSchema.required).toContain("detail");
  });
});
