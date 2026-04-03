import { afterEach, beforeEach, describe, expect, it } from "vitest";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

import { detectProjectType } from "./commands.js";

describe("detectProjectType", () => {
  let dir: string;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "devagent-cli-commands-"));
  });

  afterEach(() => {
    rmSync(dir, { recursive: true, force: true });
  });

  it("detects dotnet projects from solution files", () => {
    writeFileSync(join(dir, "Example.sln"), "");

    expect(detectProjectType(dir)).toBe("dotnet");
  });
});
