import { describe, expect, it } from "vitest";

import { resolveBundledModelsDir } from "./model-registry-path.js";

describe("resolveBundledModelsDir", () => {
  it("resolves to <repo>/models from src directory", () => {
    const cliDir = "/tmp/devagent/packages/cli/src";
    expect(resolveBundledModelsDir(cliDir)).toBe("/tmp/devagent/models");
  });

  it("resolves to <repo>/models from dist directory", () => {
    const cliDir = "/tmp/devagent/packages/cli/dist";
    expect(resolveBundledModelsDir(cliDir)).toBe("/tmp/devagent/models");
  });
});
