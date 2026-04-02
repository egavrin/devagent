import { describe, expect, it } from "vitest";
import { createDefaultRegistry } from "./index.js";

describe("createDefaultRegistry", () => {
  it("registers devagent-api as a built-in provider", () => {
    const registry = createDefaultRegistry();

    expect(registry.has("devagent-api")).toBe(true);
    expect(registry.list()).toContain("devagent-api");
  });
});
