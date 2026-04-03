import { describe, expect, it } from "vitest";

import { ITERATION_LIMIT_NOTICE } from "./App.js";

describe("interactive completion notices", () => {
  it("keeps the budget-exhausted follow-up hint stable", () => {
    expect(ITERATION_LIMIT_NOTICE).toBe("Iteration limit exhausted. Type /continue to proceed.");
  });
});
