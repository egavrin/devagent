import { describe, expect, it } from "vitest";

import { greet } from "../utils/greet.js";

describe("greet", () => {
  it("returns a friendly greeting", () => {
    expect(greet("World")).toBe("Hello, World!");
  });
});
