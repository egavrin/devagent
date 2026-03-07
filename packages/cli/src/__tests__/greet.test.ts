import { describe, it, expect } from "vitest";
import { greet } from "../utils/greet.js";

describe("greet", () => {
  it("returns a greeting for a name", () => {
    expect(greet("World")).toBe("Hello, World!");
  });
});
