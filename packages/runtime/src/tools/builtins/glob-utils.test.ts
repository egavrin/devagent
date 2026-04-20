import { describe, it, expect } from "vitest";

import { globToRegex, matchesGlob, toPosixPath } from "./glob-utils.js";

describe("glob-utils", () => {
  it("matches nested brace alternatives", () => {
    const regex = globToRegex("**/*.{ts,{js,jsx}}");
    expect(regex.test("src/file.ts")).toBe(true);
    expect(regex.test("src/file.js")).toBe(true);
    expect(regex.test("src/file.jsx")).toBe(true);
    expect(regex.test("src/file.rs")).toBe(false);
  });

  it("treats unmatched opening brace as a literal", () => {
    const regex = globToRegex("src/{literal.ts");
    expect(regex.test("src/{literal.ts")).toBe(true);
    expect(regex.test("src/literal.ts")).toBe(false);
  });

  it("supports empty brace alternatives", () => {
    const regex = globToRegex("src/file{,2}.ts");
    expect(regex.test("src/file.ts")).toBe(true);
    expect(regex.test("src/file2.ts")).toBe(true);
    expect(regex.test("src/file3.ts")).toBe(false);
  });

  it("normalizes windows-style paths", () => {
    expect(toPosixPath("src\\nested\\file.ts")).toBe("src/nested/file.ts");
  });

  it("matches against any provided candidate path", () => {
    const regex = globToRegex("*.rs");
    expect(matchesGlob(regex, ["src/main.rs", "main.rs"])).toBe(true);
    expect(matchesGlob(regex, ["src/main.rs", "src/lib.ts"])).toBe(false);
  });
});
