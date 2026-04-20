import { describe, expect, it } from "vitest";

import { renderMarkdown } from "./markdown-render.js";

describe("renderMarkdown tables", () => {
  it("renders markdown tables as aligned terminal rows without outer bars", () => {
    const rendered = renderMarkdown([
      "| Before | After |",
      "| --- | --- |",
      "| a (param) | array |",
      "| _fn | sortFn |",
    ].join("\n"));

    const lines = rendered.split("\n");
    expect(lines[0]).toContain("Before");
    expect(lines[0]).toContain(" │ ");
    expect(lines[0]?.trimStart().startsWith("│")).toBe(false);
    expect(lines[1]).toContain("─┼─");
    expect(lines[2]).toContain("a (param)");
    expect(lines[3]).toContain("_fn");
  });
});
