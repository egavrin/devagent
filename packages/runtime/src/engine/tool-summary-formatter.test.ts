import { describe, it, expect } from "vitest";
import { formatSearchFilesSummary } from "./tool-summary-formatter.js";

describe("formatSearchFilesSummary", () => {
  it("places match content lines before file list", () => {
    const toolCall = {
      name: "search_files",
      arguments: { pattern: "createPlanTool" },
    };
    const output = [
      "Found 3 matches for \"createPlanTool\"",
      "src/plan-tool.ts",
      "  42: export function createPlanTool(ctx: Context) {",
      "src/task-loop.ts",
      "  105: const tool = createPlanTool(this.context);",
      "src/index.ts",
      "  20: export { createPlanTool } from './plan-tool';",
    ].join("\n");

    const summary = formatSearchFilesSummary(toolCall, output);

    // Match lines (content) should appear before the Files: line
    const matchIdx = summary.indexOf("42: export function createPlanTool");
    const filesIdx = summary.indexOf("Files:");
    expect(matchIdx).toBeGreaterThan(-1);
    expect(filesIdx).toBeGreaterThan(-1);
    expect(matchIdx).toBeLessThan(filesIdx);
  });

  it("preserves match content under truncation (file list truncated first)", () => {
    const toolCall = {
      name: "search_files",
      arguments: { pattern: "import" },
    };

    // Create output with many long file paths and a few match lines
    const filePaths = Array.from({ length: 50 }, (_, i) =>
      `src/very/deeply/nested/module/submodule/component-${i}/index.ts`
    );
    const matchLines = [
      "  1: import { foo } from './bar';",
      "  15: import { baz } from './qux';",
    ];

    const output = [
      "Found 2 matches for \"import\"",
      ...filePaths.flatMap((fp, i) => i < 2 ? [fp, matchLines[i]!] : [fp]),
    ].join("\n");

    const summary = formatSearchFilesSummary(toolCall, output);

    // Match content should survive truncation
    expect(summary).toContain("import { foo }");
    expect(summary).toContain("import { baz }");
  });
});
