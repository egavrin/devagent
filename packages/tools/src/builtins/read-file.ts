/**
 * read_file — Read file contents with optional line range.
 * Category: readonly.
 */

import { readFileSync, existsSync } from "node:fs";
import type { ToolSpec } from "@devagent/core";
import { ToolError } from "@devagent/core";
import { FileTime } from "./file-time.js";
import { resolvePathInRepo } from "./path-guard.js";

export const readFileTool: ToolSpec = {
  name: "read_file",
  description:
    "Read the contents of a file. Optionally specify start_line and end_line to read a range. For large files (>200 lines), use start_line/end_line to read targeted sections. Always read before editing with replace_in_file.",
  category: "readonly",
  paramSchema: {
    type: "object",
    properties: {
      path: { type: "string", description: "File path (relative to repo root)" },
      start_line: { type: "number", description: "Start line (1-indexed, inclusive)" },
      end_line: { type: "number", description: "End line (1-indexed, inclusive)" },
    },
    required: ["path"],
  },
  resultSchema: {
    type: "object",
    properties: {
      content: { type: "string" },
      total_lines: { type: "number" },
    },
  },
  handler: async (params, context) => {
    const filePath = resolvePathInRepo(
      context.repoRoot,
      params["path"] as string,
      "read_file",
    );

    if (!existsSync(filePath)) {
      throw new ToolError("read_file", `File not found: ${params["path"] as string}`);
    }

    const content = readFileSync(filePath, "utf-8");
    FileTime.recordRead(filePath);
    const lines = content.split("\n");
    const totalLines = lines.length;

    const startLine = (params["start_line"] as number | undefined) ?? 1;
    const endLine = (params["end_line"] as number | undefined) ?? totalLines;

    const selectedLines = lines.slice(startLine - 1, endLine);
    const numberedContent = selectedLines
      .map((line, i) => `${startLine + i}\t${line}`)
      .join("\n");

    return {
      success: true,
      output: numberedContent,
      error: null,
      artifacts: [],
    };
  },
};
