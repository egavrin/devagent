/**
 * write_file — Write content to a file (create only).
 * Category: mutating.
 */

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { dirname } from "node:path";
import type { ToolSpec } from "../../core/index.js";
import { ToolError } from "../../core/index.js";
import { FileTime } from "./file-time.js";
import { resolvePathInRepo } from "./path-guard.js";

export const writeFileTool: ToolSpec = {
  name: "write_file",
  description: "Create a new file with content. Creates parent directories if they don't exist. Fails if the target file already exists; use replace_in_file for existing files.",
  category: "mutating",
  paramSchema: {
    type: "object",
    properties: {
      path: { type: "string", description: "File path (relative to repo root)" },
      content: { type: "string", description: "Content to write" },
    },
    required: ["path", "content"],
  },
  errorGuidance: {
    common: "write_file only creates new files. Use replace_in_file to modify existing files.",
    patterns: [
      { match: "overwrite", hint: "This file already exists. Use replace_in_file to edit it, or choose a different filename." },
    ],
  },
  resultSchema: {
    type: "object",
    properties: {
      path: { type: "string" },
    },
  },
  handler: async (params, context) => {
    const filePath = resolvePathInRepo(
      context.repoRoot,
      params["path"] as string,
      "write_file",
    );
    const content = params["content"] as string;

    if (existsSync(filePath)) {
      throw new ToolError(
        "write_file",
        `Refusing to overwrite existing file ${params["path"] as string}. Use replace_in_file for edits.`,
      );
    }

    // Ensure parent directory exists
    const dir = dirname(filePath);
    if (!existsSync(dir)) {
      mkdirSync(dir, { recursive: true });
    }

    writeFileSync(filePath, content, "utf-8");
    FileTime.recordWrite(filePath);

    return {
      success: true,
      output: `Wrote ${content.length} bytes to ${params["path"] as string}`,
      error: null,
      artifacts: [filePath],
    };
  },
};
