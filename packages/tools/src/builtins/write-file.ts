/**
 * write_file — Write content to a file (create or overwrite).
 * Category: mutating.
 */

import { writeFileSync, mkdirSync, existsSync } from "node:fs";
import { resolve, dirname } from "node:path";
import type { ToolSpec } from "@devagent/core";
import { ToolError } from "@devagent/core";
import { FileTime } from "./file-time.js";

export const writeFileTool: ToolSpec = {
  name: "write_file",
  description: "Write content to a file. Creates the file and parent directories if they don't exist. For small edits to existing files, prefer replace_in_file instead. Do not re-read after writing — the tool confirms success or failure.",
  category: "mutating",
  paramSchema: {
    type: "object",
    properties: {
      path: { type: "string", description: "File path (relative to repo root)" },
      content: { type: "string", description: "Content to write" },
    },
    required: ["path", "content"],
  },
  resultSchema: {
    type: "object",
    properties: {
      path: { type: "string" },
    },
  },
  handler: async (params, context) => {
    const filePath = resolve(context.repoRoot, params["path"] as string);
    const content = params["content"] as string;

    // For existing files, enforce pre-read (prevents overwriting content the LLM hasn't seen)
    if (existsSync(filePath) && !FileTime.wasRead(filePath)) {
      throw new ToolError(
        "write_file",
        `You must read file ${params["path"] as string} before overwriting it. Use read_file first.`,
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
