/**
 * replace_in_file — Search and replace text in a file.
 * Category: mutating.
 */

import { readFileSync, writeFileSync, existsSync } from "node:fs";
import { resolve } from "node:path";
import type { ToolSpec } from "@devagent/core";
import { ToolError } from "@devagent/core";

export const replaceInFileTool: ToolSpec = {
  name: "replace_in_file",
  description:
    "Replace occurrences of a search string with a replacement string in a file. Fails if the search string is not found.",
  category: "mutating",
  paramSchema: {
    type: "object",
    properties: {
      path: { type: "string", description: "File path (relative to repo root)" },
      search: { type: "string", description: "Text to search for (exact match)" },
      replace: { type: "string", description: "Replacement text" },
      all: { type: "boolean", description: "Replace all occurrences (default: true)" },
    },
    required: ["path", "search", "replace"],
  },
  resultSchema: {
    type: "object",
    properties: {
      replacements: { type: "number" },
    },
  },
  handler: async (params, context) => {
    const filePath = resolve(context.repoRoot, params["path"] as string);
    const search = params["search"] as string;
    const replace = params["replace"] as string;
    const replaceAll = (params["all"] as boolean | undefined) ?? true;

    if (!existsSync(filePath)) {
      throw new ToolError(
        "replace_in_file",
        `File not found: ${params["path"] as string}`,
      );
    }

    const content = readFileSync(filePath, "utf-8");

    if (!content.includes(search)) {
      throw new ToolError(
        "replace_in_file",
        `Search string not found in ${params["path"] as string}`,
      );
    }

    let newContent: string;
    let count: number;

    if (replaceAll) {
      count = content.split(search).length - 1;
      newContent = content.replaceAll(search, replace);
    } else {
      count = 1;
      newContent = content.replace(search, replace);
    }

    writeFileSync(filePath, newContent, "utf-8");

    return {
      success: true,
      output: `Replaced ${count} occurrence(s) in ${params["path"] as string}`,
      error: null,
      artifacts: [filePath],
    };
  },
};
