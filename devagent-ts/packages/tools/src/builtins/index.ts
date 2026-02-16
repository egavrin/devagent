/**
 * Built-in tool collection.
 */

import type { ToolSpec } from "@devagent/core";
import { readFileTool } from "./read-file.js";
import { writeFileTool } from "./write-file.js";
import { replaceInFileTool } from "./replace-in-file.js";
import { findFilesTool } from "./find-files.js";
import { searchFilesTool } from "./search-files.js";
import { runCommandTool } from "./run-command.js";
import { gitStatusTool, gitDiffTool, gitCommitTool } from "./git.js";

export {
  readFileTool,
  writeFileTool,
  replaceInFileTool,
  findFilesTool,
  searchFilesTool,
  runCommandTool,
  gitStatusTool,
  gitDiffTool,
  gitCommitTool,
};

/**
 * All built-in tools in registration order.
 */
export const builtinTools: ReadonlyArray<ToolSpec> = [
  readFileTool,
  writeFileTool,
  replaceInFileTool,
  findFilesTool,
  searchFilesTool,
  runCommandTool,
  gitStatusTool,
  gitDiffTool,
  gitCommitTool,
];
