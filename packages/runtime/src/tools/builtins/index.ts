/**
 * Built-in tool collection.
 */

import { createFindFilesTool, findFilesTool } from "./find-files.js";
import { gitStatusTool, gitDiffTool, gitCommitTool } from "./git.js";
import { createReadFileTool, readFileTool } from "./read-file.js";
import { replaceInFileTool } from "./replace-in-file.js";
import { runCommandTool } from "./run-command.js";
import { createSearchFilesTool, searchFilesTool } from "./search-files.js";
import { writeFileTool } from "./write-file.js";
import type { ToolSpec } from "../../core/index.js";
export {
  createReadFileTool,
  readFileTool,
  writeFileTool,
  replaceInFileTool,
  createFindFilesTool,
  findFilesTool,
  createSearchFilesTool,
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
