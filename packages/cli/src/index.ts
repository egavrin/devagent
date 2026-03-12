#!/usr/bin/env bun
/**
 * @devagent/cli — CLI frontend.
 * Supports single-query execution, review, and machine orchestration.
 *
 * Usage:
 *   devagent "explain the config system"     # Single query
 *   devagent review patch.diff --rule rule.md
 *   devagent execute --request request.json --artifact-dir out/
 */

import { main } from "./main.js";
import { extractErrorMessage } from "@devagent/runtime";

main().catch((err) => {
  console.error(extractErrorMessage(err));
  process.exit(1);
});
