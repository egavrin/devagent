#!/usr/bin/env bun
/**
 * @devagent/cli — CLI frontend.
 * Supports single-query and interactive chat modes.
 *
 * Usage:
 *   devagent "explain the config system"     # Single query
 *   devagent chat                            # Interactive mode
 *   devagent --plan "analyze the codebase"   # Plan mode (read-only)
 */

import { main } from "./main.js";
import { extractErrorMessage } from "@devagent/core";

main().catch((err) => {
  console.error(extractErrorMessage(err));
  process.exit(1);
});
