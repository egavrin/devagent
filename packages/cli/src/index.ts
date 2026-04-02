#!/usr/bin/env node
/**
 * @devagent/cli — CLI frontend.
 * Supports single-query execution, review, and machine orchestration.
 *
 * Usage:
 *   devagent "explain the config system"     # Single query
 *   devagent review patch.diff --rule rule.md
 *   devagent execute --request request.json --artifact-dir out/
 */

import { getUnsupportedRuntimeMessage } from "./runtime-version.js";

const unsupportedRuntime = getUnsupportedRuntimeMessage();
if (unsupportedRuntime) {
  process.stderr.write(`${unsupportedRuntime}\n`);
  process.exit(1);
}

void import("./main.js")
  .then(({ main }) => main())
  .catch(async (err) => {
    try {
      const { extractErrorMessage } = await import("@devagent/runtime");
      console.error(extractErrorMessage(err));
    } catch {
      console.error(err instanceof Error ? err.message : String(err));
    }
    process.exit(1);
  });
