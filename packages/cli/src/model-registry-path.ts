import { resolve } from "node:path";

/**
 * Resolve the bundled model registry directory from the CLI module directory.
 * Works for both source (`packages/cli/src`) and built (`packages/cli/dist`) layouts.
 */
export function resolveBundledModelsDir(cliDir: string): string {
  return resolve(cliDir, "..", "..", "..", "models");
}
