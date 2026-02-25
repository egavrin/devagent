import { defineWorkspace } from "vitest/config";

export default defineWorkspace([
  "packages/core",
  "packages/engine",
  "packages/tools",
  "packages/providers",
  "packages/cli",
  "packages/arkts",
]);
