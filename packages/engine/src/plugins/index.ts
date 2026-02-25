/**
 * Built-in plugins — export all built-in plugin factories.
 */

export { createCommitPlugin } from "./commit-plugin.js";
export { createReviewPlugin } from "./review-plugin.js";
export { createFeatureDevPlugin, getFeaturePhases } from "./feature-dev-plugin.js";
export type { FeaturePhase } from "./feature-dev-plugin.js";

import type { Plugin } from "@devagent/core";
import { createCommitPlugin } from "./commit-plugin.js";
import { createReviewPlugin } from "./review-plugin.js";
import { createFeatureDevPlugin } from "./feature-dev-plugin.js";

/**
 * Create all built-in plugins.
 */
export function createBuiltinPlugins(): ReadonlyArray<Plugin> {
  return [
    createCommitPlugin(),
    createReviewPlugin(),
    createFeatureDevPlugin(),
  ];
}
