/**
 * Review module — structured, rule-based patch review with LLM.
 */

export { VIOLATION_SCHEMA } from "./schema.js";
export type { Severity, ReviewChangeType, Violation, ReviewSummary, ReviewResult } from "./schema.js";

export type { ReviewConfig } from "./chunker.js";

export type { PatchReviewData } from "./validator.js";

export type { ContextItem, ContextProvider } from "./context.js";

export { runReviewPipeline } from "./pipeline.js";
export type { ReviewPipelineOptions, ReviewPipelineInput } from "./pipeline.js";
