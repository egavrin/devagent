import { Spinner } from "./format.js";
import { OutputState } from "./output-state.js";
import { TranscriptComposer } from "./transcript-composer.js";

/** Shared spinner instance used by CLI and event rendering. */
export const spinner = new Spinner();

/** Centralized mutable output state for one CLI process. */
export const outputState = new OutputState();

export const transcriptComposer = new TranscriptComposer();

let transcriptIdCounter = 0;

export function nextTranscriptId(prefix: string): string {
  return `${prefix}-${++transcriptIdCounter}`;
}
