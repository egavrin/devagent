/**
 * Shared constants, types, and helpers for TUI components.
 */

import { SafetyMode } from "@devagent/runtime";

import type { TranscriptNode } from "../transcript-composer.js";
import type { TranscriptPart } from "../transcript-presenter.js";
import type { TaskCompletionStatus } from "@devagent/runtime";

// ─── Spinner Constants ──────────────────────────────────────

export const SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

export const SPINNER_VERBS = [
  "Thinking", "Analyzing", "Considering", "Reasoning", "Processing",
  "Evaluating", "Examining", "Working", "Reflecting", "Assessing",
  "Deliberating", "Exploring", "Weighing", "Pondering", "Reviewing",
];

export interface LogEntry {
  readonly id: string;
  readonly part: TranscriptPart;
}

export type { TranscriptNode };

export interface InteractiveQueryResult {
  readonly iterations: number;
  readonly toolCalls: number;
  readonly lastText: string | null;
  readonly status: TaskCompletionStatus;
}

const APPROVAL_MODE_ORDER = [
  SafetyMode.AUTOPILOT,
  SafetyMode.DEFAULT,
] as const;

type PromptTabAction = "cycle-mode" | "complete" | "none";

export function cycleApprovalMode(mode: string): SafetyMode {
  const currentIndex = APPROVAL_MODE_ORDER.indexOf(mode as SafetyMode);
  const nextIndex = currentIndex >= 0 ? (currentIndex + 1) % APPROVAL_MODE_ORDER.length : 0;
  return APPROVAL_MODE_ORDER[nextIndex]!;
}

export function getApprovalModeColor(mode: string): "cyan" | "green" {
  switch (mode) {
    case SafetyMode.DEFAULT:
      return "cyan";
    case SafetyMode.AUTOPILOT:
      return "green";
    default:
      return "cyan";
  }
}

export function resolvePromptTabAction(key: { readonly tab?: boolean; readonly shift?: boolean }): PromptTabAction {
  if (!key.tab) return "none";
  return key.shift ? "cycle-mode" : "complete";
}

// ─── Text Helpers ───────────────────────────────────────────

export function cleanTime(text: string): string {
  return text.replace(/(\d+)\.\d+s/g, "$1s");
}

export function tokenProgressBar(used: number, max: number): string {
  const pct = Math.round((used / max) * 100);
  const width = 20;
  const filled = Math.round((pct / 100) * width);
  const bar = "▰".repeat(filled) + "▱".repeat(width - filled);
  const usedK = used >= 1000 ? `${Math.round(used / 1000)}k` : String(used);
  const maxK = max >= 1000 ? `${Math.round(max / 1000)}k` : String(max);
  return `${bar} ${usedK}/${maxK} (${pct}%)`;
}
