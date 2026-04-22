/**
 * Shared constants, types, and helpers for TUI components.
 */

import { SafetyMode } from "@devagent/runtime";
import stringWidth from "string-width";

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
const FRAMED_BODY_PREFIX = "  │ ";
const DEFAULT_TERMINAL_COLUMNS = 80;
const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: "grapheme" });
const ANSI_CSI_PATTERN = /^\x1B\[[0-?]*[ -/]*[@-~]/;
const ANSI_OSC_PATTERN = /^\x1B\][\s\S]*?(?:\x07|\x1B\\)/;

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

export function resolveTerminalColumns(columns: number | undefined): number {
  if (columns !== undefined && columns > 0) {
    return columns;
  }
  const envColumns = Number.parseInt(process.env["COLUMNS"] ?? "", 10);
  return Number.isFinite(envColumns) && envColumns > 0 ? envColumns : DEFAULT_TERMINAL_COLUMNS;
}

export function framedBodyWidth(columns: number | undefined): number {
  return Math.max(1, resolveTerminalColumns(columns) - FRAMED_BODY_PREFIX.length);
}

export function wrapAnsiTextByWidth(text: string, width: number): string[] {
  const maxWidth = Math.max(1, width);
  if (text.length === 0) return [""];

  const rows: string[] = [];
  let current = "";
  let currentWidth = 0;
  let index = 0;

  while (index < text.length) {
    const remaining = text.slice(index);
    const controlSequence = readAnsiSequence(remaining);
    if (controlSequence) {
      current += controlSequence;
      index += controlSequence.length;
      continue;
    }

    const grapheme = nextGrapheme(remaining);
    const graphemeWidth = stringWidth(grapheme);
    if (currentWidth > 0 && currentWidth + graphemeWidth > maxWidth) {
      rows.push(current);
      current = "";
      currentWidth = 0;
    }

    current += grapheme;
    currentWidth += graphemeWidth;
    index += grapheme.length;
  }

  rows.push(current);
  return rows;
}

function readAnsiSequence(text: string): string | null {
  return text.match(ANSI_CSI_PATTERN)?.[0] ?? text.match(ANSI_OSC_PATTERN)?.[0] ?? null;
}

function nextGrapheme(text: string): string {
  return GRAPHEME_SEGMENTER.segment(text)[Symbol.iterator]().next().value?.segment ?? text[0] ?? "";
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
