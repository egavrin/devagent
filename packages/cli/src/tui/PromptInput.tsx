/**
 * PromptInput — dedicated input field with border, cursor, and history.
 *
 * Visual:
 * ╭─────────────────────────────────────╮
 * │ ❯ █sk anything…                     │
 * ╰─────────────────────────────────────╯
 *
 * Features:
 * - Bordered box (round style, gray border)
 * - ❯ prompt indicator (cyan)
 * - Visible cursor (inverted character at cursor position)
 * - Multi-line: modified Return inserts a newline, continuation with … prefix
 * - History: Up/Down arrows
 * - Tab completion: slash commands + file paths
 */

import { Box, Text, useInput, useStdout } from "ink";
import { readdirSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import React, { useRef, useState } from "react";
import stringWidth from "string-width";

import { getApprovalModeColor, resolvePromptTabAction } from "./shared.js";

export const SLASH_COMMANDS = [
  "/help",
  "/clear",
  "/continue",
  "/sessions",
  "/resume",
  "/review",
  "/simplify",
  "/exit",
  "/quit",
];

const FIRST_PROMPT_PREFIX = "❯ ";
const CONTINUATION_PROMPT_PREFIX = "… ";
const INPUT_FRAME_WIDTH = 6;
const GRAPHEME_SEGMENTER = new Intl.Segmenter(undefined, { granularity: "grapheme" });

interface PromptInputKey {
  readonly return?: boolean;
  readonly shift?: boolean;
  readonly meta?: boolean;
  readonly super?: boolean;
  readonly hyper?: boolean;
  readonly upArrow?: boolean;
  readonly downArrow?: boolean;
  readonly backspace?: boolean;
  readonly delete?: boolean;
  readonly leftArrow?: boolean;
  readonly rightArrow?: boolean;
  readonly ctrl?: boolean;
}

interface PromptRow {
  readonly prefix: string;
  readonly text: string;
  readonly cursorOffset: number | null;
  readonly dim: boolean;
}

interface WrappedPromptChunk {
  readonly text: string;
  readonly start: number;
  readonly end: number;
}

interface PromptInputProps {
  readonly onSubmit: (value: string) => void;
  readonly onCycleApprovalMode?: () => void;
  readonly placeholder?: string;
  readonly history?: ReadonlyArray<string>;
  readonly cwd?: string;
  readonly approvalMode?: string;
}

interface PromptInputControls {
  readonly value: string;
  readonly cursorPos: number;
  readonly historyIndex: number;
  readonly completions: readonly string[];
  readonly completionIndex: number;
  readonly history: ReadonlyArray<string>;
  readonly cwd: string | undefined;
  readonly savedInputRef: React.MutableRefObject<string>;
  readonly onSubmit: (value: string) => void;
  readonly onCycleApprovalMode: (() => void) | undefined;
  readonly setValue: React.Dispatch<React.SetStateAction<string>>;
  readonly setCursorPos: React.Dispatch<React.SetStateAction<number>>;
  readonly setHistoryIndex: React.Dispatch<React.SetStateAction<number>>;
  readonly setCompletions: React.Dispatch<React.SetStateAction<string[]>>;
  readonly setCompletionIndex: React.Dispatch<React.SetStateAction<number>>;
}

export function shouldInsertPromptNewline(key: PromptInputKey): boolean {
  return Boolean(key.return && (key.shift || key.meta || key.super || key.hyper));
}

export function shouldSubmitPromptInput(input: string, key: PromptInputKey): boolean {
  return Boolean(key.return || /[\r\n]/.test(input));
}

export function buildPromptSubmitValue(
  input: string,
  key: PromptInputKey,
  value: string,
  cursorPos: number,
): string | null {
  if (key.return) {
    return value;
  }

  const submitIndex = input.search(/[\r\n]/);
  if (submitIndex === -1) {
    return null;
  }

  const submittedInput = input.slice(0, submitIndex);
  return value.slice(0, cursorPos) + submittedInput + value.slice(cursorPos);
}

function splitGraphemes(line: string): Array<{ readonly text: string; readonly start: number; readonly end: number }> {
  return Array.from(
    GRAPHEME_SEGMENTER.segment(line),
    ({ segment, index }) => ({
      text: segment,
      start: index,
      end: index + segment.length,
    }),
  );
}

function wrapPromptLine(line: string, contentWidth: number): WrappedPromptChunk[] {
  const width = Math.max(1, contentWidth);
  if (line.length === 0) return [{ text: "", start: 0, end: 0 }];

  const rows: WrappedPromptChunk[] = [];
  const graphemes = splitGraphemes(line);
  let rowText = "";
  let rowWidth = 0;
  let rowStart = graphemes[0]!.start;
  let rowEnd = graphemes[0]!.start;

  for (const grapheme of graphemes) {
    const graphemeWidth = stringWidth(grapheme.text);

    if (rowText.length > 0 && rowWidth + graphemeWidth > width) {
      rows.push({ text: rowText, start: rowStart, end: rowEnd });
      rowText = "";
      rowWidth = 0;
    }

    if (rowText.length === 0) {
      rowStart = grapheme.start;
    }

    rowText += grapheme.text;
    rowWidth += graphemeWidth;
    rowEnd = grapheme.end;
  }

  rows.push({ text: rowText, start: rowStart, end: rowEnd });
  return rows;
}

function rowContainsCursor(
  cursorPos: number,
  rowStart: number,
  rowEnd: number,
  isLastChunk: boolean,
): boolean {
  if (cursorPos < rowStart || cursorPos > rowEnd) return false;
  if (cursorPos === rowEnd && !isLastChunk) return false;
  return true;
}

export function buildPromptRows(
  value: string,
  cursorPos: number,
  placeholder: string,
  contentWidth: number,
): PromptRow[] {
  const promptText = value.length > 0 ? value : placeholder;
  const dim = value.length === 0;
  const logicalLines = promptText.split("\n");
  const rows: PromptRow[] = [];
  let visualRowIndex = 0;
  let globalOffset = 0;

  for (const [lineIndex, line] of logicalLines.entries()) {
    const chunks = wrapPromptLine(line, contentWidth);

    for (const [chunkIndex, chunk] of chunks.entries()) {
      const rowStart = globalOffset + chunk.start;
      const rowEnd = globalOffset + chunk.end;
      const isLastChunk = chunkIndex === chunks.length - 1;
      const cursorOffset = rowContainsCursor(cursorPos, rowStart, rowEnd, isLastChunk)
        ? cursorPos - rowStart
        : null;

      rows.push({
        prefix: visualRowIndex === 0 ? FIRST_PROMPT_PREFIX : CONTINUATION_PROMPT_PREFIX,
        text: chunk.text,
        cursorOffset,
        dim,
      });

      visualRowIndex += 1;
    }

    globalOffset += line.length;
    if (lineIndex < logicalLines.length - 1) {
      globalOffset += 1;
    }
  }

  return rows;
}

function renderPromptText(row: PromptRow): React.ReactElement {
  if (row.cursorOffset === null) {
    return row.dim ? <Text dimColor>{row.text}</Text> : <Text>{row.text}</Text>;
  }

  const before = row.text.slice(0, row.cursorOffset);
  const atCursor = row.text[row.cursorOffset] ?? " ";
  const after = row.text.slice(row.cursorOffset + 1);

  if (row.dim) {
    return (
      <Text>
        <Text dimColor>{before}</Text>
        <Text inverse>{atCursor}</Text>
        <Text dimColor>{after}</Text>
      </Text>
    );
  }

  return (
    <Text>
      {before}
      <Text inverse>{atCursor}</Text>
      {after}
    </Text>
  );
}

function submitPromptValue(input: string, key: PromptInputKey, controls: PromptInputControls): boolean {
  if (!shouldSubmitPromptInput(input, key)) {
    return false;
  }

  const submitValue = buildPromptSubmitValue(input, key, controls.value, controls.cursorPos) ?? controls.value;
  const trimmed = submitValue.trim();
  if (trimmed) {
    controls.onSubmit(trimmed);
    controls.setValue("");
    controls.setCursorPos(0);
    controls.setHistoryIndex(-1);
    controls.savedInputRef.current = "";
  }
  return true;
}

function handleHistoryKey(key: PromptInputKey, controls: PromptInputControls): boolean {
  if (key.upArrow && controls.history.length > 0) {
    if (controls.historyIndex === -1) controls.savedInputRef.current = controls.value;
    const newIndex = Math.min(controls.historyIndex + 1, controls.history.length - 1);
    const historyValue = controls.history[controls.history.length - 1 - newIndex] ?? "";
    controls.setHistoryIndex(newIndex);
    controls.setValue(historyValue);
    controls.setCursorPos(historyValue.length);
    return true;
  }

  if (!key.downArrow) {
    return false;
  }

  if (controls.historyIndex > 0) {
    const newIndex = controls.historyIndex - 1;
    const historyValue = controls.history[controls.history.length - 1 - newIndex] ?? "";
    controls.setHistoryIndex(newIndex);
    controls.setValue(historyValue);
    controls.setCursorPos(historyValue.length);
  } else if (controls.historyIndex === 0) {
    controls.setHistoryIndex(-1);
    controls.setValue(controls.savedInputRef.current);
    controls.setCursorPos(controls.savedInputRef.current.length);
  }
  return true;
}

function applyCompletion(completion: string, controls: PromptInputControls): void {
  controls.setValue(completion);
  controls.setCursorPos(completion.length);
}

function handleCompletionKey(key: PromptInputKey, controls: PromptInputControls): boolean {
  const tabAction = resolvePromptTabAction(key);
  if (tabAction === "cycle-mode") {
    controls.onCycleApprovalMode?.();
    return true;
  }

  if (tabAction !== "complete") {
    return false;
  }

  if (controls.completions.length > 0) {
    const next = (controls.completionIndex + 1) % controls.completions.length;
    controls.setCompletionIndex(next);
    applyCompletion(controls.completions[next]!, controls);
    return true;
  }

  const candidates = getCompletions(controls.value, controls.cwd);
  if (candidates.length > 0) {
    controls.setCompletions(candidates);
    controls.setCompletionIndex(0);
    applyCompletion(candidates[0]!, controls);
  }
  return true;
}

function clearCompletions(controls: PromptInputControls): void {
  if (controls.completions.length === 0) {
    return;
  }
  controls.setCompletions([]);
  controls.setCompletionIndex(0);
}

function handleEditingKey(input: string, key: PromptInputKey, controls: PromptInputControls): boolean {
  clearCompletions(controls);
  if (key.backspace || key.delete) {
    if (controls.cursorPos > 0) {
      controls.setValue(controls.value.slice(0, controls.cursorPos - 1) + controls.value.slice(controls.cursorPos));
      controls.setCursorPos(controls.cursorPos - 1);
    }
    return true;
  }
  if (key.leftArrow) {
    controls.setCursorPos(Math.max(0, controls.cursorPos - 1));
    return true;
  }
  if (key.rightArrow) {
    controls.setCursorPos(Math.min(controls.value.length, controls.cursorPos + 1));
    return true;
  }
  if (!input || key.ctrl || key.meta) {
    return false;
  }

  controls.setValue(controls.value.slice(0, controls.cursorPos) + input + controls.value.slice(controls.cursorPos));
  controls.setCursorPos(controls.cursorPos + input.length);
  controls.setHistoryIndex(-1);
  return true;
}

function insertPromptNewline(controls: PromptInputControls): void {
  controls.setValue(
    controls.value.slice(0, controls.cursorPos) + "\n" + controls.value.slice(controls.cursorPos),
  );
  controls.setCursorPos(controls.cursorPos + 1);
}

function handlePromptInput(input: string, key: PromptInputKey, controls: PromptInputControls): void {
  if (shouldInsertPromptNewline(key)) {
    insertPromptNewline(controls);
    return;
  }
  if (submitPromptValue(input, key, controls)) return;
  if (handleHistoryKey(key, controls)) return;
  if (handleCompletionKey(key, controls)) return;
  handleEditingKey(input, key, controls);
}

export function PromptInput({
  onSubmit,
  onCycleApprovalMode,
  placeholder = "Ask anything… use /review or /simplify anywhere",
  history = [],
  cwd,
  approvalMode = "autopilot",
}: PromptInputProps): React.ReactElement {
  const [value, setValue] = useState("");
  const [cursorPos, setCursorPos] = useState(0);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [completions, setCompletions] = useState<string[]>([]);
  const [completionIndex, setCompletionIndex] = useState(0);
  const savedInputRef = useRef("");
  const { stdout } = useStdout();
  const accentColor = getApprovalModeColor(approvalMode);
  const contentWidth = Math.max(1, (stdout.columns ?? 80) - INPUT_FRAME_WIDTH);
  const promptRows = buildPromptRows(value, cursorPos, placeholder, contentWidth);
  useInput((input, key) => {
    handlePromptInput(input, key, {
      value,
      cursorPos,
      historyIndex,
      completions,
      completionIndex,
      history,
      cwd,
      savedInputRef,
      onSubmit,
      onCycleApprovalMode,
      setValue,
      setCursorPos,
      setHistoryIndex,
      setCompletions,
      setCompletionIndex,
    });
  });

  return (
    <Box flexDirection="column">
      <Box borderStyle="round" borderColor={accentColor} paddingLeft={1} paddingRight={1} width="100%">
        <Box flexDirection="column" flexGrow={1}>
          {promptRows.map((row, index) => (
            <Box key={`${index}-${row.prefix}`}>
              <Text color={accentColor}>{row.prefix}</Text>
              {renderPromptText(row)}
            </Box>
          ))}
        </Box>
      </Box>
      {completions.length > 1 && (
        <Text dimColor>  Tab: {completionIndex + 1}/{completions.length} · Shift+Tab: mode</Text>
      )}
    </Box>
  );
}

// ─── Completion Logic ───────────────────────────────────────

export function getCompletions(input: string, cwd?: string): string[] {
  const lastWordMatch = input.match(/(?:^|\s)(\S+)$/);
  const lastWord = lastWordMatch?.[1] ?? "";
  const looksLikeSlashCommand = lastWord.startsWith("/") && !lastWord.slice(1).includes("/");
  if (looksLikeSlashCommand) {
    const prefix = input.slice(0, input.length - lastWord.length);
    return SLASH_COMMANDS
      .filter((command) => command.startsWith(lastWord))
      .map((command) => `${prefix}${command}`);
  }

  const trimmed = input.trim();
  const words = trimmed.split(/\s+/);
  const finalWord = words[words.length - 1] ?? "";
  if (finalWord.includes("/") || finalWord.startsWith(".")) {
    try {
      const resolvedDir = cwd ? join(cwd, dirname(finalWord)) : dirname(finalWord);
      const prefix = basename(finalWord);
      const entries = readdirSync(resolvedDir);
      return entries
        .filter((e) => e.startsWith(prefix) && !e.startsWith("."))
        .slice(0, 10)
        .map((e) => {
          const pathPrefix = words.slice(0, -1).join(" ");
          const newPath = dirname(finalWord) === "." ? e : join(dirname(finalWord), e);
          return pathPrefix ? `${pathPrefix} ${newPath}` : newPath;
        });
    } catch { /* not readable */ }
  }
  return [];
}
