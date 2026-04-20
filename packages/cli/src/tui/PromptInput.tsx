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

import React, { useRef, useState } from "react";
import { readdirSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { Box, Text, useInput, useStdout } from "ink";
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

export function shouldInsertPromptNewline(key: PromptInputKey): boolean {
  return Boolean(key.return && (key.shift || key.meta || key.super || key.hyper));
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
    if (shouldInsertPromptNewline(key)) {
      const before = value.slice(0, cursorPos);
      const after = value.slice(cursorPos);
      setValue(before + "\n" + after);
      setCursorPos(cursorPos + 1);
      return;
    }

    // Submit on plain Enter
    if (key.return) {
      const trimmed = value.trim();
      if (trimmed) {
        onSubmit(trimmed);
        setValue("");
        setCursorPos(0);
        setHistoryIndex(-1);
        savedInputRef.current = "";
      }
      return;
    }

    // History navigation
    if (key.upArrow && history.length > 0) {
      if (historyIndex === -1) savedInputRef.current = value;
      const newIndex = Math.min(historyIndex + 1, history.length - 1);
      setHistoryIndex(newIndex);
      const v = history[history.length - 1 - newIndex] ?? "";
      setValue(v);
      setCursorPos(v.length);
      return;
    }
    if (key.downArrow) {
      if (historyIndex > 0) {
        const newIndex = historyIndex - 1;
        setHistoryIndex(newIndex);
        const v = history[history.length - 1 - newIndex] ?? "";
        setValue(v);
        setCursorPos(v.length);
      } else if (historyIndex === 0) {
        setHistoryIndex(-1);
        setValue(savedInputRef.current);
        setCursorPos(savedInputRef.current.length);
      }
      return;
    }

    const tabAction = resolvePromptTabAction(key);
    if (tabAction === "cycle-mode") {
      onCycleApprovalMode?.();
      return;
    }

    // Tab completion
    if (tabAction === "complete") {
      if (completions.length > 0) {
        const next = (completionIndex + 1) % completions.length;
        setCompletionIndex(next);
        const c = completions[next]!;
        setValue(c);
        setCursorPos(c.length);
      } else {
        const candidates = getCompletions(value, cwd);
        if (candidates.length > 0) {
          setCompletions(candidates);
          setCompletionIndex(0);
          setValue(candidates[0]!);
          setCursorPos(candidates[0]!.length);
        }
      }
      return;
    }

    // Clear completions on other keys
    if (completions.length > 0) {
      setCompletions([]);
      setCompletionIndex(0);
    }

    // Backspace
    if (key.backspace || key.delete) {
      if (cursorPos > 0) {
        setValue(value.slice(0, cursorPos - 1) + value.slice(cursorPos));
        setCursorPos(cursorPos - 1);
      }
      return;
    }

    // Arrow keys
    if (key.leftArrow) { setCursorPos(Math.max(0, cursorPos - 1)); return; }
    if (key.rightArrow) { setCursorPos(Math.min(value.length, cursorPos + 1)); return; }

    // Regular input
    if (input && !key.ctrl && !key.meta) {
      const before = value.slice(0, cursorPos);
      const after = value.slice(cursorPos);
      setValue(before + input + after);
      setCursorPos(cursorPos + input.length);
      setHistoryIndex(-1);
    }
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
