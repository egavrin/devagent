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
 * - Multi-line: Shift+Enter, continuation with … prefix
 * - History: Up/Down arrows
 * - Tab completion: slash commands + file paths
 */

import React, { useState, useRef } from "react";
import { readdirSync } from "node:fs";
import { join, dirname, basename } from "node:path";
import { Box, Text, useInput } from "ink";
import { getApprovalModeColor, resolvePromptTabAction } from "./shared.js";

export const SLASH_COMMANDS = [
  "/help",
  "/clear",
  "/continue",
  "/sessions",
  "/resume",
  "/rename",
  "/review",
  "/simplify",
  "/exit",
  "/quit",
];

export interface PromptInputProps {
  readonly onSubmit: (value: string) => void;
  readonly onCycleApprovalMode?: () => void;
  readonly placeholder?: string;
  readonly history?: ReadonlyArray<string>;
  readonly cwd?: string;
  readonly approvalMode?: string;
}

export function PromptInput({
  onSubmit,
  onCycleApprovalMode,
  placeholder = "Ask anything… use /review or /simplify anywhere",
  history = [],
  cwd,
  approvalMode = "suggest",
}: PromptInputProps): React.ReactElement {
  const [value, setValue] = useState("");
  const [cursorPos, setCursorPos] = useState(0);
  const [historyIndex, setHistoryIndex] = useState(-1);
  const [completions, setCompletions] = useState<string[]>([]);
  const [completionIndex, setCompletionIndex] = useState(0);
  const savedInputRef = useRef("");
  const accentColor = getApprovalModeColor(approvalMode);

  useInput((input, key) => {
    // Submit on Enter (without Shift)
    if (key.return && !key.shift) {
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

    // Newline on Shift+Enter
    if (key.return && key.shift) {
      const before = value.slice(0, cursorPos);
      const after = value.slice(cursorPos);
      setValue(before + "\n" + after);
      setCursorPos(cursorPos + 1);
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

  // Render the text with an inverted cursor character
  const renderWithCursor = (): React.ReactElement => {
    if (!value) {
      return (
        <Text>
          <Text inverse>{placeholder[0] ?? " "}</Text>
          <Text dimColor>{placeholder.slice(1)}</Text>
        </Text>
      );
    }

    const before = value.slice(0, cursorPos);
    const atCursor = value[cursorPos] ?? " ";
    const after = value.slice(cursorPos + 1);

    return (
      <Text>
        {before}<Text inverse>{atCursor}</Text>{after}
      </Text>
    );
  };

  const lines = value.split("\n");
  const isMultiLine = lines.length > 1;

  // For multi-line, render each line with cursor on the correct one
  const renderMultiLine = (): React.ReactElement[] => {
    let charOffset = 0;
    return lines.map((line, i) => {
      const lineStart = charOffset;
      const lineEnd = charOffset + line.length;
      charOffset = lineEnd + 1; // +1 for \n

      const cursorInLine = cursorPos >= lineStart && cursorPos <= lineEnd;
      const localCursor = cursorPos - lineStart;

      return (
        <Box key={i}>
          <Text color={accentColor}>{i === 0 ? "❯ " : "… "}</Text>
          {cursorInLine ? (
            <Text>
              {line.slice(0, localCursor)}<Text inverse>{line[localCursor] ?? " "}</Text>{line.slice(localCursor + 1)}
            </Text>
          ) : (
            <Text>{line}</Text>
          )}
        </Box>
      );
    });
  };

    return (
      <Box flexDirection="column">
      <Box borderStyle="round" borderColor={accentColor} paddingLeft={1} paddingRight={1}>
        <Box flexDirection="column" flexGrow={1}>
          {isMultiLine ? (
            renderMultiLine()
          ) : (
            <Box>
              <Text color={accentColor}>❯ </Text>
              {renderWithCursor()}
            </Box>
          )}
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
