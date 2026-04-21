/**
 * CommandPalette — Ctrl+K searchable command list overlay.
 */

import { Box, Text, useInput } from "ink";
import React, { useState } from "react";

export interface Command {
  readonly name: string;
  readonly description: string;
  readonly shortcut?: string;
  readonly action: () => void;
}

interface CommandPaletteProps {
  readonly commands: ReadonlyArray<Command>;
  readonly onClose: () => void;
}
export function CommandPalette({ commands, onClose }: CommandPaletteProps): React.ReactElement {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(0);

  const filtered = filterCommands(commands, search);
  useInput((input, key) => {
    handlePaletteInput({ input, key, filtered, selected, setSearch, setSelected, onClose });
  });

  const visible = filtered.slice(0, 10);

  return (
    <Box flexDirection="column" paddingLeft={1} marginTop={1}>
      <Text bold color="cyan">Command Palette</Text>
      <Box marginTop={1}>
        <Text color="cyan">&gt; </Text>
        <Text>{search || ""}</Text>
        <Text dimColor>{!search ? "Type to filter…" : ""}</Text>
      </Box>
      <Box flexDirection="column" marginTop={1}>
        {visible.map((cmd, i) => (
          <Box key={cmd.name}>
            <Text color={i === selected ? "cyan" : undefined} bold={i === selected}>
              {i === selected ? "❯ " : "  "}
              {cmd.name}
            </Text>
            <Text dimColor> — {cmd.description}</Text>
            {cmd.shortcut && <Text dimColor> ({cmd.shortcut})</Text>}
          </Box>
        ))}
        {filtered.length === 0 && <Text dimColor>  No matching commands</Text>}
      </Box>
      <Text dimColor>  ↑↓ navigate │ Enter select │ Esc close</Text>
    </Box>
  );
}

function filterCommands(commands: ReadonlyArray<Command>, search: string): ReadonlyArray<Command> {
  if (!search) return commands;
  const needle = search.toLowerCase();
  return commands.filter((command) =>
    command.name.toLowerCase().includes(needle) ||
    command.description.toLowerCase().includes(needle),
  );
}

interface PaletteInputOptions {
  readonly input: string;
  readonly key: {
    readonly escape?: boolean;
    readonly return?: boolean;
    readonly upArrow?: boolean;
    readonly downArrow?: boolean;
    readonly backspace?: boolean;
    readonly delete?: boolean;
    readonly ctrl?: boolean;
    readonly meta?: boolean;
  };
  readonly filtered: ReadonlyArray<Command>;
  readonly selected: number;
  readonly setSearch: React.Dispatch<React.SetStateAction<string>>;
  readonly setSelected: React.Dispatch<React.SetStateAction<number>>;
  readonly onClose: () => void;
}

function handlePaletteInput(options: PaletteInputOptions): void {
  if (handlePaletteCommandKey(options)) {
    return;
  }
  handlePaletteTextKey(options);
}

function handlePaletteCommandKey(options: PaletteInputOptions): boolean {
  const { key, filtered, selected, setSelected, onClose } = options;
  if (key.escape) {
    onClose();
    return true;
  }
  if (key.return) {
    filtered[selected]?.action();
    onClose();
    return true;
  }
  if (key.upArrow) {
    setSelected((value) => Math.max(0, value - 1));
    return true;
  }
  if (key.downArrow) {
    setSelected((value) => Math.min(filtered.length - 1, value + 1));
    return true;
  }
  return false;
}

function handlePaletteTextKey(options: PaletteInputOptions): void {
  const { input, key, setSearch, setSelected } = options;
  if (key.backspace || key.delete) {
    setSearch((value) => value.slice(0, -1));
    setSelected(0);
  } else if (input && !key.ctrl && !key.meta) {
    setSearch((value) => value + input);
    setSelected(0);
  }
}
