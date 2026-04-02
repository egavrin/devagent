/**
 * CommandPalette — Ctrl+K searchable command list overlay.
 */

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";

export interface Command {
  readonly name: string;
  readonly description: string;
  readonly shortcut?: string;
  readonly action: () => void;
}

export interface CommandPaletteProps {
  readonly commands: ReadonlyArray<Command>;
  readonly onClose: () => void;
}

export function CommandPalette({ commands, onClose }: CommandPaletteProps): React.ReactElement {
  const [search, setSearch] = useState("");
  const [selected, setSelected] = useState(0);

  const filtered = search
    ? commands.filter((c) =>
        c.name.toLowerCase().includes(search.toLowerCase()) ||
        c.description.toLowerCase().includes(search.toLowerCase()),
      )
    : commands;

  useInput((input, key) => {
    if (key.escape) {
      onClose();
      return;
    }
    if (key.return) {
      if (filtered[selected]) {
        filtered[selected].action();
        onClose();
      }
      return;
    }
    if (key.upArrow) {
      setSelected((s) => Math.max(0, s - 1));
      return;
    }
    if (key.downArrow) {
      setSelected((s) => Math.min(filtered.length - 1, s + 1));
      return;
    }
    if (key.backspace || key.delete) {
      setSearch((s) => s.slice(0, -1));
      setSelected(0);
      return;
    }
    if (input && !key.ctrl && !key.meta) {
      setSearch((s) => s + input);
      setSelected(0);
    }
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
