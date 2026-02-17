/**
 * CommandPalette — modal overlay triggered by `/` in chat or Ctrl+K.
 *
 * Filtered command list with keyboard navigation (↑↓ Enter Esc).
 */

import { useState, useEffect, useRef, useCallback, useMemo } from "react";
import type { CommandInfo } from "../types";

interface CommandPaletteProps {
  readonly commands: ReadonlyArray<CommandInfo>;
  readonly isOpen: boolean;
  readonly onClose: () => void;
  readonly onSelect: (command: string, args: string) => void;
}

export function CommandPalette({
  commands,
  isOpen,
  onClose,
  onSelect,
}: CommandPaletteProps): React.JSX.Element | null {
  const [filter, setFilter] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const filtered = useMemo(() => {
    if (!filter) return commands;
    const q = filter.toLowerCase();
    return commands.filter(
      (c) =>
        c.name.toLowerCase().includes(q) ||
        c.description.toLowerCase().includes(q) ||
        c.plugin.toLowerCase().includes(q),
    );
  }, [commands, filter]);

  // Reset on open
  useEffect(() => {
    if (isOpen) {
      setFilter("");
      setSelectedIndex(0);
      // Focus input after render
      requestAnimationFrame(() => {
        inputRef.current?.focus();
      });
    }
  }, [isOpen]);

  // Keep selected index in bounds
  useEffect(() => {
    if (selectedIndex >= filtered.length) {
      setSelectedIndex(Math.max(0, filtered.length - 1));
    }
  }, [filtered.length, selectedIndex]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      switch (e.key) {
        case "ArrowDown":
          e.preventDefault();
          setSelectedIndex((i) => Math.min(i + 1, filtered.length - 1));
          break;
        case "ArrowUp":
          e.preventDefault();
          setSelectedIndex((i) => Math.max(i - 1, 0));
          break;
        case "Enter": {
          e.preventDefault();
          const cmd = filtered[selectedIndex];
          if (cmd) {
            onSelect(cmd.name, "");
            onClose();
          }
          break;
        }
        case "Escape":
          e.preventDefault();
          onClose();
          break;
      }
    },
    [filtered, selectedIndex, onSelect, onClose],
  );

  if (!isOpen) return null;

  return (
    <div className="cmd-overlay" onClick={onClose}>
      <div className="cmd-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="cmd-input-wrapper">
          <span className="cmd-slash">/</span>
          <input
            ref={inputRef}
            className="cmd-input"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search commands..."
          />
        </div>
        <div className="cmd-list">
          {filtered.length === 0 && (
            <div className="cmd-empty">No matching commands.</div>
          )}
          {filtered.map((cmd, idx) => (
            <button
              key={cmd.name}
              className={`cmd-item ${idx === selectedIndex ? "cmd-item-selected" : ""}`}
              onClick={() => {
                onSelect(cmd.name, "");
                onClose();
              }}
              onMouseEnter={() => setSelectedIndex(idx)}
            >
              <div className="cmd-item-main">
                <span className="cmd-item-name">/{cmd.name}</span>
                <span className="cmd-item-desc">{cmd.description}</span>
              </div>
              <span className="cmd-item-plugin">{cmd.plugin}</span>
            </button>
          ))}
        </div>
        <div className="cmd-footer">
          <span>↑↓ Navigate</span>
          <span>Enter Select</span>
          <span>Esc Close</span>
        </div>
      </div>
    </div>
  );
}
