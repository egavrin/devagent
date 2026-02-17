import { useState, useRef, useEffect, useCallback } from "react";
import type { ChatMessage, CommandInfo } from "../types";
import { MessageBubble } from "./MessageBubble";
import { CommandPalette } from "./CommandPalette";

interface ChatPanelProps {
  readonly messages: ReadonlyArray<ChatMessage>;
  readonly isStreaming: boolean;
  readonly onSendMessage: (content: string) => void;
  readonly onClear: () => void;
  readonly onAbort: () => void;
  readonly commands?: ReadonlyArray<CommandInfo>;
  readonly onExecuteCommand?: (name: string, args: string) => void;
}

export function ChatPanel({
  messages,
  isStreaming,
  onSendMessage,
  onClear,
  onAbort,
  commands = [],
  onExecuteCommand,
}: ChatPanelProps): React.JSX.Element {
  const [input, setInput] = useState("");
  const [paletteOpen, setPaletteOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Auto-resize textarea
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
    }
  }, [input]);

  const handleSubmit = useCallback(() => {
    if (!input.trim() || isStreaming) return;
    onSendMessage(input.trim());
    setInput("");
  }, [input, isStreaming, onSendMessage]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSubmit();
      }
      // Ctrl+K / Cmd+K to open command palette
      if (e.key === "k" && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        setPaletteOpen(true);
      }
    },
    [handleSubmit],
  );

  // Detect `/` at start to open palette
  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLTextAreaElement>) => {
      const value = e.target.value;
      if (value === "/" && input === "") {
        setPaletteOpen(true);
        setInput("");
        return;
      }
      setInput(value);
    },
    [input],
  );

  const handleCommandSelect = useCallback(
    (command: string, args: string) => {
      if (onExecuteCommand) {
        onExecuteCommand(command, args);
      }
      setPaletteOpen(false);
    },
    [onExecuteCommand],
  );

  // Global Escape to close palette
  useEffect(() => {
    const handleGlobalKey = (e: KeyboardEvent) => {
      if (e.key === "Escape" && paletteOpen) {
        setPaletteOpen(false);
      }
    };
    window.addEventListener("keydown", handleGlobalKey);
    return () => window.removeEventListener("keydown", handleGlobalKey);
  }, [paletteOpen]);

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <h2>DevAgent</h2>
            <p>AI-powered development assistant</p>
            <p className="chat-hint">
              Ask a question about your codebase, request code changes, or type{" "}
              <kbd>/</kbd> for commands.
            </p>
          </div>
        )}
        {messages.map((msg) => (
          <MessageBubble key={msg.id} message={msg} />
        ))}
        <div ref={messagesEndRef} />
      </div>

      <div className="chat-input-area">
        <div className="chat-input-row">
          <textarea
            ref={textareaRef}
            className="chat-input"
            value={input}
            onChange={handleInputChange}
            onKeyDown={handleKeyDown}
            placeholder={
              isStreaming ? "Waiting for response..." : "Type a message or / for commands..."
            }
            disabled={isStreaming}
            rows={1}
          />
          {isStreaming ? (
            <button
              className="chat-abort-btn"
              onClick={onAbort}
              title="Stop generation (Esc)"
            >
              Stop
            </button>
          ) : (
            <button
              className="chat-send-btn"
              onClick={handleSubmit}
              disabled={!input.trim()}
              title="Send message (Enter)"
            >
              Send
            </button>
          )}
        </div>
        <div className="chat-input-actions">
          <div className="chat-input-actions-left">
            <button className="chat-action-btn" onClick={onClear} title="Clear chat">
              Clear
            </button>
            {commands.length > 0 && (
              <button
                className="chat-action-btn"
                onClick={() => setPaletteOpen(true)}
                title="Command palette (Ctrl+K)"
              >
                / Commands
              </button>
            )}
          </div>
          <span className="chat-hint-small">
            Shift+Enter for newline · Ctrl+K commands
          </span>
        </div>
      </div>

      <CommandPalette
        commands={commands}
        isOpen={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        onSelect={handleCommandSelect}
      />
    </div>
  );
}
