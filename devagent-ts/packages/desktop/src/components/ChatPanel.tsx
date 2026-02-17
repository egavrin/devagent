import { useState, useRef, useEffect, useCallback } from "react";
import type { ChatMessage } from "../types";
import { MessageBubble } from "./MessageBubble";

interface ChatPanelProps {
  readonly messages: ReadonlyArray<ChatMessage>;
  readonly isStreaming: boolean;
  readonly onSendMessage: (content: string) => void;
  readonly onClear: () => void;
  readonly onAbort: () => void;
}

export function ChatPanel({
  messages,
  isStreaming,
  onSendMessage,
  onClear,
  onAbort,
}: ChatPanelProps): React.JSX.Element {
  const [input, setInput] = useState("");
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
    },
    [handleSubmit],
  );

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <h2>DevAgent</h2>
            <p>AI-powered development assistant</p>
            <p className="chat-hint">
              Ask a question about your codebase, request code changes, or start
              a workflow.
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
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={
              isStreaming ? "Waiting for response..." : "Type a message..."
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
          <button className="chat-action-btn" onClick={onClear} title="Clear chat">
            Clear
          </button>
          <span className="chat-hint-small">
            Shift+Enter for newline
          </span>
        </div>
      </div>
    </div>
  );
}
