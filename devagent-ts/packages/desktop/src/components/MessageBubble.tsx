import type { ChatMessage } from "../types";

interface MessageBubbleProps {
  readonly message: ChatMessage;
}

export function MessageBubble({
  message,
}: MessageBubbleProps): React.JSX.Element {
  const isUser = message.role === "user";
  const isTool = message.role === "tool";

  return (
    <div className={`message ${isUser ? "message-user" : "message-assistant"}`}>
      <div className="message-header">
        <span className="message-role">
          {isUser ? "You" : isTool ? `Tool: ${message.toolName ?? "unknown"}` : "DevAgent"}
        </span>
        <span className="message-time">
          {new Date(message.timestamp).toLocaleTimeString()}
        </span>
      </div>
      <div className="message-content">
        {message.isStreaming && !message.content ? (
          <span className="message-thinking">Thinking...</span>
        ) : (
          <pre className="message-text">{message.content}</pre>
        )}
      </div>
    </div>
  );
}
