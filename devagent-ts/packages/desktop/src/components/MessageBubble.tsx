/**
 * MessageBubble — renders chat messages with basic markdown support.
 *
 * Supports: **bold**, `inline code`, ```code blocks```, [links](url).
 * Custom parser, no external dependencies.
 */

import type { ChatMessage } from "../types";

interface MessageBubbleProps {
  readonly message: ChatMessage;
}

/**
 * Minimal markdown-to-JSX renderer.
 * Handles: code blocks, inline code, bold, and links.
 */
function renderMarkdown(text: string): React.JSX.Element {
  const elements: React.JSX.Element[] = [];
  const lines = text.split("\n");

  let i = 0;
  let key = 0;

  while (i < lines.length) {
    const line = lines[i]!;

    // Fenced code block
    if (line.startsWith("```")) {
      const lang = line.slice(3).trim();
      const codeLines: string[] = [];
      i++;
      while (i < lines.length && !lines[i]!.startsWith("```")) {
        codeLines.push(lines[i]!);
        i++;
      }
      i++; // skip closing ```
      elements.push(
        <div key={key++} className="md-code-block">
          {lang && <div className="md-code-lang">{lang}</div>}
          <pre className="md-code-pre"><code>{codeLines.join("\n")}</code></pre>
        </div>,
      );
      continue;
    }

    // Regular line — parse inline elements
    elements.push(
      <div key={key++} className="md-line">
        {renderInline(line)}
      </div>,
    );
    i++;
  }

  return <div className="md-content">{elements}</div>;
}

function renderInline(text: string): React.ReactNode {
  const nodes: React.ReactNode[] = [];
  let remaining = text;
  let key = 0;

  while (remaining.length > 0) {
    // Inline code: `...`
    const codeMatch = remaining.match(/^`([^`]+)`/);
    if (codeMatch) {
      nodes.push(
        <code key={key++} className="md-inline-code">{codeMatch[1]}</code>,
      );
      remaining = remaining.slice(codeMatch[0].length);
      continue;
    }

    // Bold: **...**
    const boldMatch = remaining.match(/^\*\*([^*]+)\*\*/);
    if (boldMatch) {
      nodes.push(
        <strong key={key++} className="md-bold">{boldMatch[1]}</strong>,
      );
      remaining = remaining.slice(boldMatch[0].length);
      continue;
    }

    // Link: [text](url)
    const linkMatch = remaining.match(/^\[([^\]]+)\]\(([^)]+)\)/);
    if (linkMatch) {
      nodes.push(
        <a
          key={key++}
          className="md-link"
          href={linkMatch[2]}
          target="_blank"
          rel="noopener noreferrer"
        >
          {linkMatch[1]}
        </a>,
      );
      remaining = remaining.slice(linkMatch[0].length);
      continue;
    }

    // Plain text — consume until next special character
    const nextSpecial = remaining.search(/[`*\[]/);
    if (nextSpecial === -1) {
      nodes.push(remaining);
      break;
    } else if (nextSpecial === 0) {
      // Special char that didn't match a pattern — consume one char
      nodes.push(remaining[0]);
      remaining = remaining.slice(1);
    } else {
      nodes.push(remaining.slice(0, nextSpecial));
      remaining = remaining.slice(nextSpecial);
    }
  }

  return nodes;
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
        ) : isUser ? (
          <pre className="message-text">{message.content}</pre>
        ) : (
          renderMarkdown(message.content)
        )}
      </div>
    </div>
  );
}
