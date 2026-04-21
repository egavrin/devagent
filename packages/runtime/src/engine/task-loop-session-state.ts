import { SESSION_STATE_MARKER } from "./session-state.js";
import type { SessionState } from "./session-state.js";
import type { Message } from "../core/index.js";
import { MessageRole, estimateMessageTokens } from "../core/index.js";

type SessionStateTier = "full" | "compact" | "minimal";

interface SessionStateInjectionHost {
  readonly sessionState: SessionState | null;
  messages: Message[];
  estimatedTokens: number;
  getEffectiveContextBudget(): number;
}

export function injectTaskLoopSessionState(
  loop: SessionStateInjectionHost,
  knownTokenEstimate?: number,
): void {
  if (!loop.sessionState) return;

  const content = loop.sessionState.toSystemMessage(getSessionStateTier(loop, knownTokenEstimate));
  if (!content) return;

  removeExistingSessionStateMessage(loop);
  const message: Message = { role: MessageRole.SYSTEM, content };
  const insertIndex = loop.messages[0]?.role === MessageRole.SYSTEM ? 1 : 0;
  loop.messages.splice(insertIndex, 0, message);
  loop.estimatedTokens += estimateMessageTokens([message]);
}

function getSessionStateTier(
  loop: SessionStateInjectionHost,
  knownTokenEstimate?: number,
): SessionStateTier {
  const maxBudget = loop.getEffectiveContextBudget();
  if (maxBudget <= 0) return "full";
  const totalEstimate = knownTokenEstimate ?? loop.estimatedTokens;
  const headroom = maxBudget - totalEstimate;
  if (headroom > 8000) return "full";
  if (headroom > 3000) return "compact";
  return "minimal";
}

function removeExistingSessionStateMessage(loop: SessionStateInjectionHost): void {
  const previousMessages = loop.messages;
  loop.messages = [];
  for (const message of previousMessages) {
    if (isSessionStateMessage(message)) {
      loop.estimatedTokens -= estimateMessageTokens([message]);
    } else {
      loop.messages.push(message);
    }
  }
}

function isSessionStateMessage(message: Message): boolean {
  return message.role === MessageRole.SYSTEM && Boolean(message.content?.startsWith(SESSION_STATE_MARKER));
}
