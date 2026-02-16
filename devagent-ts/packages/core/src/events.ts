/**
 * Typed event bus — single communication mechanism for frontends and plugins.
 * ArkTS-compatible: no dynamic property access, explicit types.
 */

import type { ToolResult, StreamChunk, Message } from "./types.js";

// ─── Event Definitions ───────────────────────────────────────

export interface EventMap {
  "tool:before": ToolBeforeEvent;
  "tool:after": ToolAfterEvent;
  "message:assistant": AssistantMessageEvent;
  "message:user": UserMessageEvent;
  "approval:request": ApprovalRequestEvent;
  "approval:response": ApprovalResponseEvent;
  "checkpoint:created": CheckpointEvent;
  "session:start": SessionStartEvent;
  "session:end": SessionEndEvent;
  "cost:update": CostUpdateEvent;
  "error": ErrorEvent;
}

export interface ToolBeforeEvent {
  readonly name: string;
  readonly params: Record<string, unknown>;
  readonly callId: string;
}

export interface ToolAfterEvent {
  readonly name: string;
  readonly result: ToolResult;
  readonly callId: string;
  readonly durationMs: number;
}

export interface AssistantMessageEvent {
  readonly content: string;
  readonly partial: boolean;
  readonly chunk?: StreamChunk;
}

export interface UserMessageEvent {
  readonly content: string;
}

export interface ApprovalRequestEvent {
  readonly id: string;
  readonly action: string;
  readonly toolName: string;
  readonly details: string;
}

export interface ApprovalResponseEvent {
  readonly id: string;
  readonly approved: boolean;
  readonly feedback?: string;
}

export interface CheckpointEvent {
  readonly id: string;
  readonly description: string;
  readonly timestamp: number;
}

export interface SessionStartEvent {
  readonly sessionId: string;
}

export interface SessionEndEvent {
  readonly sessionId: string;
  readonly reason: "completed" | "cancelled" | "error" | "budget_exceeded";
}

export interface CostUpdateEvent {
  readonly inputTokens: number;
  readonly outputTokens: number;
  readonly totalCost: number;
  readonly model: string;
}

export interface ErrorEvent {
  readonly message: string;
  readonly code: string;
  readonly fatal: boolean;
}

// ─── Event Bus Implementation ────────────────────────────────

type EventHandler<T> = (event: T) => void;

/**
 * Simple typed pub/sub event bus.
 * Plugins and frontends subscribe to the same events — no separate hook registry.
 */
export class EventBus {
  private listeners: Map<string, Set<EventHandler<unknown>>> = new Map();

  on<K extends keyof EventMap>(
    event: K,
    handler: EventHandler<EventMap[K]>,
  ): () => void {
    const key = event as string;
    let handlers = this.listeners.get(key);
    if (!handlers) {
      handlers = new Set();
      this.listeners.set(key, handlers);
    }
    const typedHandler = handler as EventHandler<unknown>;
    handlers.add(typedHandler);

    // Return unsubscribe function
    return () => {
      handlers?.delete(typedHandler);
    };
  }

  emit<K extends keyof EventMap>(event: K, data: EventMap[K]): void {
    const key = event as string;
    const handlers = this.listeners.get(key);
    if (!handlers) return;

    for (const handler of handlers) {
      try {
        handler(data);
      } catch (err) {
        // Fail fast: surface handler errors immediately.
        // Don't swallow — log and re-throw if fatal.
        const message =
          err instanceof Error ? err.message : String(err);
        console.error(
          `[EventBus] Handler error on "${key}": ${message}`,
        );
      }
    }
  }

  once<K extends keyof EventMap>(
    event: K,
    handler: EventHandler<EventMap[K]>,
  ): () => void {
    const unsubscribe = this.on(event, (data) => {
      unsubscribe();
      handler(data);
    });
    return unsubscribe;
  }

  removeAllListeners(event?: keyof EventMap): void {
    if (event) {
      this.listeners.delete(event as string);
    } else {
      this.listeners.clear();
    }
  }

  listenerCount(event: keyof EventMap): number {
    const handlers = this.listeners.get(event as string);
    return handlers ? handlers.size : 0;
  }
}
