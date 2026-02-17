/**
 * useEngineBridge — connects the desktop UI to the DevAgent engine.
 *
 * Spawns the CLI with `--desktop` flag via Tauri's shell plugin.
 * Communicates via JSON-lines protocol over stdin/stdout.
 *
 * Other hooks subscribe to specific message types via onMessage().
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  ChatMessage,
  AppMode,
  ToolExecution,
  ApprovalRequest,
  CostState,
} from "../types";

// ─── Types ─────────────────────────────────────────────────

interface BridgeState {
  readonly connected: boolean;
  readonly streaming: boolean;
  readonly error: string | null;
  readonly ready: boolean;
}

export type MessageHandler = (data: Record<string, unknown>) => void;

export interface EngineBridgeResult {
  readonly state: BridgeState;
  readonly messages: ReadonlyArray<ChatMessage>;
  readonly tools: ReadonlyArray<ToolExecution>;
  readonly approvalRequests: ReadonlyArray<ApprovalRequest>;
  readonly costState: CostState;
  readonly workingDir: string;
  sendQuery: (content: string, mode: AppMode) => void;
  sendRaw: (data: Record<string, unknown>) => void;
  onMessage: (type: string, handler: MessageHandler) => () => void;
  abort: () => void;
  clearMessages: () => void;
  setProvider: (provider: string, model: string, apiKey?: string) => void;
  setApproval: (mode: string) => void;
  respondApproval: (id: string, approved: boolean) => void;
}

// Reference to the spawned child process
interface ChildProcess {
  write: (data: string) => Promise<void>;
  kill: () => Promise<void>;
}

// ─── Tauri Detection ───────────────────────────────────────

const checkIsTauri = (): boolean => {
  if (typeof window === "undefined") return false;
  return "__TAURI_INTERNALS__" in window || "__TAURI__" in window;
};

// ─── Hook ──────────────────────────────────────────────────

export function useEngineBridge(): EngineBridgeResult {
  const [state, setState] = useState<BridgeState>({
    connected: false,
    streaming: false,
    error: null,
    ready: false,
  });
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [tools, setTools] = useState<ToolExecution[]>([]);
  const [approvalRequests, setApprovalRequests] = useState<ApprovalRequest[]>([]);
  const [costState, setCostState] = useState<CostState>({
    inputTokens: 0,
    outputTokens: 0,
    totalCost: 0,
  });
  const [workingDir, setWorkingDir] = useState<string>("");

  const childRef = useRef<ChildProcess | null>(null);
  const currentAssistantId = useRef<string | null>(null);

  // Pub/sub for message types — other hooks register handlers here
  const messageHandlers = useRef<Map<string, Set<MessageHandler>>>(new Map());

  // ─── Pub/Sub API ──────────────────────────────────────────

  const onMessage = useCallback(
    (type: string, handler: MessageHandler): (() => void) => {
      const handlers = messageHandlers.current;
      if (!handlers.has(type)) {
        handlers.set(type, new Set());
      }
      handlers.get(type)!.add(handler);

      // Return unsubscribe function
      return () => {
        const set = handlers.get(type);
        if (set) {
          set.delete(handler);
          if (set.size === 0) handlers.delete(type);
        }
      };
    },
    [],
  );

  // ─── Message Processing ───────────────────────────────────

  const appendToAssistant = useCallback((text: string) => {
    const id = currentAssistantId.current;
    if (!id) return;
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === id ? { ...msg, content: msg.content + text } : msg,
      ),
    );
  }, []);

  const finishAssistant = useCallback((_iterations?: number) => {
    const id = currentAssistantId.current;
    if (!id) return;
    setMessages((prev) =>
      prev.map((msg) =>
        msg.id === id ? { ...msg, isStreaming: false } : msg,
      ),
    );
    currentAssistantId.current = null;
    setState((prev) => ({ ...prev, streaming: false }));
  }, []);

  const handleLine = useCallback(
    (line: string) => {
      const trimmed = line.trim();
      if (!trimmed) return;

      let data: Record<string, unknown>;
      try {
        data = JSON.parse(trimmed) as Record<string, unknown>;
      } catch {
        appendToAssistant(trimmed + "\n");
        return;
      }

      const msgType = data["type"] as string;

      // Dispatch to registered handlers first
      const handlers = messageHandlers.current.get(msgType);
      if (handlers) {
        for (const handler of handlers) {
          handler(data);
        }
      }

      // Built-in handling
      switch (msgType) {
        case "ready":
          setState((prev) => ({ ...prev, connected: true, ready: true }));
          break;

        case "text":
          appendToAssistant(data["content"] as string);
          break;

        case "tool_start": {
          const toolExec: ToolExecution = {
            id: (data["callId"] as string) ?? crypto.randomUUID(),
            name: data["name"] as string,
            params: (data["params"] as Record<string, unknown>) ?? {},
            status: "running",
            timestamp: Date.now(),
          };
          setTools((prev) => [...prev, toolExec]);
          appendToAssistant(`\n🔧 ${data["name"] as string}\n`);
          break;
        }

        case "tool_end": {
          const callId = data["callId"] as string;
          const success = data["success"] as boolean;
          const error = data["error"] as string | null;
          const durationMs = data["durationMs"] as number;

          setTools((prev) =>
            prev.map((t) =>
              t.id === callId
                ? {
                    ...t,
                    status: success ? ("done" as const) : ("error" as const),
                    result: success ? `✓ (${durationMs}ms)` : undefined,
                    error: error ?? undefined,
                    durationMs,
                  }
                : t,
            ),
          );

          if (error) {
            appendToAssistant(`❌ Error: ${error}\n`);
          }
          break;
        }

        case "approval_request": {
          const request: ApprovalRequest = {
            id: data["id"] as string,
            toolName: data["toolName"] as string,
            details: data["details"] as string,
            timestamp: Date.now(),
          };
          setApprovalRequests((prev) => [...prev, request]);
          appendToAssistant(`\n⚠️ Waiting for approval: ${request.toolName}\n`);
          break;
        }

        case "done":
          finishAssistant(data["iterations"] as number | undefined);
          break;

        case "error": {
          const errMsg = data["message"] as string;
          const fatal = data["fatal"] as boolean;
          if (fatal) {
            setState((prev) => ({ ...prev, error: errMsg }));
          }
          appendToAssistant(`\n❌ ${errMsg}\n`);
          if (currentAssistantId.current) {
            finishAssistant();
          }
          break;
        }

        case "cost_update":
          setCostState({
            inputTokens: (data["inputTokens"] as number) ?? 0,
            outputTokens: (data["outputTokens"] as number) ?? 0,
            totalCost: (data["totalCost"] as number) ?? 0,
          });
          break;

        case "working_dir":
          setWorkingDir((data["dir"] ?? data["path"]) as string);
          break;

        case "mode_changed":
        case "provider_changed":
        case "approval_changed":
          break;

        default:
          // Unknown messages are dispatched to handlers above, no built-in handling needed
          break;
      }
    },
    [appendToAssistant, finishAssistant],
  );

  // ─── Send to child process ────────────────────────────────

  const sendToChild = useCallback(
    async (data: Record<string, unknown>) => {
      const child = childRef.current;
      if (!child) return;
      try {
        await child.write(JSON.stringify(data) + "\n");
      } catch {
        setState((prev) => ({
          ...prev,
          error: "Failed to send message to engine",
        }));
      }
    },
    [],
  );

  // ─── Spawn CLI Process ────────────────────────────────────

  useEffect(() => {
    const isTauri = checkIsTauri();

    if (!isTauri) {
      setState({ connected: true, streaming: false, error: null, ready: true });
      setWorkingDir(window.location.pathname || "/tmp");
      return;
    }

    let killed = false;

    void (async () => {
      try {
        const { Command } = await import("@tauri-apps/plugin-shell");
        const { invoke } = await import("@tauri-apps/api/core");

        let cliPath: string;
        try {
          cliPath = await invoke<string>("get_cli_path");
        } catch (pathErr) {
          const msg = pathErr instanceof Error ? pathErr.message : String(pathErr);
          setState((prev) => ({
            ...prev,
            connected: false,
            error: `CLI not found: ${msg}`,
          }));
          return;
        }

        // Get initial working directory
        try {
          const dir = await invoke<string>("get_working_directory");
          setWorkingDir(dir);
        } catch {
          // Fallback — will be set by the engine
        }

        const cmd = Command.create("devagent-cli", [cliPath, "--desktop"]);

        let stdoutBuffer = "";

        cmd.stdout.on("data", (chunk: string) => {
          stdoutBuffer += chunk;
          const lines = stdoutBuffer.split("\n");
          stdoutBuffer = lines.pop() ?? "";
          for (const line of lines) {
            if (line.trim()) {
              handleLine(line);
            }
          }
        });

        cmd.stderr.on("data", (chunk: string) => {
          console.warn("[engine stderr]", chunk);
        });

        cmd.on("error", (error: string) => {
          setState((prev) => ({
            ...prev,
            connected: false,
            error: `Engine process error: ${error}`,
          }));
        });

        cmd.on("close", (data: { code: number | null }) => {
          if (!killed) {
            setState((prev) => ({
              ...prev,
              connected: false,
              ready: false,
              error: data.code !== 0 ? `Engine exited with code ${data.code}` : null,
            }));
          }
        });

        const child = await cmd.spawn();

        childRef.current = {
          write: async (data: string) => {
            await child.write(data);
          },
          kill: async () => {
            killed = true;
            await child.kill();
          },
        };

        setState((prev) => ({ ...prev, connected: true }));
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setState((prev) => ({
          ...prev,
          connected: false,
          error: `Failed to start engine: ${msg}`,
        }));
      }
    })();

    return () => {
      if (childRef.current) {
        void childRef.current.kill();
        childRef.current = null;
      }
    };
  }, [handleLine]);

  // ─── Public API ───────────────────────────────────────────

  const sendRaw = useCallback(
    (data: Record<string, unknown>) => {
      void sendToChild(data);
    },
    [sendToChild],
  );

  const sendQuery = useCallback(
    (content: string, mode: AppMode) => {
      if (state.streaming) return;

      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        timestamp: Date.now(),
      };

      const assistantId = crypto.randomUUID();
      const assistantMsg: ChatMessage = {
        id: assistantId,
        role: "assistant",
        content: "",
        timestamp: Date.now(),
        isStreaming: true,
      };

      currentAssistantId.current = assistantId;
      setMessages((prev) => [...prev, userMsg, assistantMsg]);
      setTools([]);
      // Reset cost per query
      setCostState({ inputTokens: 0, outputTokens: 0, totalCost: 0 });
      setState((prev) => ({ ...prev, streaming: true, error: null }));

      if (checkIsTauri() && childRef.current) {
        void sendToChild({ type: "query", content, mode });
      } else {
        setTimeout(() => {
          appendToAssistant(
            `Simulated response (browser dev mode).\n\n` +
              `**Mode:** ${mode}\n` +
              `**Query:** "${content}"\n\n` +
              `Run \`cargo tauri dev\` for real engine responses.`,
          );
          finishAssistant();
        }, 800);
      }
    },
    [state.streaming, appendToAssistant, finishAssistant, sendToChild],
  );

  const abort = useCallback(() => {
    if (checkIsTauri() && childRef.current) {
      void sendToChild({ type: "abort" });
    }
    finishAssistant();
  }, [finishAssistant, sendToChild]);

  const clearMessages = useCallback(() => {
    setMessages([]);
    setTools([]);
    currentAssistantId.current = null;
  }, []);

  const setProvider = useCallback(
    (provider: string, model: string, apiKey?: string) => {
      void sendToChild({
        type: "set_provider",
        provider,
        model,
        ...(apiKey ? { apiKey } : {}),
      });
    },
    [sendToChild],
  );

  const setApproval = useCallback(
    (mode: string) => {
      void sendToChild({ type: "set_approval", mode });
    },
    [sendToChild],
  );

  const respondApproval = useCallback(
    (id: string, approved: boolean) => {
      void sendToChild({ type: "approval_response", id, approved });
      setApprovalRequests((prev) => prev.filter((r) => r.id !== id));
      appendToAssistant(approved ? `\n✅ Approved\n` : `\n🚫 Denied\n`);
    },
    [sendToChild, appendToAssistant],
  );

  return {
    state,
    messages,
    tools,
    approvalRequests,
    costState,
    workingDir,
    sendQuery,
    sendRaw,
    onMessage,
    abort,
    clearMessages,
    setProvider,
    setApproval,
    respondApproval,
  };
}
