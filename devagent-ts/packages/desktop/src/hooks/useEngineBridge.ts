/**
 * useEngineBridge — connects the desktop UI to the DevAgent engine.
 *
 * Spawns the CLI with `--desktop` flag via Tauri's shell plugin.
 * Communicates via JSON-lines protocol over stdin/stdout.
 *
 * Incoming from CLI (stdout):
 *   {"type":"ready"}
 *   {"type":"text","content":"partial text..."}
 *   {"type":"tool_start","name":"read_file","callId":"...","params":{}}
 *   {"type":"tool_end","name":"read_file","callId":"...","success":true,"output":"..."}
 *   {"type":"approval_request","id":"...","toolName":"write_file","details":"..."}
 *   {"type":"done","iterations":3}
 *   {"type":"error","message":"...","fatal":false}
 *
 * Outgoing to CLI (stdin):
 *   {"type":"query","content":"...","mode":"act"}
 *   {"type":"set_mode","mode":"plan"}
 *   {"type":"set_provider","provider":"anthropic","model":"...","apiKey":"..."}
 *   {"type":"abort"}
 *   {"type":"approval_response","id":"...","approved":true}
 */

import { useCallback, useEffect, useRef, useState } from "react";
import type { ChatMessage, AppMode, ToolExecution, ApprovalRequest } from "../types";

interface BridgeState {
  readonly connected: boolean;
  readonly streaming: boolean;
  readonly error: string | null;
  readonly ready: boolean;
}

interface EngineBridgeResult {
  readonly state: BridgeState;
  readonly messages: ReadonlyArray<ChatMessage>;
  readonly tools: ReadonlyArray<ToolExecution>;
  readonly approvalRequests: ReadonlyArray<ApprovalRequest>;
  sendQuery: (content: string, mode: AppMode) => void;
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

/**
 * Bridge to the DevAgent engine via CLI subprocess.
 *
 * In Tauri mode: spawns CLI subprocess with --desktop flag.
 * In browser mode: simulates responses for development.
 */
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

  const childRef = useRef<ChildProcess | null>(null);
  const currentAssistantId = useRef<string | null>(null);

  // Detect Tauri v2 environment.
  // Tauri v2 always injects __TAURI_INTERNALS__ into the WebView.
  // __TAURI__ is only available with "withGlobalTauri": true in tauri.conf.json.
  // Check both for maximum reliability.
  const checkIsTauri = (): boolean => {
    if (typeof window === "undefined") return false;
    const hasTauri =
      "__TAURI_INTERNALS__" in window ||
      "__TAURI__" in window;
    return hasTauri;
  };

  // ─── Message Processing ─────────────────────────────────

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
        // Non-JSON output from CLI (e.g. stderr leak) — append as text
        appendToAssistant(trimmed + "\n");
        return;
      }

      const msgType = data["type"] as string;

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

          // Also show tool call in chat as a system-style message
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
                    result: success
                      ? `✓ (${durationMs}ms)`
                      : undefined,
                    error: error ?? undefined,
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

        case "mode_changed":
        case "provider_changed":
        case "approval_changed":
        case "cost_update":
          // Status updates — could be displayed in UI
          break;

        default:
          // Unknown message type — ignore
          break;
      }
    },
    [appendToAssistant, finishAssistant],
  );

  // ─── Send to child process ──────────────────────────────

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

  // ─── Spawn CLI Process ──────────────────────────────────

  useEffect(() => {
    const isTauri = checkIsTauri();
    console.log("[EngineBridge] checkIsTauri:", isTauri,
      "__TAURI_INTERNALS__" in (typeof window !== "undefined" ? window : {}),
      "__TAURI__" in (typeof window !== "undefined" ? window : {}));

    if (!isTauri) {
      // Browser dev mode — mark as ready with simulated connection
      console.warn("[EngineBridge] Not in Tauri — entering browser simulation mode");
      setState({ connected: true, streaming: false, error: null, ready: true });
      return;
    }

    console.log("[EngineBridge] Tauri detected — spawning CLI subprocess");
    let killed = false;

    void (async () => {
      try {
        const { Command } = await import("@tauri-apps/plugin-shell");

        // Resolve CLI path via Rust backend (searches monorepo structure)
        const { invoke } = await import("@tauri-apps/api/core");
        let cliPath: string;
        try {
          cliPath = await invoke<string>("get_cli_path");
          console.log("[EngineBridge] CLI path resolved:", cliPath);
        } catch (pathErr) {
          const msg = pathErr instanceof Error ? pathErr.message : String(pathErr);
          console.error("[EngineBridge] CLI path resolution failed:", msg);
          setState((prev) => ({
            ...prev,
            connected: false,
            error: `CLI not found: ${msg}`,
          }));
          return;
        }

        // Spawn CLI with --desktop flag
        // "devagent-cli" is a scoped command in capabilities/default.json
        // that maps to `bun <resolved-cli-path> --desktop`
        console.log("[EngineBridge] Spawning: bun", cliPath, "--desktop");
        const cmd = Command.create("devagent-cli", [
          cliPath,
          "--desktop",
        ]);

        // Buffer for incomplete lines
        let stdoutBuffer = "";

        cmd.stdout.on("data", (chunk: string) => {
          console.log("[engine stdout]", chunk.substring(0, 200));
          stdoutBuffer += chunk;
          const lines = stdoutBuffer.split("\n");
          // Process all complete lines
          stdoutBuffer = lines.pop() ?? "";
          for (const line of lines) {
            if (line.trim()) {
              handleLine(line);
            }
          }
        });

        cmd.stderr.on("data", (chunk: string) => {
          // stderr from CLI — log as error context
          console.warn("[engine stderr]", chunk);
        });

        cmd.on("error", (error: string) => {
          console.error("[EngineBridge] Process error:", error);
          setState((prev) => ({
            ...prev,
            connected: false,
            error: `Engine process error: ${error}`,
          }));
        });

        cmd.on("close", (data: { code: number | null }) => {
          console.log("[EngineBridge] Process closed with code:", data.code);
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
        console.log("[EngineBridge] CLI subprocess spawned successfully");

        childRef.current = {
          write: async (data: string) => {
            console.log("[EngineBridge] Writing to stdin:", data.substring(0, 100));
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
        console.error("[EngineBridge] Failed to start engine:", msg);
        setState((prev) => ({
          ...prev,
          connected: false,
          error: `Failed to start engine: ${msg}`,
        }));
      }
    })();

    return () => {
      // Cleanup on unmount
      if (childRef.current) {
        void childRef.current.kill();
        childRef.current = null;
      }
    };
  }, [handleLine]);

  // ─── Public API ─────────────────────────────────────────

  const sendQuery = useCallback(
    (content: string, mode: AppMode) => {
      if (state.streaming) return;

      // Add user message
      const userMsg: ChatMessage = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        timestamp: Date.now(),
      };

      // Create assistant placeholder
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
      setState((prev) => ({ ...prev, streaming: true, error: null }));

      if (checkIsTauri() && childRef.current) {
        // Send query to CLI subprocess
        void sendToChild({ type: "query", content, mode });
      } else {
        // Browser dev mode: simulate response
        setTimeout(() => {
          appendToAssistant(
            `This is a simulated response (browser dev mode).\n\n` +
              `**Mode:** ${mode}\n` +
              `**Query:** "${content}"\n\n` +
              `In the Tauri desktop app, this connects to the DevAgent CLI engine ` +
              `via the \`--desktop\` JSON-lines protocol. The engine provides real ` +
              `LLM-powered responses with tool calling, file operations, and code analysis.\n\n` +
              `To test with a real engine, run: \`cargo-tauri dev\``,
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
    sendQuery,
    abort,
    clearMessages,
    setProvider,
    setApproval,
    respondApproval,
  };
}
