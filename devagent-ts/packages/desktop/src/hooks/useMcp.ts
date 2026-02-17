/**
 * useMcp — manages MCP server state via the engine bridge.
 */

import { useCallback, useEffect, useState } from "react";
import type { McpServerInfo } from "../types";
import type { EngineBridgeResult } from "./useEngineBridge";

interface UseMcpResult {
  readonly servers: ReadonlyArray<McpServerInfo>;
  readonly loading: boolean;
  refresh: () => void;
  restartServer: (name: string) => void;
  toggleServer: (name: string, enabled: boolean) => void;
}

export function useMcp(bridge: EngineBridgeResult): UseMcpResult {
  const [servers, setServers] = useState<McpServerInfo[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const unsub = bridge.onMessage("mcp_servers", (data) => {
      setServers((data["servers"] as McpServerInfo[]) ?? []);
      setLoading(false);
    });

    return unsub;
  }, [bridge]);

  const refresh = useCallback(() => {
    setLoading(true);
    bridge.sendRaw({ type: "list_mcp_servers" });
  }, [bridge]);

  const restartServer = useCallback(
    (name: string) => {
      bridge.sendRaw({ type: "restart_mcp_server", name });
    },
    [bridge],
  );

  const toggleServer = useCallback(
    (name: string, enabled: boolean) => {
      bridge.sendRaw({ type: "toggle_mcp_server", name, enabled });
    },
    [bridge],
  );

  useEffect(() => {
    if (bridge.state.ready) {
      refresh();
    }
  }, [bridge.state.ready, refresh]);

  return { servers, loading, refresh, restartServer, toggleServer };
}
