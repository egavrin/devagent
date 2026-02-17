/**
 * useCommands — manages plugin commands via the engine bridge.
 */

import { useCallback, useEffect, useState } from "react";
import type { CommandInfo } from "../types";
import type { EngineBridgeResult } from "./useEngineBridge";

interface UseCommandsResult {
  readonly commands: ReadonlyArray<CommandInfo>;
  readonly loading: boolean;
  executeCommand: (name: string, args: string) => void;
  refresh: () => void;
}

export function useCommands(bridge: EngineBridgeResult): UseCommandsResult {
  const [commands, setCommands] = useState<CommandInfo[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const unsub = bridge.onMessage("commands_list", (data) => {
      setCommands((data["commands"] as CommandInfo[]) ?? []);
      setLoading(false);
    });

    return unsub;
  }, [bridge]);

  const executeCommand = useCallback(
    (name: string, args: string) => {
      bridge.sendRaw({ type: "execute_command", command: name, args });
    },
    [bridge],
  );

  const refresh = useCallback(() => {
    setLoading(true);
    bridge.sendRaw({ type: "list_commands" });
  }, [bridge]);

  useEffect(() => {
    if (bridge.state.ready) {
      refresh();
    }
  }, [bridge.state.ready, refresh]);

  return { commands, loading, executeCommand, refresh };
}
