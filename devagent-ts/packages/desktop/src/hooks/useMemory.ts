/**
 * useMemory — manages cross-session memory via the engine bridge.
 */

import { useCallback, useEffect, useState } from "react";
import type { MemoryEntry, MemoryCategory } from "../types";
import type { EngineBridgeResult } from "./useEngineBridge";

interface UseMemoryResult {
  readonly memories: ReadonlyArray<MemoryEntry>;
  readonly summary: Record<MemoryCategory, number> | null;
  readonly loading: boolean;
  search: (query: string, category?: MemoryCategory) => void;
  deleteMemory: (id: string) => void;
  refresh: () => void;
}

export function useMemory(bridge: EngineBridgeResult): UseMemoryResult {
  const [memories, setMemories] = useState<MemoryEntry[]>([]);
  const [summary, setSummary] = useState<Record<MemoryCategory, number> | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    const unsub1 = bridge.onMessage("memories", (data) => {
      setMemories((data["entries"] as MemoryEntry[]) ?? []);
      setLoading(false);
    });

    const unsub2 = bridge.onMessage("memory_summary", (data) => {
      setSummary(data["summary"] as Record<MemoryCategory, number>);
    });

    const unsub3 = bridge.onMessage("memory_deleted", () => {
      // Re-fetch after deletion
      bridge.sendRaw({ type: "search_memories" });
      bridge.sendRaw({ type: "get_memory_summary" });
    });

    return () => {
      unsub1();
      unsub2();
      unsub3();
    };
  }, [bridge]);

  const search = useCallback(
    (query: string, category?: MemoryCategory) => {
      setLoading(true);
      bridge.sendRaw({
        type: "search_memories",
        ...(query ? { query } : {}),
        ...(category ? { category } : {}),
      });
    },
    [bridge],
  );

  const deleteMemory = useCallback(
    (id: string) => {
      bridge.sendRaw({ type: "delete_memory", id });
    },
    [bridge],
  );

  const refresh = useCallback(() => {
    setLoading(true);
    bridge.sendRaw({ type: "search_memories" });
    bridge.sendRaw({ type: "get_memory_summary" });
  }, [bridge]);

  useEffect(() => {
    if (bridge.state.ready) {
      refresh();
    }
  }, [bridge.state.ready, refresh]);

  return { memories, summary, loading, search, deleteMemory, refresh };
}
