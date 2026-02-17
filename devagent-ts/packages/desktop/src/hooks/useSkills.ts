/**
 * useSkills — manages skills discovery and loading via the engine bridge.
 */

import { useCallback, useEffect, useState } from "react";
import type { SkillInfo } from "../types";
import type { EngineBridgeResult } from "./useEngineBridge";

interface UseSkillsResult {
  readonly skills: ReadonlyArray<SkillInfo>;
  readonly loading: boolean;
  readonly loadedInstructions: Record<string, string>;
  refresh: () => void;
  loadSkill: (name: string) => void;
}

export function useSkills(bridge: EngineBridgeResult): UseSkillsResult {
  const [skills, setSkills] = useState<SkillInfo[]>([]);
  const [loading, setLoading] = useState(false);
  const [loadedInstructions, setLoadedInstructions] = useState<Record<string, string>>({});

  useEffect(() => {
    const unsub1 = bridge.onMessage("skills_list", (data) => {
      setSkills((data["skills"] as SkillInfo[]) ?? []);
      setLoading(false);
    });

    const unsub2 = bridge.onMessage("skill_loaded", (data) => {
      const name = data["name"] as string;
      const instructions = data["instructions"] as string;
      setLoadedInstructions((prev) => ({ ...prev, [name]: instructions }));
    });

    return () => {
      unsub1();
      unsub2();
    };
  }, [bridge]);

  const refresh = useCallback(() => {
    setLoading(true);
    bridge.sendRaw({ type: "list_skills" });
  }, [bridge]);

  const loadSkill = useCallback(
    (name: string) => {
      bridge.sendRaw({ type: "load_skill", name });
    },
    [bridge],
  );

  // Auto-fetch on mount when ready
  useEffect(() => {
    if (bridge.state.ready) {
      refresh();
    }
  }, [bridge.state.ready, refresh]);

  return { skills, loading, loadedInstructions, refresh, loadSkill };
}
