/**
 * useDiffs — collects file diffs from tool executions.
 *
 * Subscribes to `file_diff` messages from the bridge and manages
 * accept/reject state for each diff.
 */

import { useCallback, useEffect, useState } from "react";
import type { FileDiff, DiffHunk, DiffLine } from "../types";
import type { EngineBridgeResult } from "./useEngineBridge";

interface UseDiffsResult {
  readonly diffs: ReadonlyArray<FileDiff>;
  readonly hasPending: boolean;
  acceptDiff: (id: string) => void;
  rejectDiff: (id: string) => void;
  acceptAll: () => void;
  clearResolved: () => void;
}

/**
 * Parse a unified diff string into structured hunks.
 */
function parseUnifiedDiff(diffText: string): DiffHunk[] {
  const lines = diffText.split("\n");
  const hunks: DiffHunk[] = [];
  let currentHunk: DiffHunk | null = null;
  let oldLine = 0;
  let newLine = 0;

  for (const line of lines) {
    // Hunk header: @@ -old,count +new,count @@
    const hunkMatch = line.match(/^@@\s+-(\d+)(?:,(\d+))?\s+\+(\d+)(?:,(\d+))?\s+@@(.*)$/);
    if (hunkMatch) {
      const oldStart = parseInt(hunkMatch[1]!, 10);
      const oldCount = parseInt(hunkMatch[2] ?? "1", 10);
      const newStart = parseInt(hunkMatch[3]!, 10);
      const newCount = parseInt(hunkMatch[4] ?? "1", 10);

      currentHunk = {
        header: line,
        oldStart,
        newStart,
        oldCount,
        newCount,
        lines: [],
      };
      hunks.push(currentHunk);
      oldLine = oldStart;
      newLine = newStart;
      continue;
    }

    if (!currentHunk) continue;

    // Skip diff headers (--- a/file, +++ b/file, diff --git, index)
    if (line.startsWith("---") || line.startsWith("+++") || line.startsWith("diff ") || line.startsWith("index ")) {
      continue;
    }

    let diffLine: DiffLine;
    if (line.startsWith("+")) {
      diffLine = { type: "add", content: line.substring(1), newLineNumber: newLine++ };
    } else if (line.startsWith("-")) {
      diffLine = { type: "remove", content: line.substring(1), oldLineNumber: oldLine++ };
    } else {
      // Context line (starts with space or is empty)
      diffLine = {
        type: "context",
        content: line.startsWith(" ") ? line.substring(1) : line,
        oldLineNumber: oldLine++,
        newLineNumber: newLine++,
      };
    }

    currentHunk = {
      ...currentHunk,
      lines: [...currentHunk.lines, diffLine],
    };
    // Update the hunk reference in the array
    hunks[hunks.length - 1] = currentHunk;
  }

  return hunks;
}

export function useDiffs(bridge: EngineBridgeResult): UseDiffsResult {
  const [diffs, setDiffs] = useState<FileDiff[]>([]);

  useEffect(() => {
    const unsub = bridge.onMessage("file_diff", (data) => {
      const filePath = data["filePath"] as string;
      const diffText = data["diff"] as string;
      const toolCallId = data["toolCallId"] as string;

      const hunks = parseUnifiedDiff(diffText);

      const fileDiff: FileDiff = {
        id: crypto.randomUUID(),
        filePath,
        hunks,
        status: "pending",
        toolCallId,
        timestamp: Date.now(),
      };

      setDiffs((prev) => [...prev, fileDiff]);
    });

    return unsub;
  }, [bridge]);

  const hasPending = diffs.some((d) => d.status === "pending");

  const acceptDiff = useCallback((id: string) => {
    setDiffs((prev) =>
      prev.map((d) => (d.id === id ? { ...d, status: "accepted" as const } : d)),
    );
  }, []);

  const rejectDiff = useCallback((id: string) => {
    setDiffs((prev) =>
      prev.map((d) => (d.id === id ? { ...d, status: "rejected" as const } : d)),
    );
  }, []);

  const acceptAll = useCallback(() => {
    setDiffs((prev) =>
      prev.map((d) =>
        d.status === "pending" ? { ...d, status: "accepted" as const } : d,
      ),
    );
  }, []);

  const clearResolved = useCallback(() => {
    setDiffs((prev) => prev.filter((d) => d.status === "pending"));
  }, []);

  return { diffs, hasPending, acceptDiff, rejectDiff, acceptAll, clearResolved };
}
