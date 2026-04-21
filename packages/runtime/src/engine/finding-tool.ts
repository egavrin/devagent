/**
 * save_finding — Persist analysis findings that survive context compaction.
 *
 * When the LLM analyzes code, diffs, or tool output, it should call
 * save_finding to store conclusions. After compaction, these findings
 * appear in the session state system message so the LLM retains its
 * analysis without needing to re-read the source material.
 */

import type { SessionState } from "./session-state.js";
import type { ToolSpec } from "../core/index.js";

const FINDING_PARAM_SCHEMA = {
  type: "object",
  properties: {
    title: {
      type: "string",
      description:
        "Short title for the finding (e.g., 'SQL injection in login handler'). " +
        "Used for deduplication — saving with the same title updates the existing finding.",
    },
    detail: {
      type: "string",
      description:
        "Detailed description of the finding. Include: what was found, where (file + line), " +
        "severity, and any recommendations. Max 500 chars.",
    },
  },
  required: ["title", "detail"],
};

export function createFindingTool(
  getSessionState: () => SessionState | undefined,
  getIteration: () => number,
): ToolSpec {
  return {
    name: "save_finding",
    description:
      "Persist an analysis finding that survives context compaction. " +
      "Call this after analyzing code, diffs, or tool output to save your conclusions. " +
      "After compaction, saved findings appear in context so you don't need to re-read files. " +
      "Use for: bug reports, code review issues, architecture observations, security findings.",
    category: "state",
    paramSchema: FINDING_PARAM_SCHEMA,
    resultSchema: {
      type: "object",
      properties: {
        saved: { type: "boolean" },
      },
    },
    handler: async (params) => {
      const title = params["title"] as string;
      const detail = params["detail"] as string;

      if (!title.trim()) {
        return {
          success: false,
          output: "",
          error: "Finding title cannot be empty.",
          artifacts: [],
        };
      }

      const sessionState = getSessionState();
      if (!sessionState) {
        return {
          success: false,
          output: "",
          error: "Session state not available. Finding not saved.",
          artifacts: [],
        };
      }

      sessionState.addFinding(title.trim(), detail.trim(), getIteration());

      return {
        success: true,
        output: `Finding saved: "${title.trim()}". This will survive context compaction.`,
        error: null,
        artifacts: [],
      };
    },
  };
}
