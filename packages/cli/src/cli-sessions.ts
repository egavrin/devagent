import { bold, dim } from "./format.js";
import { buildSessionPreview, renderSessionPreview } from "./session-preview.js";
import type { Session } from "@devagent/runtime";

export function renderSessionsList(sessions: ReadonlyArray<Session>, now: number = Date.now()): string {
  if (sessions.length === 0) return `${dim("No sessions found.")}\n`;
  const lines = [bold("Recent sessions:"), ""];
  for (const session of sessions) lines.push(renderSessionPreview(buildSessionPreview(session), now));
  lines.push("", dim("Use --resume <full-id-or-unique-prefix> to continue a session."));
  return lines.join("\n") + "\n";
}
