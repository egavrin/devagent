import { createInterface } from "node:readline";

import { bold, dim, formatTranscriptPart, yellow } from "./format.js";
import { nextTranscriptId, spinner, transcriptComposer } from "./main-state.js";
import type { StatusLine } from "./status-line.js";
import {
  presentApprovalRequestEvent,
  presentApprovalResponseEvent,
} from "./transcript-presenter.js";
import type { EventBus, EventMap } from "@devagent/runtime";

export function registerApprovalEvents(
  bus: EventBus,
  statusLine: StatusLine | null,
  writeUiLine: (line: string) => void,
): void {
  bus.on("approval:request", (event) => handleApprovalRequest(bus, statusLine, event));
  bus.on("approval:response", (event) => {
    const part = presentApprovalResponseEvent(event);
    transcriptComposer.appendPart(nextTranscriptId("approval"), part);
    const rendered = formatTranscriptPart(part);
    if (rendered) writeUiLine(rendered);
  });
}

function handleApprovalRequest(
  bus: EventBus,
  statusLine: StatusLine | null,
  event: EventMap["approval:request"],
): void {
  spinner.stop();
  statusLine?.suspend();
  const approvalPart = presentApprovalRequestEvent(event);
  if (approvalPart.kind !== "approval") return;
  transcriptComposer.appendPart(nextTranscriptId("approval"), approvalPart);
  writeApprovalBox(approvalPart.data.toolName, approvalPart.data.details);
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  rl.question(yellow("  Approve? [y]once / [n]o / [s]ession: "), (answer) => {
    rl.close();
    emitApprovalResponse(bus, statusLine, event.id, answer);
  });
}

function writeApprovalBox(toolName: string, details: string): void {
  const border = yellow("─".repeat(60));
  const toolLine = `  ${bold(toolName)}`;
  const detailLine = `  ${dim(details.length > 56 ? details.slice(0, 53) + "..." : details)}`;
  process.stderr.write(`\n${border}\n${toolLine}\n${detailLine}\n${border}\n`);
}

function emitApprovalResponse(
  bus: EventBus,
  statusLine: StatusLine | null,
  id: string,
  answer: string,
): void {
  const lower = answer.toLowerCase().trim();
  bus.emit("approval:response", {
    id,
    approved: lower.startsWith("y") || lower.startsWith("s"),
    feedback: lower.startsWith("s") ? "session" : undefined,
  });
  statusLine?.resume();
}
