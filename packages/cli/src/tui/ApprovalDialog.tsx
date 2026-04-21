/**
 * ApprovalDialog — Ink-based approval prompt for tool execution.
 * Shows bordered box with tool info, y/n/a keybindings, and optional rejection reason.
 */

import { Box, Text, useInput } from "ink";
import React, { useState } from "react";

export interface ApprovalRequest {
  readonly id: string;
  readonly toolName: string;
  readonly details: string;
}

interface ApprovalDialogProps {
  readonly request: ApprovalRequest;
  readonly onResponse: (approved: boolean, session?: boolean, reason?: string) => void;
}

export function ApprovalDialog({ request, onResponse }: ApprovalDialogProps): React.ReactElement {
  const [mode, setMode] = useState<"choose" | "reject-reason">("choose");
  const [reason, setReason] = useState("");
  useInput((input, key) => {
    if (mode === "choose") {
      handleApprovalChoice(input, key, onResponse, setMode);
      return;
    }
    handleRejectReasonInput({ input, key, reason, setMode, setReason, onResponse });
  });

  const truncatedDetails = request.details.length > 70
    ? request.details.slice(0, 67) + "..."
    : request.details;

  return (
    <Box flexDirection="column" paddingLeft={1} marginTop={1}>
      <Text bold color="yellow">Approve tool execution?</Text>
      <Box marginTop={1}>
        <Text bold>{request.toolName}</Text>
      </Box>
      <Text dimColor>{truncatedDetails}</Text>

      {mode === "choose" && (
        <Box marginTop={1}>
          <Text color="green">[y]</Text><Text>es  </Text>
          <Text color="red">[n]</Text><Text>o  </Text>
          <Text color="cyan">[s]</Text><Text>ession  </Text>
          <Text dimColor>Esc cancel</Text>
        </Box>
      )}

      {mode === "reject-reason" && (
        <Box flexDirection="column" marginTop={1}>
          <Text dimColor>Why reject? (Enter to submit, Esc to go back)</Text>
          <Box>
            <Text color="red">&gt; </Text>
            <Text>{reason || " "}</Text>
          </Box>
        </Box>
      )}
    </Box>
  );
}

function handleApprovalChoice(
  input: string,
  key: { readonly escape?: boolean },
  onResponse: ApprovalDialogProps["onResponse"],
  setMode: React.Dispatch<React.SetStateAction<"choose" | "reject-reason">>,
): void {
  const choice = input.toLowerCase();
  if (choice === "y") onResponse(true);
  else if (choice === "s") onResponse(true, true);
  else if (choice === "n") setMode("reject-reason");
  else if (key.escape) onResponse(false);
}

interface RejectReasonInputOptions {
  readonly input: string;
  readonly key: { readonly return?: boolean; readonly escape?: boolean; readonly backspace?: boolean; readonly delete?: boolean; readonly ctrl?: boolean; readonly meta?: boolean };
  readonly reason: string;
  readonly setMode: React.Dispatch<React.SetStateAction<"choose" | "reject-reason">>;
  readonly setReason: React.Dispatch<React.SetStateAction<string>>;
  readonly onResponse: ApprovalDialogProps["onResponse"];
}

function handleRejectReasonInput(options: RejectReasonInputOptions): void {
  const { input, key, reason, setMode, setReason, onResponse } = options;
  if (key.return) {
    onResponse(false, false, reason.trim() || undefined);
  } else if (key.escape) {
    setMode("choose");
    setReason("");
  } else if (key.backspace || key.delete) {
    setReason((current) => current.slice(0, -1));
  } else if (input && !key.ctrl && !key.meta) {
    setReason((current) => current + input);
  }
}
