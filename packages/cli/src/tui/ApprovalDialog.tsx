/**
 * ApprovalDialog — Ink-based approval prompt for tool execution.
 * Shows bordered box with tool info, y/n/a keybindings, and optional rejection reason.
 */

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";

export interface ApprovalRequest {
  readonly id: string;
  readonly toolName: string;
  readonly details: string;
}

export interface ApprovalDialogProps {
  readonly request: ApprovalRequest;
  readonly onResponse: (approved: boolean, always?: boolean, reason?: string) => void;
}

export function ApprovalDialog({ request, onResponse }: ApprovalDialogProps): React.ReactElement {
  const [mode, setMode] = useState<"choose" | "reject-reason">("choose");
  const [reason, setReason] = useState("");

  useInput((input, key) => {
    if (mode === "choose") {
      const k = input.toLowerCase();
      if (k === "y") {
        onResponse(true);
      } else if (k === "a") {
        onResponse(true, true);
      } else if (k === "n") {
        setMode("reject-reason");
      } else if (key.escape) {
        onResponse(false);
      }
    } else if (mode === "reject-reason") {
      if (key.return) {
        onResponse(false, false, reason.trim() || undefined);
      } else if (key.escape) {
        setMode("choose");
        setReason("");
      } else if (key.backspace || key.delete) {
        setReason((r) => r.slice(0, -1));
      } else if (input && !key.ctrl && !key.meta) {
        setReason((r) => r + input);
      }
    }
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
          <Text color="cyan">[a]</Text><Text>lways  </Text>
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
