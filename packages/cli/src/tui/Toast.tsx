/**
 * Toast — auto-dismiss notification overlay.
 */

import { Box, Text } from "ink";
import React, { useEffect } from "react";

export type ToastVariant = "info" | "success" | "warning" | "error";

export interface ToastMessage {
  readonly id: string;
  readonly message: string;
  readonly variant: ToastVariant;
  readonly durationMs?: number;
}

interface ToastProps {
  readonly toasts: ReadonlyArray<ToastMessage>;
  readonly onDismiss: (id: string) => void;
}

const variantColors: Record<ToastVariant, string> = {
  info: "cyan",
  success: "green",
  warning: "yellow",
  error: "red",
};

const variantIcons: Record<ToastVariant, string> = {
  info: "ℹ",
  success: "✓",
  warning: "△",
  error: "✗",
};

function ToastItem({ toast, onDismiss }: { toast: ToastMessage; onDismiss: (id: string) => void }): React.ReactElement {
  useEffect(() => {
    const duration = toast.durationMs ?? 5000;
    const timer = setTimeout(() => onDismiss(toast.id), duration);
    return () => clearTimeout(timer);
  }, [toast.id, toast.durationMs, onDismiss]);

  const color = variantColors[toast.variant];
  const icon = variantIcons[toast.variant];

  return (
    <Box>
      <Text color={color}>{icon} </Text>
      <Text color={color}>{toast.message.slice(0, 60)}</Text>
    </Box>
  );
}

export function ToastContainer({ toasts, onDismiss }: ToastProps): React.ReactElement | null {
  if (toasts.length === 0) return null;

  return (
    <Box flexDirection="column" flexShrink={0}>
      {toasts.slice(-3).map((toast) => (
        <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
      ))}
    </Box>
  );
}
