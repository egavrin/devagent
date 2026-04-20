/**
 * Spinner — animated thinking indicator with metrics suffix.
 */

import { Text } from "ink";
import React, { useState, useEffect } from "react";

import { SPINNER_FRAMES, SPINNER_VERBS } from "./shared.js";

interface SpinnerProps {
  readonly active: boolean;
  readonly message?: string;
  readonly suffix?: string;
}

export function Spinner({ active, message, suffix }: SpinnerProps): React.ReactElement | null {
  const [frame, setFrame] = useState(0);
  const [startedAt] = useState(() => Date.now());
  const [verb] = useState(() => SPINNER_VERBS[Math.floor(Math.random() * SPINNER_VERBS.length)]!);

  useEffect(() => {
    if (!active) return;
    const timer = setInterval(() => {
      setFrame((f) => (f + 1) % SPINNER_FRAMES.length);
    }, 80);
    return () => clearInterval(timer);
  }, [active]);

  if (!active) return null;

  const elapsed = Date.now() - startedAt;
  const elapsedStr = elapsed >= 1000 ? ` (${(elapsed / 1000).toFixed(1)}s)` : "";
  const display = message ?? `${verb}…`;

  return (
    <Text>
      <Text color="cyan">{SPINNER_FRAMES[frame]}</Text>
      {" "}
      <Text dimColor>{display}{elapsedStr}</Text>
      {suffix && <Text dimColor> {suffix}</Text>}
    </Text>
  );
}
