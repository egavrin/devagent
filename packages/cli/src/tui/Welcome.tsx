/**
 * Welcome — shown on TUI startup before first query.
 */

import React from "react";
import { existsSync, writeFileSync, mkdirSync } from "node:fs";
import { join } from "node:path";
import { homedir } from "node:os";
import { Box, Text } from "ink";

const FIRST_RUN_PATH = join(homedir(), ".devagent", ".first-run-done");

function isFirstRun(): boolean {
  return !existsSync(FIRST_RUN_PATH);
}

function markFirstRunDone(): void {
  try {
    mkdirSync(join(homedir(), ".devagent"), { recursive: true });
    writeFileSync(FIRST_RUN_PATH, new Date().toISOString());
  } catch { /* non-fatal */ }
}

export interface WelcomeProps {
  readonly model: string;
  readonly version?: string;
}

export function Welcome({ model, version }: WelcomeProps): React.ReactElement {
  const [firstRun] = React.useState(() => {
    const result = isFirstRun();
    if (result) markFirstRunDone();
    return result;
  });

  return (
    <Box flexDirection="column" marginBottom={1}>
      <Text bold color="cyan">
        {'  ╔══════════════════╗\n'}
        {'  ║    devagent      ║\n'}
        {'  ╚══════════════════╝'}
      </Text>
      <Text dimColor>  {model}{version ? ` • v${version}` : ""}</Text>
      {firstRun ? (
        <Box flexDirection="column" marginTop={1}>
          <Text dimColor>  💡 Tips:</Text>
          <Text dimColor>  • Ctrl+K opens the command palette</Text>
          <Text dimColor>  • Shift+Enter for multi-line input</Text>
          <Text dimColor>  • Tab completes slash commands and file paths</Text>
          <Text dimColor>  • Shift+Tab cycles suggest, auto-edit, and full-auto</Text>
          <Text dimColor>  • Type /continue after an iteration-limit pause</Text>
          <Text dimColor>  • PgUp/PgDn scrolls through history</Text>
          <Text dimColor>  • Type /help for all commands</Text>
        </Box>
      ) : (
        <Text dimColor>  Type /help for commands • Ctrl+K palette • Shift+Tab mode</Text>
      )}
    </Box>
  );
}
