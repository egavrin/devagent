/**
 * Welcome — shown on TUI startup before first query.
 */

import { Box, Text } from "ink";
import { existsSync, writeFileSync, mkdirSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import React from "react";

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

interface WelcomeProps {
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
          <Text dimColor>  • Shift+Enter or Option+Enter for multi-line input</Text>
          <Text dimColor>  • Tab completes slash commands and file paths</Text>
          <Text dimColor>  • Shift+Tab toggles default and autopilot</Text>
          <Text dimColor>  • Type /continue after an iteration-limit pause</Text>
          <Text dimColor>  • Use terminal scrollback (for example PgUp/PgDn) to review prior output</Text>
          <Text dimColor>  • Type /help for all commands</Text>
        </Box>
      ) : (
        <Text dimColor>  Type /help for commands • Ctrl+K palette • Shift+Tab safety</Text>
      )}
    </Box>
  );
}
