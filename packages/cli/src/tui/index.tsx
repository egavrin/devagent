/**
 * TUI entry points — Ink apps for interactive and single-shot modes.
 * No alt-screen — content scrolls naturally in terminal scrollback.
 */

import { render } from "ink";
import React from "react";

import { App, type AppProps } from "./App.js";
import { SingleShotApp, type SingleShotAppProps } from "./SingleShotApp.js";

export async function startTui(props: AppProps): Promise<void> {
  const { waitUntilExit } = render(<App {...props} />, {
    exitOnCtrlC: false,
  });
  await waitUntilExit();
}

export async function startSingleShotTui(props: SingleShotAppProps): Promise<void> {
  const { waitUntilExit } = render(<SingleShotApp {...props} />, {
    exitOnCtrlC: false,
  });
  await waitUntilExit();
}
