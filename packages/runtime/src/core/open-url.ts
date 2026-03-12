/**
 * Cross-platform URL opener. Zero dependencies.
 * Falls back gracefully if the browser can't be opened (SSH, headless).
 */

import { exec } from "node:child_process";

/**
 * Open a URL in the user's default browser.
 * Does not throw — failures are silently ignored (the CLI shows the URL as fallback).
 */
export function openUrl(url: string): void {
  const cmd =
    process.platform === "darwin"
      ? "open"
      : process.platform === "win32"
        ? "start"
        : "xdg-open";

  try {
    exec(`${cmd} "${url}"`, (err) => {
      // Ignore errors — caller should always print URL as fallback
      void err;
    });
  } catch {
    // Ignore — caller should always print URL as fallback
  }
}
