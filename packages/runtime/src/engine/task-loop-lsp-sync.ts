import type { LSPDocumentSync, ToolResult } from "../core/index.js";

export async function syncLSPAfterToolResult(
  toolName: string,
  args: Record<string, unknown>,
  result: ToolResult,
  lspSync: LSPDocumentSync | null | undefined,
): Promise<void> {
  if (!lspSync) return;
  const targets = getSyncTargets(toolName, args, result);
  if (targets.length === 0) return;

  await Promise.all(targets.map(({ filePath, didSave }) =>
    lspSync.syncFile(filePath, { didSave }).catch(() => {
      // LSP sync is best-effort. The original tool result must remain intact
      // when a language server is missing, crashed, or temporarily stale.
    }),
  ));
}

function getSyncTargets(
  toolName: string,
  args: Record<string, unknown>,
  result: ToolResult,
): ReadonlyArray<{ readonly filePath: string; readonly didSave: boolean }> {
  if (toolName === "read_file") {
    if (!result.success) return [];
    const filePath = args["path"];
    return typeof filePath === "string" && !filePath.includes("://")
      ? [{ filePath, didSave: false }]
      : [];
  }

  if (toolName === "write_file" || toolName === "replace_in_file") {
    return result.artifacts
      .filter((artifact): artifact is string => typeof artifact === "string" && artifact.length > 0)
      .map((filePath) => ({ filePath, didSave: true }));
  }

  return [];
}
