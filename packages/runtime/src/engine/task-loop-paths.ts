export function normalizeRepoPath(filePath: string): string {
  const normalized = filePath.trim().replaceAll("\\", "/");
  return normalized.startsWith("./") ? normalized.slice(2) : normalized;
}
