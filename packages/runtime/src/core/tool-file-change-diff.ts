const MAX_LCS_CELLS = 200_000;

export interface DiffOperation {
  readonly type: "context" | "add" | "delete";
  readonly text: string;
}

export function diffSnapshotLines(
  before: ReadonlyArray<string>,
  after: ReadonlyArray<string>,
): ReadonlyArray<DiffOperation> | null {
  const prefix = countCommonPrefix(before, after);
  const suffix = countCommonSuffix(before, after, prefix);
  const beforeMiddle = before.slice(prefix, before.length - suffix);
  const afterMiddle = after.slice(prefix, after.length - suffix);
  const middle = buildMiddleDiffOperations(beforeMiddle, afterMiddle);
  if (!middle) return null;

  return [
    ...before.slice(0, prefix).map((text) => ({ type: "context", text }) satisfies DiffOperation),
    ...middle,
    ...before.slice(before.length - suffix).map((text) => ({ type: "context", text }) satisfies DiffOperation),
  ];
}

export function countCommonPrefix(
  beforeLines: ReadonlyArray<string>,
  afterLines: ReadonlyArray<string>,
): number {
  let prefix = 0;
  while (
    prefix < beforeLines.length &&
    prefix < afterLines.length &&
    beforeLines[prefix] === afterLines[prefix]
  ) {
    prefix++;
  }
  return prefix;
}

export function countCommonSuffix(
  beforeLines: ReadonlyArray<string>,
  afterLines: ReadonlyArray<string>,
  prefix: number,
): number {
  let suffix = 0;
  while (
    suffix < beforeLines.length - prefix &&
    suffix < afterLines.length - prefix &&
    beforeLines[beforeLines.length - 1 - suffix] === afterLines[afterLines.length - 1 - suffix]
  ) {
    suffix++;
  }
  return suffix;
}

function buildMiddleDiffOperations(
  before: ReadonlyArray<string>,
  after: ReadonlyArray<string>,
): ReadonlyArray<DiffOperation> | null {
  if (before.length === 0) {
    return after.map((text) => ({ type: "add", text }) satisfies DiffOperation);
  }
  if (after.length === 0) {
    return before.map((text) => ({ type: "delete", text }) satisfies DiffOperation);
  }
  if ((before.length + 1) * (after.length + 1) > MAX_LCS_CELLS) {
    return null;
  }
  return buildDiffOperationsFromLcs(before, after, buildLcsMatrix(before, after));
}

function buildLcsMatrix(
  before: ReadonlyArray<string>,
  after: ReadonlyArray<string>,
): Uint32Array[] {
  const dp = Array.from({ length: before.length + 1 }, () => new Uint32Array(after.length + 1));
  for (let oldIndex = before.length - 1; oldIndex >= 0; oldIndex--) {
    fillLcsRow(dp, before, after, oldIndex);
  }
  return dp;
}

function fillLcsRow(
  dp: ReadonlyArray<Uint32Array>,
  before: ReadonlyArray<string>,
  after: ReadonlyArray<string>,
  oldIndex: number,
): void {
  for (let newIndex = after.length - 1; newIndex >= 0; newIndex--) {
    dp[oldIndex]![newIndex] = before[oldIndex] === after[newIndex]
      ? dp[oldIndex + 1]![newIndex + 1]! + 1
      : Math.max(dp[oldIndex + 1]![newIndex]!, dp[oldIndex]![newIndex + 1]!);
  }
}

function buildDiffOperationsFromLcs(
  before: ReadonlyArray<string>,
  after: ReadonlyArray<string>,
  dp: ReadonlyArray<Uint32Array>,
): ReadonlyArray<DiffOperation> {
  const operations: DiffOperation[] = [];
  let oldIndex = 0;
  let newIndex = 0;

  while (oldIndex < before.length && newIndex < after.length) {
    if (before[oldIndex] === after[newIndex]) {
      operations.push({ type: "context", text: before[oldIndex]! });
      oldIndex++;
      newIndex++;
      continue;
    }

    if (dp[oldIndex + 1]![newIndex]! >= dp[oldIndex]![newIndex + 1]!) {
      operations.push({ type: "delete", text: before[oldIndex]! });
      oldIndex++;
      continue;
    }

    operations.push({ type: "add", text: after[newIndex]! });
    newIndex++;
  }

  while (oldIndex < before.length) {
    operations.push({ type: "delete", text: before[oldIndex]! });
    oldIndex++;
  }
  while (newIndex < after.length) {
    operations.push({ type: "add", text: after[newIndex]! });
    newIndex++;
  }

  return operations;
}
