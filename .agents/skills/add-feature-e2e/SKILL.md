---
name: add-feature-e2e
description: End-to-end guide for adding a feature to DevAgent, including cross-package surface changes, docs and validation updates, and final verification.
---

# Add Feature End-to-End

Step-by-step guide for adding a feature to DevAgent, from understanding the codebase to verifying the final result.

## Phase 1: Understand

1. Read `AGENTS.md` — understand fail-fast philosophy, anti-patterns, project structure
2. Identify which packages your feature touches:
   - **Runtime types/config/tool/review/task-loop work** → `packages/runtime/`
   - **LLM provider** → `packages/providers/`
   - **CLI/prompt** → `packages/cli/`
   - **Machine execution contract** → `packages/executor/`
   - **ArkTS integration** → `packages/arkts/`
3. Decide whether this is also a public surface change:
   - **CLI/runtime/docs/help/prompt/session UX change** → pair with `surface-change-e2e`
   - **Provider or model-registry change** → pair with `provider-adapter-change`
   - **Live validation coverage change** → pair with `live-validation-authoring`
   - **Executor contract change** → pair with `execute-contract`
4. Read existing similar features to understand the pattern

## Phase 2: Design

Draft the approach before writing code:

- Which types need to be added/changed?
- Which modules need modification?
- What's the wiring path (how does the feature connect to the rest)?
- What tests will verify it works?
- Which docs, help text, prompt assets, or live-validation scenarios need updating?

## Phase 3: Types First

Add or extend types in `packages/runtime/src/core/types.ts`:

```typescript
export interface MyFeatureConfig {
  readonly enabled: boolean;
  readonly setting: string;
}
```

Export from `packages/runtime/src/core/index.ts`:

```typescript
export type { MyFeatureConfig } from "./types.js";
```

Run `bun run typecheck` to verify types compile.

## Phase 4: Tests First

Write failing tests before implementation. Test file location:

```
packages/<pkg>/src/<module>.test.ts
```

Standard test structure:

```typescript
import { describe, it, expect, beforeEach, afterEach } from "vitest";

describe("myFeature", () => {
  it("does the expected thing", () => {
    // Arrange
    // Act
    // Assert
  });
});
```

Run to verify failure:

```bash
cd packages/<pkg> && bunx vitest run src/<module>.test.ts
```

## Phase 5: Implement

Write minimal code to make tests pass. Follow these patterns per package:

### Adding a Tool (`packages/runtime/`)

1. Create `packages/runtime/src/tools/builtins/my-tool.ts` implementing `ToolSpec`
2. Add to `builtinTools` array in `packages/runtime/src/tools/builtins/index.ts`

### Adding a Provider (`packages/providers/`)

1. Create factory function: `(config: ProviderConfig) => LLMProvider`
2. Register in `createDefaultRegistry()` in `packages/providers/src/index.ts`

### Adding a Judge or Runtime Helper (`packages/runtime/`)

1. Create `packages/runtime/src/engine/my-judge.ts` using `llm-judge.ts` utilities
2. Wire into TaskLoop or appropriate caller

### Adding Config (`packages/runtime/`)

1. Add type to `types.ts`
2. Add parsing in `config.ts` with defaults
3. Export from `index.ts`

## Phase 6: Wire Up

1. Export new public APIs from the package's `index.ts`
2. Register runtime tools/providers in the appropriate registry or bootstrap
3. If CLI-visible, update `packages/cli/src/main.ts`
4. If prompt-relevant, update `packages/cli/src/prompts/`

## Phase 7: Public Surface Pass

If the feature changes human-facing behavior, do this before calling the work done:

1. Reconcile README, `WORKFLOW.md`, command help, prompt assets, and nearby documentation tests.
2. Decide whether the change needs a live-validation scenario update or a `release-matrix.md` update.
3. Keep the public story aligned with the supported DevAgent surface; do not leave docs or help text behind the code.

## Phase 8: Integration Test

If the feature crosses package boundaries, add an integration test:

```typescript
// packages/<pkg>/src/integration.test.ts
describe("myFeature integration", () => {
  it("works end-to-end across packages", async () => {
    // Set up the full pipeline
    // Exercise the feature
    // Verify the result
  });
});
```

## Phase 9: Verify

Run the smallest valid verification set first, then widen if the feature changes public behavior:

```bash
bun run typecheck   # Zero errors
bun run test        # All tests pass
bun run build       # Clean build
```

For public surface changes, add:

```bash
bun run check:oss
```

For packaging, install, or release-critical changes, escalate to:

```bash
bun run test:bundle-smoke
bun run test:live-validation
```

Review your changes:

```bash
git diff --stat     # Scope check
git diff            # Content review
```

## Module Dependency Rules

```
runtime ← cli, executor, providers, arkts
providers ← cli
runtime provides the internal task loop, review pipeline, and tool layer
```

Keep `runtime` as the internal center of gravity. Do not recreate the old
`core` / `engine` / `tools` package split in new guidance or code.
