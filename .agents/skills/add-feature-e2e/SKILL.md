---
name: add-feature-e2e
description: End-to-end guide for adding a feature to DevAgent — types, tests, implement, wire, verify.
---

# Add Feature End-to-End

Step-by-step guide for adding a feature to DevAgent, from understanding the codebase to verifying the final result.

## Phase 1: Understand

1. Read `AGENTS.md` — understand fail-fast philosophy, anti-patterns, project structure
2. Identify which packages your feature touches:
   - **Types/config** → `packages/core/`
   - **Tool** → `packages/tools/`
   - **Engine feature** (judge, plugin, agent) → `packages/engine/`
   - **LLM provider** → `packages/providers/`
   - **CLI/prompt** → `packages/cli/`
3. Read existing similar features to understand the pattern

## Phase 2: Design

Draft the approach before writing code:

- Which types need to be added/changed?
- Which modules need modification?
- What's the wiring path (how does the feature connect to the rest)?
- What tests will verify it works?

## Phase 3: Types First

Add or extend types in `packages/core/src/types.ts`:

```typescript
export interface MyFeatureConfig {
  readonly enabled: boolean;
  readonly setting: string;
}
```

Export from `packages/core/src/index.ts`:

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

### Adding a Tool (`packages/tools/`)

1. Create `packages/tools/src/builtins/my-tool.ts` implementing `ToolSpec`
2. Add to `builtinTools` array in `packages/tools/src/builtins/index.ts`

### Adding a Provider (`packages/providers/`)

1. Create factory function: `(config: ProviderConfig) => LLMProvider`
2. Register in `createDefaultRegistry()` in `packages/providers/src/index.ts`

### Adding a Plugin (`packages/engine/`)

1. Create `packages/engine/src/plugins/my-plugin.ts` implementing `Plugin`
2. Add factory to `createBuiltinPlugins()` in `packages/engine/src/plugins/index.ts`

### Adding a Judge (`packages/engine/`)

1. Create `packages/engine/src/my-judge.ts` using `llm-judge.ts` utilities
2. Wire into TaskLoop or appropriate caller

### Adding Config (`packages/core/`)

1. Add type to `types.ts`
2. Add parsing in `config.ts` with defaults
3. Export from `index.ts`

## Phase 6: Wire Up

1. Export new public APIs from the package's `index.ts`
2. Register tools/plugins/providers in the appropriate registry
3. If CLI-visible, update `packages/cli/src/main.ts`
4. If prompt-relevant, update `packages/cli/src/prompts/`

## Phase 7: Integration Test

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

## Phase 8: Verify

Run the full verification checklist:

```bash
bun run typecheck   # Zero errors
bun run test        # All tests pass
bun run build       # Clean build
```

Review your changes:

```bash
git diff --stat     # Scope check
git diff            # Content review
```

## Module Dependency Rules

```
core ← tools, providers (core imports nothing from other packages)
core, tools, providers ← engine
core, engine ← cli
```

Never import from a dependent package (e.g., core must never import from engine).
