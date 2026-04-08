# packages/providers

This file provides guidance to AI agents working in this component.

## Purpose and Scope

- **component**: `packages/providers`
- **purpose**: Provider registry and provider-specific adapters for Anthropic, OpenAI-compatible APIs, ChatGPT, Copilot, DeepSeek, OpenRouter, and Ollama.
- **primary languages/platforms**: TypeScript, Bun, Node.js, Vercel AI SDK integrations

## Important Paths and Interfaces

```text
src/index.ts                 # Built-in provider registration and provider wrappers
src/openai.ts                # OpenAI-compatible provider implementation
src/anthropic.ts             # Anthropic adapter
src/registry.ts              # Provider registry
src/shared.ts                # Shared provider helpers
src/*.test.ts                # Provider behavior and regression tests
package.json                 # Build/test scripts
```

## Build, Test, and Run

```bash
cd packages/providers
bun run build
bun run typecheck
bun run test
```

## Architecture and Workflows

- `src/index.ts` registers the built-in providers and encodes provider-specific request shaping such as ChatGPT codex options, Copilot headers, and OpenAI-compatible base URLs.
- Changes here can break production behavior without any CLI surface change. Keep endpoint, auth-header, and capability changes tightly scoped and test-backed.
- Prefer adjusting the dedicated provider test alongside each behavior change rather than folding multiple providers into one broad refactor.
- `providers` may depend on `runtime`, but it should remain independent from `cli`, `executor`, and `arkts`.

## Generated Files, Conventions, and Pitfalls

- Provider differences are intentional. Avoid “normalizing” away custom headers, stripped fields, or capability flags unless the external API contract actually changed.
- Secret handling matters here: never hardcode real credentials, and keep logs and errors free of token leakage.
- Do not edit `dist/`; it is build output.

## Local Skills

- No component-local skill directory exists here. Use the shared repo skills from `.agents/skills`, especially `provider-adapter-change` when the task mentions provider proxy, auth mismatch, model registry drift, or streaming behavior, plus `security-checklist`, `testing`, and `review`.
