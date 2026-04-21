# Provider Verification Matrix

| Change type | Likely files | Minimum checks | Add these when needed |
| --- | --- | --- | --- |
| OpenAI-compatible transport, headers, proxy, streaming | `packages/providers/src/openai.ts`, `packages/providers/src/network.ts`, `packages/providers/src/shared.ts`, nearby tests | `cd packages/providers && bun run test`, `bun run typecheck` | CLI tests if provider config or doctor output changed |
| Registry or provider availability | `packages/providers/src/index.ts`, `packages/providers/src/registry.ts`, `models/*.toml`, nearby tests | `cd packages/providers && bun run test`, `bun run typecheck` | `bun run check:oss` if docs or provider list changed |
| Credential or auth expectation changes | Provider files plus CLI config or auth files | `cd packages/providers && bun run test`, `cd packages/cli && bun run test`, `bun run typecheck` | `security-checklist`; provider smoke for real auth flows |
| Doctor or remediation copy changes | Provider files, `packages/cli/src/doctor.test.ts`, `README.md` | `cd packages/cli && bun run test`, `bun run check:oss` | `validate-user-surface` if the fix path is release-critical |

Prefer provider-local tests first, then add cross-package verification only when the provider change is surfaced through the CLI or public docs.
