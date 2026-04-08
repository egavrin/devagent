# Release Checklist

Use this lighter checklist before escalating to the heavyweight release-validation skill.

## Commands

```bash
bun run build
bun run typecheck
bun run test
bun run check:oss
bun run test:bundle-smoke
```

## Compare

- Root `README.md` install and upgrade guidance
- `package.json` version, engines, and scripts
- bundle and publish scripts under `scripts/`
- any release-facing CLI help or doctor guidance changed by the work

## Escalate to `validate-user-surface`

- install or bootstrap behavior changed materially
- auth, provider, TUI, review, or `devagent execute` flows changed
- the publish bundle passes smoke tests but the public contract still needs live proof
