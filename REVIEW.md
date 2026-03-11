# Review Guide

Review priorities for `devagent`:

1. Correctness
2. Regression risk
3. Contract drift
4. Test coverage
5. Docs parity

Blocking findings include:

- regressions in `devagent execute`
- incorrect artifact or event output
- verify-command behavior changes without test coverage
- provider/model handling that breaks the validated DevAgent path
- docs claiming support beyond what the code and validation prove

PR expectations:

- keep changes narrow and test-backed
- include evidence for any “validated” or “production-grade” claim
- avoid large mixed refactors that combine executor, provider, and docs churn without clear boundaries
