import { describe, expect, it } from "vitest";

import { collectCredentialStatusEntries } from "./auth.js";

describe("collectCredentialStatusEntries", () => {
  it("reports provider-specific env vars without adding a duplicate generic gateway row", () => {
    const rows = collectCredentialStatusEntries({}, {
      OPENAI_API_KEY: "sk-openai",
      DEVAGENT_API_KEY: "ilg-gateway",
    });

    expect(rows).toEqual([
      { id: "devagent-api", source: "env:DEVAGENT_API_KEY", masked: "ilg-...eway" },
      { id: "openai", source: "env:OPENAI_API_KEY", masked: "sk-o...enai" },
    ]);
  });
});
