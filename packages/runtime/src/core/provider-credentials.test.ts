import { describe, expect, it } from "vitest";

import {
  formatResolvedCredentialSource,
  resolveProviderCredentialStatus,
} from "./provider-credentials.js";

describe("resolveProviderCredentialStatus", () => {
  it("prefers provider-specific env vars for API providers", () => {
    const result = resolveProviderCredentialStatus({
      providerId: "openai",
      env: { OPENAI_API_KEY: "sk-openai" },
    });

    expect(result.hasCredential).toBe(true);
    expect(result.source).toBe("env");
    expect(result.envVar).toBe("OPENAI_API_KEY");
    expect(result.apiKey).toBe("sk-openai");
  });

  it("does not use DEVAGENT_API_KEY for non-gateway providers", () => {
    const result = resolveProviderCredentialStatus({
      providerId: "openai",
      env: { DEVAGENT_API_KEY: "ilg-only" },
    });

    expect(result.hasCredential).toBe(false);
    expect(result.source).toBe("missing");
  });

  it("uses DEVAGENT_API_KEY for the devagent-api provider", () => {
    const result = resolveProviderCredentialStatus({
      providerId: "devagent-api",
      env: { DEVAGENT_API_KEY: "ilg-only" },
    });

    expect(result.hasCredential).toBe(true);
    expect(result.source).toBe("env");
    expect(result.envVar).toBe("DEVAGENT_API_KEY");
    expect(result.apiKey).toBe("ilg-only");
  });

  it("prefers config values ahead of env and stored credentials", () => {
    const result = resolveProviderCredentialStatus({
      providerId: "openai",
      providerConfig: { model: "gpt-4.1", apiKey: "config-key" },
      topLevelApiKey: "top-level-key",
      storedCredential: { type: "api", key: "stored-key", storedAt: 1 },
      env: { OPENAI_API_KEY: "env-key" },
    });

    expect(result.hasCredential).toBe(true);
    expect(result.source).toBe("provider-config");
    expect(result.apiKey).toBe("config-key");
  });

  it("falls back to stored credentials when an explicit provider env ref is unset", () => {
    const result = resolveProviderCredentialStatus({
      providerId: "openai",
      providerConfigApiKey: "env:DEVAGENT_TEST_OPENAI_KEY",
      storedCredential: { type: "api", key: "stored-key", storedAt: 1 },
      env: { OPENAI_API_KEY: "default-env-key" },
    });

    expect(result.hasCredential).toBe(true);
    expect(result.source).toBe("stored");
    expect(result.apiKey).toBe("stored-key");
  });

  it("reports the referenced env var when an explicit top-level env ref is unset with no stored credential", () => {
    const result = resolveProviderCredentialStatus({
      providerId: "openai",
      topLevelApiKey: "env:DEVAGENT_TEST_TOP_LEVEL_KEY",
      env: { OPENAI_API_KEY: "default-env-key" },
    });

    expect(result.hasCredential).toBe(false);
    expect(result.source).toBe("missing");
    expect(result.envVar).toBe("DEVAGENT_TEST_TOP_LEVEL_KEY");
  });

  it("resolves oauth credentials from stored login when present", () => {
    const result = resolveProviderCredentialStatus({
      providerId: "chatgpt",
      storedCredential: {
        type: "oauth",
        accessToken: "oauth-token",
        storedAt: 1,
      },
    });

    expect(result.hasCredential).toBe(true);
    expect(result.source).toBe("stored");
  });
});

describe("formatResolvedCredentialSource", () => {
  it("formats env labels with the env var name", () => {
    expect(formatResolvedCredentialSource({
      providerId: "openai",
      credentialMode: "api",
      hasCredential: true,
      source: "env",
      envVar: "OPENAI_API_KEY",
      apiKey: "sk-openai",
    })).toBe("env (OPENAI_API_KEY)");
  });
});
