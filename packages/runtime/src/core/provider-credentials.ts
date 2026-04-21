import type { CredentialInfo } from "./credentials.js";
import type { ProviderConfig } from "./types.js";

export type ProviderCredentialMode = "api" | "oauth" | "none";

export interface ProviderCredentialDescriptor {
  readonly id: string;
  readonly envVar: string | null;
  readonly hint: string;
  readonly credentialMode: ProviderCredentialMode;
}

export interface ResolvedProviderCredentialStatus {
  readonly providerId: string;
  readonly credentialMode: ProviderCredentialMode;
  readonly hasCredential: boolean;
  readonly source: "provider-config" | "top-level-config" | "env" | "stored" | "missing" | "not-required";
  readonly envVar?: string;
  readonly apiKey?: string;
}

interface ResolvedConfiguredApiKey {
  readonly source: "literal" | "env" | "missing-env" | "absent";
  readonly apiKey?: string;
  readonly envVar?: string;
}

interface CredentialResolutionContext {
  readonly providerId: string;
  readonly credentialMode: ProviderCredentialMode;
  readonly providerConfig?: ProviderConfig;
  readonly providerConfigApiKey?: string | null;
  readonly topLevelApiKey?: string;
  readonly storedCredential: CredentialInfo | null;
  readonly env: NodeJS.ProcessEnv;
  readonly descriptor?: ProviderCredentialDescriptor;
}

const PROVIDER_CREDENTIALS: readonly ProviderCredentialDescriptor[] = [
  { id: "anthropic", envVar: "ANTHROPIC_API_KEY", hint: "set ANTHROPIC_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "openai", envVar: "OPENAI_API_KEY", hint: "set OPENAI_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "devagent-api", envVar: "DEVAGENT_API_KEY", hint: "set DEVAGENT_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "deepseek", envVar: "DEEPSEEK_API_KEY", hint: "set DEEPSEEK_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "openrouter", envVar: "OPENROUTER_API_KEY", hint: "set OPENROUTER_API_KEY or devagent auth login", credentialMode: "api" },
  { id: "chatgpt", envVar: null, hint: "devagent auth login (ChatGPT Plus/Pro)", credentialMode: "oauth" },
  { id: "github-copilot", envVar: null, hint: "devagent auth login (GitHub device flow)", credentialMode: "oauth" },
  { id: "ollama", envVar: null, hint: "local — no API key needed (ollama must be running)", credentialMode: "none" },
] as const;

export function listProviderCredentialDescriptors(): ReadonlyArray<ProviderCredentialDescriptor> {
  return PROVIDER_CREDENTIALS;
}

export function getProviderCredentialDescriptor(providerId: string): ProviderCredentialDescriptor | undefined {
  return PROVIDER_CREDENTIALS.find((provider) => provider.id === providerId);
}

export function getProviderCredentialEnvVar(providerId: string): string | null {
  return getProviderCredentialDescriptor(providerId)?.envVar ?? null;
}

function resolveConfiguredApiKey(
  value: string | null | undefined,
  env: NodeJS.ProcessEnv,
): ResolvedConfiguredApiKey {
  if (!value) {
    return { source: "absent" };
  }

  if (!value.startsWith("env:")) {
    return {
      source: "literal",
      apiKey: value,
    };
  }

  const envVar = value.slice(4);
  const envValue = env[envVar];
  if (envValue === undefined) {
    return {
      source: "missing-env",
      envVar,
    };
  }

  return {
    source: "env",
    envVar,
    apiKey: envValue,
  };
}
export function resolveProviderCredentialStatus(opts: {
  readonly providerId: string;
  readonly providerConfig?: ProviderConfig;
  readonly providerConfigApiKey?: string | null;
  readonly topLevelApiKey?: string;
  readonly storedCredential?: CredentialInfo | null;
  readonly env?: NodeJS.ProcessEnv;
}): ResolvedProviderCredentialStatus {
  const descriptor = getProviderCredentialDescriptor(opts.providerId);
  const credentialMode = descriptor?.credentialMode ?? "api";
  const context: CredentialResolutionContext = {
    providerId: opts.providerId,
    credentialMode,
    providerConfig: opts.providerConfig,
    providerConfigApiKey: opts.providerConfigApiKey,
    topLevelApiKey: opts.topLevelApiKey,
    storedCredential: opts.storedCredential ?? null,
    env: opts.env ?? process.env,
    descriptor,
  };

  if (credentialMode === "none") {
    return buildCredentialStatus(context, true, "not-required");
  }

  if (credentialMode === "oauth") {
    return resolveOAuthCredentialStatus(context);
  }

  return resolveApiCredentialStatus(context);
}

function resolveOAuthCredentialStatus(
  context: CredentialResolutionContext,
): ResolvedProviderCredentialStatus {
  if (context.providerConfig?.oauthToken) {
    return buildCredentialStatus(context, true, "provider-config");
  }
  if (context.storedCredential?.type === "oauth") {
    return buildCredentialStatus(context, true, "stored");
  }
  return buildCredentialStatus(context, false, "missing");
}

function resolveApiCredentialStatus(
  context: CredentialResolutionContext,
): ResolvedProviderCredentialStatus {
  return resolveApiKeySource(context, "provider-config", getProviderApiKeyValue(context))
    ?? resolveApiKeySource(context, "top-level-config", context.topLevelApiKey)
    ?? resolveDescriptorEnvCredential(context)
    ?? resolveStoredApiCredential(context)
    ?? buildCredentialStatus(context, false, "missing");
}

function getProviderApiKeyValue(context: CredentialResolutionContext): string | null | undefined {
  return context.providerConfigApiKey !== undefined
    ? context.providerConfigApiKey
    : context.providerConfig?.apiKey;
}

function resolveApiKeySource(
  context: CredentialResolutionContext,
  source: "provider-config" | "top-level-config",
  value: string | null | undefined,
): ResolvedProviderCredentialStatus | null {
  const resolved = resolveConfiguredApiKey(value, context.env);
  if (resolved.source === "literal") {
    return buildCredentialStatus(context, true, source, { apiKey: resolved.apiKey });
  }
  if (resolved.source === "env") {
    return buildCredentialStatus(context, true, "env", {
      apiKey: resolved.apiKey,
      envVar: resolved.envVar,
    });
  }
  if (resolved.source === "missing-env") {
    return resolveMissingEnvCredential(context, resolved.envVar);
  }
  return null;
}

function resolveMissingEnvCredential(
  context: CredentialResolutionContext,
  envVar: string | undefined,
): ResolvedProviderCredentialStatus {
  const stored = resolveStoredApiCredential(context);
  return stored ?? buildCredentialStatus(context, false, "missing", { envVar });
}

function resolveDescriptorEnvCredential(
  context: CredentialResolutionContext,
): ResolvedProviderCredentialStatus | null {
  const envVar = context.descriptor?.envVar;
  const apiKey = envVar ? context.env[envVar] : undefined;
  return envVar && apiKey
    ? buildCredentialStatus(context, true, "env", { envVar, apiKey })
    : null;
}

function resolveStoredApiCredential(
  context: CredentialResolutionContext,
): ResolvedProviderCredentialStatus | null {
  return context.storedCredential?.type === "api"
    ? buildCredentialStatus(context, true, "stored", { apiKey: context.storedCredential.key })
    : null;
}

function buildCredentialStatus(
  context: CredentialResolutionContext,
  hasCredential: boolean,
  source: ResolvedProviderCredentialStatus["source"],
  values?: Pick<ResolvedProviderCredentialStatus, "apiKey" | "envVar">,
): ResolvedProviderCredentialStatus {
  return {
    providerId: context.providerId,
    credentialMode: context.credentialMode,
    hasCredential,
    source,
    ...values,
  };
}

export function formatResolvedCredentialSource(status: ResolvedProviderCredentialStatus): string {
  switch (status.source) {
    case "provider-config":
      return "config";
    case "top-level-config":
      return "config";
    case "env":
      return status.envVar ? `env (${status.envVar})` : "env";
    case "stored":
      return status.credentialMode === "oauth" ? "stored oauth" : "stored api key";
    case "not-required":
      return `missing (not required for ${status.providerId})`;
    case "missing":
      return "missing";
  }
}
