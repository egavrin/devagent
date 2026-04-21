import type { ProviderModelCompatibilityIssue } from "../provider-model-compat.js";
import type { CredentialInfo, DevAgentConfig } from "@devagent/runtime";

export type DoctorCheckStatus = "pass" | "blocking" | "advisory";

export interface DoctorCheck {
  readonly label: string;
  readonly status: DoctorCheckStatus;
  readonly detail?: string;
}

export interface DoctorIssue {
  readonly title: string;
  readonly detail: string;
  readonly nextSteps: ReadonlyArray<string>;
}

export interface DoctorProviderStatus {
  readonly id: string;
  readonly hint: string;
  readonly active: boolean;
  readonly hasCredential: boolean;
}

export interface DoctorLspStatus {
  readonly label: string;
  readonly found: boolean;
  readonly install: string;
}

export interface DoctorProviderCredentialIssue {
  readonly status: "blocking" | "advisory";
  readonly detail: string;
}

export interface DoctorReportInput {
  readonly version: string;
  readonly runtimeLabel: string;
  readonly runtimeError?: string;
  readonly gitError?: string;
  readonly configPath?: string;
  readonly configSearchPaths: ReadonlyArray<string>;
  readonly config: DevAgentConfig;
  readonly providerStatuses: ReadonlyArray<DoctorProviderStatus>;
  readonly providerCredentialIssue?: DoctorProviderCredentialIssue;
  readonly modelRegistryError?: string;
  readonly modelRegistryCount?: number;
  readonly modelRegistered: boolean;
  readonly modelProviders?: ReadonlyArray<string>;
  readonly providerModelIssue?: ProviderModelCompatibilityIssue;
  readonly lspStatuses: ReadonlyArray<DoctorLspStatus>;
  readonly platformLabel: string;
  readonly providerSource: "cli" | "env" | "config" | "default";
  readonly modelSource: "cli" | "env" | "config" | "default";
  readonly credentialSource: string;
}

export interface DoctorReport {
  readonly version: string;
  readonly blockingIssues: ReadonlyArray<DoctorIssue>;
  readonly runtimeCheck: DoctorCheck;
  readonly gitCheck: DoctorCheck;
  readonly configCheck: DoctorCheck;
  readonly providerCheck: DoctorCheck;
  readonly providerStatuses: ReadonlyArray<DoctorProviderStatus>;
  readonly modelRegistryCheck: DoctorCheck;
  readonly modelCheck: DoctorCheck;
  readonly providerModelCheck: DoctorCheck;
  readonly effectiveConfig: {
    readonly provider: string;
    readonly providerSource: "cli" | "env" | "config" | "default";
    readonly model: string;
    readonly modelSource: "cli" | "env" | "config" | "default";
    readonly credentialSource: string;
    readonly modelProviders?: ReadonlyArray<string>;
  };
  readonly lspStatuses: ReadonlyArray<DoctorLspStatus>;
  readonly platformCheck: DoctorCheck;
  readonly ok: boolean;
  readonly summaryStatus: "pass" | "advisory" | "blocking";
}

export type DoctorStoredCredentials = Readonly<Record<string, CredentialInfo>>;
