import {
  createRoutingLSPTools,
  extractErrorMessage,
} from "@devagent/runtime";

import {
  LSPRouter,
  createCompilerFallbackProvider,
  createRoutingDiagnosticProvider,
  lazyUpgradeLSP,
} from "./double-check-wiring.js";
import { dim, formatError } from "./format.js";
import { spinner } from "./main-state.js";
import type { CliArgs, LSPSetupResult } from "./main-types.js";
import type { DevAgentConfig, DoubleCheck, ToolRegistry } from "@devagent/runtime";

interface LSPSetupOptions {
  readonly config: DevAgentConfig;
  readonly cliArgs: CliArgs;
  readonly projectRoot: string;
  readonly toolRegistry: ToolRegistry;
  readonly doubleCheck: DoubleCheck;
  readonly trackInternalLSPDiagnostics: () => void;
}

async function startConfiguredLSPServers(options: LSPSetupOptions): Promise<LSPSetupResult> {
  if (!options.config.lsp?.servers || options.config.lsp.servers.length === 0) {
    return { lspRouter: null, hasLSPDiagnostics: false };
  }
  const lspRouter = new LSPRouter(options.projectRoot);
  await Promise.allSettled(options.config.lsp.servers.map(async (serverConfig) => {
    try {
      await lspRouter.addServer(serverConfig);
      if (options.cliArgs.verbosity !== "quiet") {
        process.stderr.write(dim(`[lsp] Started: ${serverConfig.command} (${serverConfig.languages.join(", ")})`) + "\n");
      }
    } catch (err) {
      process.stderr.write(formatError(`LSP start failed for ${serverConfig.command}: ${extractErrorMessage(err)}. Skipping.`) + "\n");
    }
  }));
  const hasLSPDiagnostics = registerRoutingLSPTools(lspRouter, options.toolRegistry);
  options.doubleCheck.setDiagnosticProvider(createRoutingDiagnosticProvider(lspRouter, options.trackInternalLSPDiagnostics));
  return { lspRouter, hasLSPDiagnostics };
}

function registerRoutingLSPTools(lspRouter: LSPRouter, toolRegistry: ToolRegistry): boolean {
  const clients = lspRouter.getClients();
  if (clients.length === 0) return false;
  const resolver = (filePath: string) => lspRouter.getClientForFile(filePath);
  for (const tool of createRoutingLSPTools(resolver)) toolRegistry.register(tool);
  return true;
}

function shouldLogSingleShotLsp(cliArgs: CliArgs): boolean {
  return cliArgs.verbosity !== "quiet" && Boolean(cliArgs.query) && !process.stderr.isTTY;
}

function scheduleLazyLSPUpgrade(options: LSPSetupOptions, lazyRouter: LSPRouter): void {
  lazyUpgradeLSP({
    repoRoot: options.projectRoot,
    doubleCheck: options.doubleCheck,
    lspRouter: lazyRouter,
    onLSPDiagnostics: options.trackInternalLSPDiagnostics,
    onServerStarted: (server) => {
      if (shouldLogSingleShotLsp(options.cliArgs)) spinner.log(dim(`[lsp] Auto-detected: ${server.command} (${server.languages.join(", ")})`));
    },
    onUpgradeComplete: (count) => {
      handleLazyLSPUpgradeComplete(count, lazyRouter, options);
    },
    onError: (err) => {
      if (options.cliArgs.verbosity !== "quiet") spinner.log(dim(`[lsp] Lazy detection failed: ${err.message}`));
    },
  }).catch(() => {
    // Compiler fallback remains active.
  });
}

function handleLazyLSPUpgradeComplete(count: number, lazyRouter: LSPRouter, options: LSPSetupOptions): void {
  if (count > 0) {
    registerRoutingLSPTools(lazyRouter, options.toolRegistry);
    if (shouldLogSingleShotLsp(options.cliArgs)) spinner.log(dim(`[lsp] Upgraded to LSP diagnostics (${count} server(s))`));
    return;
  }
  if (shouldLogSingleShotLsp(options.cliArgs)) {
    spinner.log(dim("[double-check] Using compiler fallback diagnostics (no LSP servers in PATH)"));
  }
}

export async function setupLSP(options: LSPSetupOptions): Promise<LSPSetupResult> {
  const configured = await startConfiguredLSPServers(options);
  if (configured.hasLSPDiagnostics || !options.doubleCheck.isEnabled()) return configured;
  options.doubleCheck.setDiagnosticProvider(createCompilerFallbackProvider(options.projectRoot));
  const lazyRouter = new LSPRouter(options.projectRoot);
  scheduleLazyLSPUpgrade(options, lazyRouter);
  return { lspRouter: lazyRouter, hasLSPDiagnostics: false };
}
