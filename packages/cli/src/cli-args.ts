import { SafetyMode, extractErrorMessage } from "@devagent/runtime";
import { readFileSync as nodeReadFileSync } from "node:fs";

import { getVersion } from "./cli-version.js";
import { formatError } from "./format.js";
import { getSafetyPreset } from "./main-safety.js";
import type { CliArgs, CliSubcommand, ReviewArgs } from "./main-types.js";

const SUBCOMMANDS = new Set<CliSubcommand["name"]>([
  "help",
  "doctor",
  "config",
  "configure",
  "setup",
  "init",
  "update",
  "completions",
  "install-lsp",
]);

export function writeStdout(message = ""): void {
  process.stdout.write(`${message}\n`);
}

export function renderReviewHelpText(): string {
  return `Usage:
  devagent review <file> --rule <rule_file> [--json]

Run the rule-based review pipeline on a patch or diff file.`;
}

export function loadQueryFromFile(
  path: string,
  readFileSync: (path: string, encoding: "utf-8") => string = nodeReadFileSync,
  inlineQuery: string | null = null,
): string {
  if (inlineQuery) throw new Error("Cannot specify both --file and an inline query");
  let raw: string;
  try {
    raw = readFileSync(path, "utf-8");
  } catch (error) {
    const message = extractErrorMessage(error);
    if (message.includes("ENOENT")) throw new Error(`Input file not found: ${path}`);
    throw error;
  }
  const query = raw.trim();
  if (query.length === 0) throw new Error(`Input file is empty: ${path}`);
  return query;
}

export function parseArgs(argv: string[]): CliArgs {
  const args = argv.slice(2);
  const result = createDefaultCliArgs();
  for (let i = 0; i < args.length; i++) {
    const parsed = parseCliArg(args, i, result);
    i = parsed.index;
    if (parsed.stop) break;
  }
  if (result.continue_ && (result.query || result.file)) {
    result.usageError = "--continue does not accept a query or file input. Use --resume <id> to target a specific session.";
  }
  return result;
}

function createDefaultCliArgs(): CliArgs {
  return {
    query: null,
    file: null,
    safetyMode: null,
    modeParseError: null,
    usageError: null,
    provider: null,
    model: null,
    maxIterations: null,
    reasoning: null,
    verbosity: "normal",
    verboseCategories: undefined,
    authCommand: null,
    sessionsCommand: false,
    resume: null,
    continue_: false,
    review: null,
    subcommand: null,
  };
}

interface ParsedCliArg {
  readonly index: number;
  readonly stop: boolean;
}

function parseCliArg(args: string[], index: number, result: CliArgs): ParsedCliArg {
  const arg = args[index]!;
  handleImmediateCliArg(arg);
  if (SUBCOMMANDS.has(arg as CliSubcommand["name"])) {
    result.subcommand = { name: arg as CliSubcommand["name"], args: args.slice(index + 1) };
    return { index, stop: true };
  }
  if (arg === "sessions") {
    result.sessionsCommand = true;
    return { index, stop: true };
  }
  if (arg === "review") return parseReviewCommand(args, index + 1, result);
  if (arg === "auth") {
    result.authCommand = { subcommand: args[index + 1] ?? "login", args: args.slice(index + 2) };
    return { index, stop: true };
  }
  const nextIndex = parseCliFlag(args, index, result);
  if (nextIndex !== null) return { index: nextIndex, stop: false };
  if (!arg.startsWith("-")) result.query = arg;
  return { index, stop: false };
}

function handleImmediateCliArg(arg: string): void {
  if (arg === "version" || arg === "--version" || arg === "-V") {
    writeStdout(getVersion());
    process.exit(0);
  }
  if (arg === "--help" || arg === "-h") {
    printHelp();
    process.exit(0);
  }
}

function parseReviewCommand(args: string[], index: number, result: CliArgs): ParsedCliArg {
  const reviewArgs: ReviewArgs = { patchFile: "", ruleFile: null, jsonOutput: false, help: false };
  for (let i = index; i < args.length; i++) i = parseReviewArg(args, i, result, reviewArgs);
  result.review = reviewArgs;
  return { index: args.length, stop: true };
}

function parseReviewArg(args: string[], index: number, result: CliArgs, reviewArgs: ReviewArgs): number {
  const arg = args[index]!;
  const handled = parseReviewValueArg(args, index, result, reviewArgs);
  if (handled !== null) return handled;
  if (arg === "--help" || arg === "-h") reviewArgs.help = true;
  if (arg === "--json") reviewArgs.jsonOutput = true;
  if (!arg.startsWith("-")) reviewArgs.patchFile = arg;
  return index;
}

function parseReviewValueArg(args: string[], index: number, result: CliArgs, reviewArgs: ReviewArgs): number | null {
  const arg = args[index]!;
  const value = args[index + 1];
  if (!value) return null;
  if (arg === "--rule") {
    reviewArgs.ruleFile = value;
    return index + 1;
  }
  if (arg === "--provider") {
    result.provider = value;
    return index + 1;
  }
  if (arg === "--model") {
    result.model = value;
    return index + 1;
  }
  return null;
}

function parseCliFlag(args: string[], index: number, result: CliArgs): number | null {
  const arg = args[index]!;
  if (parseModeFlag(args, index, result)) return arg === "--mode" ? index + 1 : index;
  if (arg === "--suggest" || arg === "--auto-edit" || arg === "--full-auto") {
    result.modeParseError = `${arg} has been removed. Use ${arg === "--full-auto" ? "--mode autopilot" : "--mode default"} instead.`;
    return index;
  }
  if (parseSimpleCliFlag(arg, result)) return index;
  if (arg.startsWith("--verbose=")) {
    result.verboseCategories = arg.slice("--verbose=".length);
    return index;
  }
  return parseValueFlag(args, index, result);
}

function parseSimpleCliFlag(arg: string, result: CliArgs): boolean {
  if (arg === "-q" || arg === "--quiet") {
    result.verbosity = "quiet";
    return true;
  }
  if (arg === "-v" || arg === "--verbose") {
    result.verbosity = "verbose";
    return true;
  }
  if (arg === "--continue") {
    result.continue_ = true;
    return true;
  }
  return false;
}

function parseModeFlag(args: string[], index: number, result: CliArgs): boolean {
  const arg = args[index]!;
  const value = arg === "--mode" ? args[index + 1] : arg.startsWith("--mode=") ? arg.slice("--mode=".length) : undefined;
  if (value === undefined) return false;
  if (value === SafetyMode.DEFAULT || value === SafetyMode.AUTOPILOT) {
    result.safetyMode = value;
  } else {
    result.modeParseError = `Invalid --mode value: ${value}. Expected one of: default, autopilot.`;
  }
  return true;
}

function parseValueFlag(args: string[], index: number, result: CliArgs): number | null {
  const arg = args[index]!;
  const value = args[index + 1];
  if (!value) return null;
  const handler = VALUE_FLAG_HANDLERS.find((entry) => entry.flags.includes(arg));
  if (!handler) return null;
  handler.apply(result, value);
  return index + 1;
}

interface ValueFlagHandler {
  readonly flags: readonly string[];
  readonly apply: (result: CliArgs, value: string) => void;
}

const VALUE_FLAG_HANDLERS: readonly ValueFlagHandler[] = [
  { flags: ["--file", "-f"], apply: (result, value) => { result.file = value; } },
  { flags: ["--provider"], apply: (result, value) => { result.provider = value; } },
  { flags: ["--model"], apply: (result, value) => { result.model = value; } },
  { flags: ["--max-iterations"], apply: applyMaxIterations },
  { flags: ["--reasoning"], apply: applyReasoning },
  { flags: ["--resume"], apply: (result, value) => { result.resume = value; } },
];

function applyMaxIterations(result: CliArgs, value: string): void {
  const parsed = parseInt(value, 10);
  result.maxIterations = isNaN(parsed) ? null : parsed;
}

function applyReasoning(result: CliArgs, value: string): void {
  if (value === "low" || value === "medium" || value === "high") result.reasoning = value;
}

export function renderHelpText(): string {
  return `
devagent — AI-powered development agent

Usage:
  devagent                         Interactive TUI
  devagent "<query>"              Natural language query
  devagent -f <path>               Read query from file
  devagent sessions                List recent sessions
  devagent review <file> --rule <rule_file> [--json]
                                  Rule-based patch review
  devagent execute --request <file> --artifact-dir <dir>
                                  Public machine execution contract

Commands:
  devagent help                   Show top-level help
  devagent setup                  Guided global configuration wizard
  devagent doctor                 Check environment and dependencies
  devagent config <...>           Inspect or edit global config directly
  devagent sessions               List recent sessions
  devagent auth <...>             Manage provider credentials
  devagent install-lsp            Install LSP servers for code intelligence
  devagent execute                Execute an SDK request and write artifacts
  devagent update                 Update to latest version
  devagent completions <shell>    Generate shell completions (bash/zsh/fish)
  devagent version                Show version and runtime info

Auth:
  devagent auth login             Store API key for a provider
  devagent auth status            Show configured credentials
  devagent auth logout [provider|--all]
                                  Remove stored credentials

Options:
  -f, --file <path>    Read query from file
  --mode <mode>        Interactive safety mode: default, autopilot
  --provider <name>     LLM provider (anthropic, openai, devagent-api, deepseek, openrouter, ollama, chatgpt, github-copilot)
  --model <id>          Model ID
  --max-iterations <n>  Max tool-call iterations (default: 0 (unlimited))
  --reasoning <level>   Reasoning effort: low, medium, high
  --resume <id>         Resume a previous session by ID
  --continue            Resume the most recent session
  -v, --verbose         Verbose output (show full tool params and results)
  -q, --quiet           Quiet output (errors only)
  -V, --version         Show version
  -h, --help            Show this help

Environment:
  DEVAGENT_PROVIDER     Default provider
  DEVAGENT_MODEL        Default model
  ANTHROPIC_API_KEY     Anthropic API key
  OPENAI_API_KEY        OpenAI API key
  DEVAGENT_API_KEY      Devagent API gateway key
  DEEPSEEK_API_KEY      DeepSeek API key
  OPENROUTER_API_KEY    OpenRouter API key
  HTTPS_PROXY           HTTPS proxy for outbound provider traffic
  HTTP_PROXY            HTTP proxy for outbound provider traffic
  NO_PROXY              Hosts that should bypass the outbound proxy
  NODE_EXTRA_CA_CERTS   Extra CA bundle for enterprise TLS interception

Installation:
  bun run build && bun run install-cli    Install as 'devagent' command
`;
}

export function printHelp(): void {
  writeStdout(renderHelpText());
}

export function printHelpUsageError(): never {
  process.stderr.write(formatError("Usage: devagent help") + "\n");
  process.stderr.write(renderHelpText() + "\n");
  process.exit(2);
}

export function buildConfigOverridesFromCliArgs(cliArgs: CliArgs): Partial<import("@devagent/runtime").DevAgentConfig> {
  return {
    ...(cliArgs.provider ? { provider: cliArgs.provider } : {}),
    ...(cliArgs.model ? { model: cliArgs.model } : {}),
    ...(cliArgs.safetyMode
      ? {
          approval: {
            ...getSafetyPreset(cliArgs.safetyMode),
            auditLog: false,
            toolOverrides: {},
            pathRules: [],
          },
        }
      : {}),
  };
}
