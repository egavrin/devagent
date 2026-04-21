import { hasHelpFlag, writeStdout } from "./shared.js";

function renderCompletionsHelpText(): string {
  return `Usage:
  devagent completions <bash|zsh|fish>

Generate shell completions for the public CLI surface.`;
}
const COMMANDS = [
  "setup", "doctor", "config", "update", "completions",
  "version", "sessions", "review", "auth", "execute",
];
const FLAGS = [
  "--help", "--version", "--provider", "--model", "--max-iterations",
  "--reasoning", "--resume", "--continue", "--mode", "--verbose", "--quiet", "--file",
];

export function runCompletions(args: ReadonlyArray<string> = []): void {
  if (hasHelpFlag(args)) {
    writeStdout(renderCompletionsHelpText());
    return;
  }

  const shell = args[0] ?? "";
  switch (shell) {
    case "bash":
      writeStdout(bashCompletions());
      writeStdout("\n# Add to ~/.bashrc:\n#   eval \"$(devagent completions bash)\"");
      break;
    case "zsh":
      writeStdout(zshCompletions());
      writeStdout("\n# Add to ~/.zshrc:\n#   eval \"$(devagent completions zsh)\"");
      break;
    case "fish":
      writeStdout(fishCompletions());
      writeStdout("\n# Save to ~/.config/fish/completions/devagent.fish:\n#   devagent completions fish > ~/.config/fish/completions/devagent.fish");
      break;
    default:
      writeStdout("Usage: devagent completions <bash|zsh|fish>");
      writeStdout("\nExamples:");
      writeStdout("  eval \"$(devagent completions bash)\"   # Add to ~/.bashrc");
      writeStdout("  eval \"$(devagent completions zsh)\"    # Add to ~/.zshrc");
      writeStdout("  devagent completions fish > ~/.config/fish/completions/devagent.fish");
      break;
  }
}

function bashCompletions(): string {
  return `_devagent_completions() {
  local cur="\${COMP_WORDS[COMP_CWORD]}"
  local prev="\${COMP_WORDS[COMP_CWORD-1]}"

  case "\${prev}" in
    devagent)
      COMPREPLY=( $(compgen -W "${COMMANDS.join(" ")} ${FLAGS.join(" ")}" -- "\${cur}") )
      return 0
      ;;
    config)
      COMPREPLY=( $(compgen -W "get set path" -- "\${cur}") )
      return 0
      ;;
    auth)
      COMPREPLY=( $(compgen -W "login status logout" -- "\${cur}") )
      return 0
      ;;
    completions)
      COMPREPLY=( $(compgen -W "bash zsh fish" -- "\${cur}") )
      return 0
      ;;
    --provider)
      COMPREPLY=( $(compgen -W "anthropic openai devagent-api deepseek openrouter ollama chatgpt github-copilot" -- "\${cur}") )
      return 0
      ;;
    --reasoning)
      COMPREPLY=( $(compgen -W "low medium high" -- "\${cur}") )
      return 0
      ;;
  esac

  if [[ "\${cur}" == -* ]]; then
    COMPREPLY=( $(compgen -W "${FLAGS.join(" ")}" -- "\${cur}") )
  else
    COMPREPLY=( $(compgen -W "${COMMANDS.join(" ")}" -- "\${cur}") )
  fi
}
complete -F _devagent_completions devagent`;
}

function zshCompletions(): string {
  return `#compdef devagent

_devagent() {
  local -a commands flags

  commands=(
${COMMANDS.map((c) => `    '${c}:${c} command'`).join("\n")}
  )

  flags=(
    '--help[Show help]'
    '--version[Show version]'
    '--provider[LLM provider]:provider:(anthropic openai devagent-api deepseek openrouter ollama chatgpt github-copilot)'
    '--model[Model ID]:model:'
    '--max-iterations[Max iterations]:number:'
    '--reasoning[Reasoning effort]:level:(low medium high)'
    '--resume[Resume session]:session_id:'
    '--continue[Resume most recent session]'
    '--mode[Interactive safety mode]:mode:(default autopilot)'
    '--verbose[Verbose output]'
    '--quiet[Quiet output]'
    '--file[Read query from file]:file:_files'
  )

  _arguments -s \\
    '1:command:->command' \\
    '*::arg:->args' \\
    \${flags}

  case \$state in
    command)
      _describe 'command' commands
      ;;
    args)
      case \$words[1] in
        config) _values 'subcommand' get set path ;;
        auth) _values 'subcommand' login status logout ;;
        completions) _values 'shell' bash zsh fish ;;
      esac
      ;;
  esac
}

_devagent`;
}

function fishCompletions(): string {
  const lines = [
    "# devagent completions for fish",
    "complete -c devagent -e",
    "",
    "# Commands",
    ...COMMANDS.map((c) => `complete -c devagent -n '__fish_use_subcommand' -a '${c}' -d '${c}'`),
    "",
    "# Flags",
    "complete -c devagent -l help -s h -d 'Show help'",
    "complete -c devagent -l version -s V -d 'Show version'",
    "complete -c devagent -l provider -x -a 'anthropic openai devagent-api deepseek openrouter ollama chatgpt github-copilot' -d 'LLM provider'",
    "complete -c devagent -l model -x -d 'Model ID'",
    "complete -c devagent -l max-iterations -x -d 'Max iterations'",
    "complete -c devagent -l reasoning -x -a 'low medium high' -d 'Reasoning effort'",
    "complete -c devagent -l resume -x -d 'Resume session by ID'",
    "complete -c devagent -l continue -d 'Resume most recent session'",
    "complete -c devagent -l mode -d 'Interactive safety mode' -a 'default autopilot'",
    "complete -c devagent -l verbose -s v -d 'Verbose output'",
    "complete -c devagent -l quiet -s q -d 'Quiet output'",
    "complete -c devagent -l file -s f -r -d 'Read query from file'",
    "",
    "# Subcommands",
    "complete -c devagent -n '__fish_seen_subcommand_from config' -a 'get set path'",
    "complete -c devagent -n '__fish_seen_subcommand_from auth' -a 'login status logout'",
    "complete -c devagent -n '__fish_seen_subcommand_from completions' -a 'bash zsh fish'",
  ];
  return lines.join("\n");
}
