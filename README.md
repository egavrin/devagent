# DevAgent

> **Proof of Concept**: LLM-powered CLI for development workflows. Interact with your codebase using natural language.

## Installation

```bash
git clone https://github.com/egavrin/devagent.git
cd devagent
pip install -e .
```

## Setup

1. Copy the config template:
```bash
cp .devagent.toml.example .devagent.toml
```

2. Edit `.devagent.toml` with your API details:
```toml
provider = "deepseek"
model = "deepseek-coder"
api_key = "your-api-key"
base_url = "https://api.deepseek.com"
auto_approve_code = false
```

### Setup troubleshooting

#### Error during `pip install...`: Consider using a build backend that supports PEP 660

In case you see this error, try upgrading your `pip` and `setuptools` to the newest version:

```bash
sudo python -m pip install --upgrade pip
sudo python -m pip install --upgrade setuptools
```

#### Unable to find `devagent` command after successfull install

For local installation, `devagent` launcher is created in `$HOME/.local/bin` directory, make sure it is in your `$PATH`.

## Usage

### One-shot queries

```bash
devagent "summarize this repository"
devagent "find all TODO comments"
devagent "explain how the config system works"
```

### Interactive chat session

```bash
devagent chat
```

### Development workflow integration

```bash
# Code review
devagent "review my last commit for issues"

# Planning
devagent "plan how to add user authentication"

# Code maintenance
devagent "refactor the database module"
```

### Custom LLM Workflows

DevAgent supports customizing the LLM behavior with three powerful options:

#### `--system`: Custom System Prompt

Extend or customize the system prompt to define LLM behavior and role:

```bash
# Inline system prompt
devagent query --system "You are a security expert" "review this code"

# System prompt from file
devagent query --system prompts/code_reviewer.md "analyze main.py"
```

#### `--prompt`: User Prompt from File or String

Load user prompts from files or pass them inline:

```bash
# Prompt from file
devagent --prompt input.txt

# Combine with system context
devagent --system "Explain like I'm 5" --prompt question.txt
```

#### `--format`: Structured Output with JSON Schema

Specify the output format using JSON Schema:

```bash
# Request structured JSON output
devagent query \
  --system prompts/reviewer.md \
  --prompt code.py \
  --format schemas/code_review.json
```

**Example format schema** (`schemas/code_review.json`):
```json
{
  "type": "object",
  "properties": {
    "issues": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "severity": {"type": "string"},
          "line": {"type": "integer"},
          "description": {"type": "string"}
        }
      }
    },
    "summary": {"type": "string"}
  },
  "required": ["issues", "summary"]
}
```

**Key Features:**
- **File auto-detection**: If argument is a file path, reads content; otherwise uses as literal string
- **Generic mechanism**: Works for any workflow - code review, data extraction, test generation, etc.
- **Composable**: Combine with existing `--plan`, `--direct` flags

### `--plan`: Work Planning with LLM Intelligence

DevAgent can analyze query complexity and create structured work plans:

```bash
# LLM assesses complexity and routes appropriately
devagent --plan "how many lines in commands.py"
# Result: DIRECT execution (simple query, no planning overhead)

devagent --plan "implement user authentication with tests"
# Result: Creates plan with multiple tasks, executes sequentially
```

**Key Features:**
- **LLM-driven complexity assessment**: Automatically determines if planning is needed
- **Smart task breakdown**: Creates minimal necessary tasks (1 for simple, 3-5 for complex)
- **Early termination**: Stops when query is answered (no redundant verification)
- **Progress tracking**: Shows completion percentage and task status

**Auto-Detection and Planning:**
```bash
# Complex queries automatically trigger planning mode
devagent "design and implement a logging system"
devagent --plan "add user authentication feature"

# Specialized commands for structured workflows
devagent create-design "REST API" --output design.md
devagent generate-tests "auth module" --coverage 95
devagent write-code design.md --test-file tests/test_auth.py
```

See [CLI Commands Reference](AGENTS.md#cli-commands-reference) for details.

## Testing & Quality

```bash
pip install -e .[dev]

# Fast tests only (default, ~5 seconds)
pytest

# Full suite including slow integration tests (~3 minutes)
pytest -m ""

# Run code quality checks
make quality

# Auto-fix code issues
make fix
```

See [Development Guide](docs/DEVELOPMENT.md) for complete quality tooling and testing guidance.

## Documentation

- **[User Guide](docs/USER_GUIDE.md)** - Complete usage instructions and examples
- **[Development Guide](docs/DEVELOPMENT.md)** - Architecture, testing, and contribution guidelines
- **[API Reference](docs/API_REFERENCE.md)** - Technical documentation for all components
- **[Changelog](docs/CHANGELOG.md)** - Version history and current status
- **[AGENTS.md](AGENTS.md)** - AI agent instructions and automation guide

That's it. This PoC demonstrates how LLMs can assist development workflows while keeping humans in control.
