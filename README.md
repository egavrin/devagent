# AI Dev Agent

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

### Interactive shell

```bash
devagent shell
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

## Testing

```bash
pip install -e .[dev]
pytest
```

That's it. This PoC demonstrates how LLMs can assist development workflows while keeping humans in control.
