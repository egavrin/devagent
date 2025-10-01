from __future__ import annotations

import subprocess
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import pytest
from click.testing import CliRunner

import ai_dev_agent.cli as cli_module
from ai_dev_agent.cli import cli
from ai_dev_agent.cli.router import IntentDecision
from ai_dev_agent.providers.llm.base import ToolCall, ToolCallResult
from ai_dev_agent.core.utils.config import Settings


Predicate = Callable[[str], bool]
Rule = Tuple[Predicate, IntentDecision]


@dataclass
class AssistHarness:
    repo_root: Path
    runner: CliRunner
    _rules: list[Rule]
    _client_ref: dict[str, object]

    def configure_router(self, *rules: Rule) -> None:
        self._rules[:] = list(rules)

    def invoke(self, prompt: str, *, expect_exit: int | None = 0) -> tuple:
        start = time.perf_counter()
        result = self.runner.invoke(cli, [prompt], catch_exceptions=False)
        duration = time.perf_counter() - start
        if expect_exit is not None:
            assert result.exit_code == expect_exit, result.output
        return result, duration

    def read_text(self, relative: str) -> str:
        return (self.repo_root / relative).read_text(encoding="utf-8")

    def path(self, relative: str) -> Path:
        return self.repo_root / relative

    def set_client(self, client: object) -> None:
        self._client_ref["client"] = client


@pytest.fixture()
def assist_harness(tmp_path: Path, monkeypatch) -> AssistHarness:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    examples = repo_root / "examples"
    examples.mkdir()
    (examples / "python_app.py").write_text(
        "def greet(name):\n    return f\"Hello, {name}!\"\n",
        encoding="utf-8",
    )
    (examples / "javascript_app.js").write_text(
        "function buildGreeting(name) {\n  return `Hello, ${name}!`;\n}\n",
        encoding="utf-8",
    )
    (examples / "go_app.go").write_text(
        "package main\n\nimport \"fmt\"\n\nfunc main() {\n    fmt.Println(\"Hello from Go sample\")\n}\n",
        encoding="utf-8",
    )
    (repo_root / "README.md").write_text("Sample workspace for assist tests.\n", encoding="utf-8")

    subprocess.run(["git", "init"], cwd=repo_root, check=True, capture_output=True)

    settings = Settings()
    settings.workspace_root = repo_root
    settings.state_file = repo_root / ".devagent" / "state.json"
    settings.ensure_state_dir()
    settings.api_key = "test-key"
    settings.auto_approve_code = True
    settings.react_enable_planner = False

    monkeypatch.setattr(cli_module, "load_settings", lambda path=None: settings)

    class DummyClient:
        pass

    dummy_client = DummyClient()
    client_ref: dict[str, object] = {"client": dummy_client}
    monkeypatch.setattr(cli_module, "get_llm_client", lambda ctx: client_ref["client"])

    router_rules: list[Rule] = []

    class StubRouter:
        def __init__(self, client, settings_obj, **kwargs) -> None:
            self.client = client
            self.settings = settings_obj
            self.extra = kwargs

        def route(self, prompt: str) -> IntentDecision:
            for predicate, decision in router_rules:
                if predicate(prompt):
                    return decision
            raise AssertionError(f"No router rule matched prompt: {prompt!r}")

    monkeypatch.setattr(cli_module, "IntentRouter", StubRouter)
    monkeypatch.chdir(repo_root)

    runner = CliRunner()
    return AssistHarness(repo_root=repo_root, runner=runner, _rules=router_rules, _client_ref=client_ref)


def test_assist_code_search_returns_expected_matches(assist_harness: AssistHarness) -> None:
    assist_harness.configure_router(
        (
            lambda prompt: "find greet" in prompt.lower(),
            IntentDecision(tool="code.search", arguments={"query": "def greet"}),
        ),
    )

    result, duration = assist_harness.invoke("Find greet function definition")

    assert "examples/python_app.py" in result.output
    assert "def greet" in result.output
    assert 0.0 < duration < 2.0


def test_assist_can_patch_existing_file(assist_harness: AssistHarness) -> None:
    before = assist_harness.read_text("examples/python_app.py")
    assert "Return a greeting" not in before

    diff_text = textwrap.dedent(
        '''
        diff --git a/examples/python_app.py b/examples/python_app.py
        index 1111111..2222222 100644
        --- a/examples/python_app.py
        +++ b/examples/python_app.py
        @@ -1,2 +1,3 @@
        -def greet(name):
        -    return f"Hello, {name}!"
        +def greet(name):
        +    """Return a greeting for the given name."""
        +    return f"Hello, {name}!"
        '''
    ).strip() + "\n"

    assist_harness.configure_router(
        (
            lambda prompt: "update greet" in prompt.lower(),
            IntentDecision(tool="fs.write_patch", arguments={"diff": diff_text}),
        ),
    )

    result, duration = assist_harness.invoke("Update greet helper with documentation")

    assert "Patch applied" in result.output
    assert "examples/python_app.py" in result.output
    updated = assist_harness.read_text("examples/python_app.py")
    assert "Return a greeting for the given name." in updated
    assert 0.0 < duration < 2.0


def test_assist_can_create_and_run_script(assist_harness: AssistHarness) -> None:
    diff_text = textwrap.dedent(
        """
        diff --git a/scripts/hello.sh b/scripts/hello.sh
        new file mode 100755
        index 0000000..1111111
        --- /dev/null
        +++ b/scripts/hello.sh
        @@ -0,0 +1,2 @@
        +#!/usr/bin/env bash
        +echo "Hello from assist script"
        """
    ).strip() + "\n"

    assist_harness.configure_router(
        (
            lambda prompt: "create hello script" in prompt.lower(),
            IntentDecision(tool="fs.write_patch", arguments={"diff": diff_text}),
        ),
    )

    create_result, create_duration = assist_harness.invoke("Create hello script for demo")

    assert "Patch applied" in create_result.output
    script_path = assist_harness.path("scripts/hello.sh")
    assert script_path.is_file(), create_result.output
    assert "Hello from assist script" in script_path.read_text(encoding="utf-8")
    assert 0.0 < create_duration < 2.0

    assist_harness.configure_router(
        (
            lambda prompt: "run hello script" in prompt.lower(),
            IntentDecision(tool="exec", arguments={"cmd": "bash", "args": ["scripts/hello.sh"]}),
        ),
    )

    run_result, run_duration = assist_harness.invoke("Run hello script now")

    assert "Exit: 0" in run_result.output
    assert "Hello from assist script" in run_result.output
    assert 0.0 < run_duration < 2.0


def test_assist_direct_response_answers_question(assist_harness: AssistHarness) -> None:
    assist_harness.configure_router(
        (
            lambda prompt: "what is this project" in prompt.lower(),
            IntentDecision(tool=None, arguments={"text": "Sample project overview."}),
        ),
    )

    result, duration = assist_harness.invoke("What is this project about?")

    lines = [line.strip() for line in result.output.strip().splitlines() if line.strip()]
    assert "Sample project overview." in lines
    assert lines[-1].startswith("✅ Completed in ")
    assert lines[-1].endswith("(direct)")
    assert 0.0 < duration < 1.0


def test_assist_react_flow_executes_tool_sequence(assist_harness: AssistHarness) -> None:
    class ScriptedToolClient:
        def __init__(self) -> None:
            self.invocations: List[List[str | None]] = []

        def invoke_tools(self, messages, tools, temperature=0.1, **_kwargs):
            transcript = [getattr(m, "content", None) for m in messages]
            self.invocations.append(transcript)
            if len(self.invocations) == 1:
                return ToolCallResult(
                    calls=[ToolCall(name="code_search", arguments={"query": "def greet"}, call_id="call_1")],
                    message_content=None,
                    raw_tool_calls=[
                        {
                            "id": "call_1",
                            "function": {
                                "name": "code_search",
                                "arguments": '{"query": "def greet"}',
                            },
                        }
                    ],
                )
            return ToolCallResult(calls=[], message_content="ReAct located the greet helper." )

        def configure_timeout(self, *_args, **_kwargs) -> None:
            self._timeout = _kwargs.get("timeout")

        def configure_retry(self, *_args, **_kwargs) -> None:
            self._retry = (_args, _kwargs)

    react_client = ScriptedToolClient()
    react_client.configure_timeout(timeout=60.0)
    react_client.configure_retry("dummy")
    assist_harness.set_client(react_client)
    assist_harness.configure_router()  # ensure direct routing is not used

    prompt = "Use the react loop to find where greet is implemented"
    result, duration = assist_harness.invoke(prompt)

    assert "⚡ Direct execution mode" in result.output
    assert all("Phase" not in line for line in result.output.splitlines())
    assert '🔍 code.search "def greet"' in result.output
    assert "ReAct located the greet helper." in result.output
    assert len(react_client.invocations) >= 2
    assert 0.0 < duration < 3.0
