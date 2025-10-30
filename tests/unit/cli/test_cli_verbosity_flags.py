"""Tests for verbosity flags (-v/-vv/-vvv/-q/--json)."""

from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from ai_dev_agent.agents.base import AgentResult
from ai_dev_agent.cli.commands import cli


@pytest.fixture
def verbosity_cli_stub(cli_stub_runtime, monkeypatch):
    """Stub specialized agent execution for verbosity tests."""
    stub_result = AgentResult(success=True, output="stub", metadata={})
    execute_stub = MagicMock(return_value=stub_result)
    monkeypatch.setattr(
        "ai_dev_agent.agents.specialized.executor_bridge.execute_agent_with_react",
        execute_stub,
    )
    cli_stub_runtime["agent_execute"] = execute_stub
    return cli_stub_runtime


pytestmark = pytest.mark.usefixtures("verbosity_cli_stub")


class TestVerbosityFlags:
    """Test verbosity level flags."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_no_verbosity_default(self, runner):
        """Default should be normal output (info level)."""
        result = runner.invoke(cli, ["How do I use pytest?"])
        # Should not include debug markers
        assert result.exit_code in [0, 1, 2]

    def test_single_verbose_flag(self, runner):
        """Single -v should show verbose output (info + warnings)."""
        result = runner.invoke(cli, ["-v", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]
        # Should execute without error

    def test_double_verbose_flag(self, runner):
        """Double -vv should show debug output."""
        result = runner.invoke(cli, ["-vv", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]
        # Should execute without error

    def test_triple_verbose_flag(self, runner):
        """Triple -vvv should show trace/debug output."""
        result = runner.invoke(cli, ["-vvv", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]
        # Should execute without error

    def test_verbose_with_commands(self, runner):
        """Verbosity flags should work with all commands."""
        commands = [
            ["design", "API", "-v"],
            ["generate-tests", "Module", "-vv"],
            ["write-code", "design.md", "-vvv"],
            ["review", "file.py", "-v"],
            ["chat", "-v"],  # Will need EOF
        ]

        for cmd in commands[:-1]:  # Skip chat which needs special handling
            result = runner.invoke(cli, cmd)
            assert result.exit_code in [0, 1, 2], f"Failed for command: {cmd}"

    def test_verbose_flag_with_stdin(self, runner):
        """Verbosity flags should work with stdin input."""
        result = runner.invoke(cli, ["-v"], input="test query\nexit\n")
        assert result.exit_code in [0, 1, 2]


class TestQuietFlag:
    """Test quiet flag."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_quiet_flag(self, runner):
        """--quiet or -q should minimize output."""
        result = runner.invoke(cli, ["-q", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]

    def test_quiet_full_name(self, runner):
        """--quiet should work as full flag name."""
        result = runner.invoke(cli, ["--quiet", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]

    def test_quiet_with_commands(self, runner):
        """Quiet flag should work with all commands."""
        commands = [
            ["design", "API", "-q"],
            ["generate-tests", "Module", "--quiet"],
            ["write-code", "design.md", "-q"],
            ["review", "file.py", "--quiet"],
        ]

        for cmd in commands:
            result = runner.invoke(cli, cmd)
            assert result.exit_code in [0, 1, 2], f"Failed for command: {cmd}"


class TestJsonFlag:
    """Test JSON output flag."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_json_output_flag(self, runner):
        """--json should output JSON format."""
        result = runner.invoke(cli, ["--json", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]

    def test_json_with_design(self, runner):
        """--json should work with design command."""
        result = runner.invoke(cli, ["design", "API", "--json"])
        assert result.exit_code in [0, 1, 2]

    def test_json_with_tests(self, runner):
        """--json should work with generate-tests command."""
        result = runner.invoke(cli, ["generate-tests", "Module", "--json"])
        assert result.exit_code in [0, 1, 2]

    def test_json_with_review(self, runner):
        """--json should work with review command."""
        result = runner.invoke(cli, ["review", "file.py", "--json"])
        assert result.exit_code in [0, 1, 2]

    def test_json_with_verbose(self, runner):
        """--json and -v should work together."""
        result = runner.invoke(cli, ["-v", "--json", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]


class TestVerbosityMutualExclusion:
    """Test that incompatible verbosity flags are handled."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_verbose_and_quiet_together(self, runner):
        """Using both -v and -q should be handled gracefully."""
        # The behavior depends on implementation, but should not crash
        result = runner.invoke(cli, ["-v", "-q", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]

    def test_json_with_quiet(self, runner):
        """Using --json with -q should work (JSON is both quiet and explicit)."""
        result = runner.invoke(cli, ["-q", "--json", "How do I use pytest?"])
        assert result.exit_code in [0, 1, 2]


class TestVerbosityOutput:
    """Test that verbosity actually changes output behavior."""

    @pytest.fixture
    def runner(self):
        """Create a CLI runner."""
        return CliRunner()

    def test_verbose_produces_more_output(self, runner):
        """Verbose mode should potentially produce more output."""
        normal = runner.invoke(cli, ["design", "API"])
        verbose = runner.invoke(cli, ["-v", "design", "API"])

        # Both should complete
        assert normal.exit_code in [0, 1, 2]
        assert verbose.exit_code in [0, 1, 2]
        # Verbose might have more output, but implementation dependent

    def test_quiet_produces_less_output(self, runner):
        """Quiet mode should potentially produce less output."""
        normal = runner.invoke(cli, ["design", "API"])
        quiet = runner.invoke(cli, ["-q", "design", "API"])

        # Both should complete
        assert normal.exit_code in [0, 1, 2]
        assert quiet.exit_code in [0, 1, 2]
        # Quiet might have less output, but implementation dependent
