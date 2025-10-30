"""Tests for CLI commands module."""

from __future__ import annotations

from click.testing import CliRunner

from ai_dev_agent.cli.commands import NaturalLanguageGroup, cli


def test_natural_language_group_basic():
    """Test NaturalLanguageGroup basic functionality."""
    group = NaturalLanguageGroup()
    assert isinstance(group, NaturalLanguageGroup)


def test_cli_help():
    """Test CLI help command works."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "AI-assisted development agent CLI" in result.output


def test_cli_verbose_flag():
    """Test CLI verbose flag."""
    runner = CliRunner()
    # Just test it doesn't crash
    result = runner.invoke(cli, ["--verbose", "--help"])
    assert result.exit_code == 0


def test_cli_plan_flag():
    """Test CLI plan flag."""
    runner = CliRunner()
    result = runner.invoke(cli, ["--plan", "--help"])
    assert result.exit_code == 0


def test_query_command_no_prompt():
    """Test query command with no prompt raises error."""
    runner = CliRunner()
    result = runner.invoke(cli, ["query"])
    assert result.exit_code != 0
    assert "Provide a request" in result.output or result.exception is not None


def test_natural_language_group_resolve_with_plan_flag(monkeypatch):
    """Test NaturalLanguageGroup resolve_command with --plan flag."""
    import click

    # Create a dummy command to avoid UsageError
    @click.command()
    def query():
        pass

    group = NaturalLanguageGroup()
    group.add_command(query)

    ctx = click.Context(group)

    # Test with --plan flag - should filter it out and set meta
    args = ["--plan", "test", "input"]
    try:
        _name, _cmd, _filtered_args = group.resolve_command(ctx, args)
        # Should have filtered the flag and set pending prompt
        assert (
            ctx.meta.get("_use_planning") is True or ctx.meta.get("_pending_nl_prompt") is not None
        )
    except click.UsageError:
        # May raise if it tries to route to query command
        assert (
            ctx.meta.get("_pending_nl_prompt") is not None
            or ctx.meta.get("_use_planning") is not None
        )


def test_natural_language_group_resolve_with_direct_flag(monkeypatch):
    """Test NaturalLanguageGroup resolve_command with --direct flag."""
    import click

    # Create a dummy command to avoid UsageError
    @click.command()
    def query():
        pass

    group = NaturalLanguageGroup()
    group.add_command(query)

    ctx = click.Context(group)

    # Test with --direct flag - should filter it out and set meta
    args = ["--direct", "test", "input"]
    try:
        _name, _cmd, _filtered_args = group.resolve_command(ctx, args)
        # Should have filtered the flag and set pending prompt
        assert (
            ctx.meta.get("_use_planning") is False or ctx.meta.get("_pending_nl_prompt") is not None
        )
    except click.UsageError:
        # May raise if it tries to route to query command
        assert (
            ctx.meta.get("_pending_nl_prompt") is not None
            or ctx.meta.get("_use_planning") is not None
        )
