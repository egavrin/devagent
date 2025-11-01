import click
import pytest

from ai_dev_agent.cli.handlers.registry_handlers import _build_exec_payload
from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.tools import RUN


def test_build_exec_payload_requires_cmd():
    ctx = click.Context(click.Command("test"), obj={"settings": Settings()})
    with pytest.raises(click.ClickException) as exc:
        _build_exec_payload(ctx, {})
    message = str(exc.value)
    assert f"{RUN} requires 'cmd'" in message
    assert "System:" in message


def test_build_exec_payload_windows_autocorrect(monkeypatch):
    ctx = click.Context(click.Command("test"), obj={"settings": Settings()})

    monkeypatch.setattr(
        "ai_dev_agent.cli.handlers.registry_handlers.build_system_context",
        lambda: {"os": "Windows"},
    )

    payload, _ = _build_exec_payload(ctx, {"cmd": "ls"})
    assert payload["cmd"].startswith("dir")
