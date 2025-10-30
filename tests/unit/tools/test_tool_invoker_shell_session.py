import os
from pathlib import Path

import pytest

from ai_dev_agent.core.utils.config import Settings
from ai_dev_agent.engine.react.tool_invoker import RegistryToolInvoker
from ai_dev_agent.engine.react.types import ActionRequest
from ai_dev_agent.tools import RUN
from ai_dev_agent.tools.execution.shell_session import ShellSessionManager


@pytest.mark.skipif(os.name == "nt", reason="Shell session manager requires POSIX shell")
def test_run_commands_share_shell_session(tmp_path):
    manager = ShellSessionManager(shell=["/bin/sh"])
    session_id = manager.create_session(cwd=Path(tmp_path))

    try:
        invoker = RegistryToolInvoker(
            workspace=Path(tmp_path),
            settings=Settings(workspace_root=Path(tmp_path)),
            shell_session_manager=manager,
            shell_session_id=session_id,
        )

        set_var_action = ActionRequest(
            step_id="s1",
            thought="set env var",
            tool=RUN,
            args={"cmd": "export TEST_VAR=from_shell"},
        )

        observation_set = invoker(set_var_action)
        assert observation_set.success

        read_var_action = ActionRequest(
            step_id="s2",
            thought="read env var",
            tool=RUN,
            args={"cmd": "printf '%s' \"$TEST_VAR\""},
        )

        observation_read = invoker(read_var_action)
        assert observation_read.success
        stdout_tail = observation_read.metrics.get("stdout_tail")
        assert isinstance(stdout_tail, str)
        assert "from_shell" in stdout_tail
    finally:
        manager.close_all()
