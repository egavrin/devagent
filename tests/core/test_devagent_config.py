import textwrap
from pathlib import Path

import pytest

from ai_dev_agent.core.utils.devagent_config import DevAgentConfig, load_devagent_yaml


def test_load_devagent_yaml_missing(tmp_path):
    missing = tmp_path / "devagent.yaml"
    assert load_devagent_yaml(missing) is None


def test_load_devagent_yaml_parses_sections(tmp_path):
    config_path = tmp_path / "devagent.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            build:
              cmd: make build
            tests:
              cmd: pytest
            coverage:
              cmd: pytest --cov
              threshold:
                diff: "0.75"
                project: "0.9"
            lint:
              cmd: flake8
            types:
              cmd: mypy
            format:
              cmd: black
            gates:
              - name: diff.size
                lte_lines: "50"
                lte_files: "3"
            sandbox:
              shell_allowlist:
                - ls
              time_limit_sec: "120"
              memory_limit_mb: "512"
            index:
              ctags:
                cmd: ctags
                db: tags.db
                refresh_sec: "60"
            react:
              iteration:
                global_cap: "10"
                phases:
                  design:
                    - name: plan
                      description: outline work
                      max_iterations: 5
            budget_control:
              max_budget: 100
            """
        ),
        encoding="utf-8",
    )

    cfg = load_devagent_yaml(config_path)
    assert isinstance(cfg, DevAgentConfig)
    assert cfg.build_cmd == "make build"
    assert cfg.test_cmd == "pytest"
    assert cfg.coverage_cmd == "pytest --cov"
    assert cfg.threshold_diff == pytest.approx(0.75)
    assert cfg.threshold_project == pytest.approx(0.9)
    assert cfg.diff_limit_lines == 50
    assert cfg.diff_limit_files == 3
    assert cfg.sandbox_allowlist == ("ls",)
    assert cfg.sandbox_time_limit == 120
    assert cfg.sandbox_memory_limit == 512
    assert cfg.ctags_cmd == "ctags"
    assert cfg.ctags_db == "tags.db"
    assert cfg.ctags_refresh_sec == 60
    assert cfg.react_iteration_global_cap == 10
    assert "design" in cfg.react_phase_overrides
    assert cfg.budget_control == {"max_budget": 100}


def test_load_devagent_yaml_invalid_returns_none(tmp_path):
    invalid = tmp_path / "devagent.yaml"
    invalid.write_text("- just\n- a\n- list\n", encoding="utf-8")
    assert load_devagent_yaml(invalid) is None
