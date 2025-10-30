from __future__ import annotations

from pathlib import Path

from ai_dev_agent.tools.analysis.security import scan_for_secrets


def test_secret_scan_detects_known_patterns(tmp_path: Path) -> None:
    target = tmp_path / "config.txt"
    target.write_text("aws_key=AKIAABCDEFGHIJKLMNOP\n", encoding="utf-8")
    result = scan_for_secrets(tmp_path, ["config.txt"])
    assert result.count >= 1
    assert any(f.detector == "aws_access_key" for f in result.findings)


def test_secret_scan_detects_high_entropy(tmp_path: Path) -> None:
    target = tmp_path / "tokens.env"
    target.write_text('TOKEN="ABCD1234EFGH5678IJKL9012MNOPQRST"\n', encoding="utf-8")
    result = scan_for_secrets(tmp_path, ["tokens.env"])
    assert result.count >= 1
    assert any(f.detector == "high_entropy" for f in result.findings)
