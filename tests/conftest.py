import sys
from pathlib import Path

try:
    import pytest_timeout  # type: ignore  # noqa: F401

    _HAS_PYTEST_TIMEOUT = True
except Exception:  # pragma: no cover - fallback path for environments without plugin
    _HAS_PYTEST_TIMEOUT = False

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_addoption(parser):
    """Register optional ini entries when pytest-timeout is unavailable."""
    if _HAS_PYTEST_TIMEOUT:
        return

    try:
        parser.addini(
            "timeout",
            "Per-test timeout (seconds). Ignored when pytest-timeout plugin is missing.",
            default="0",
        )
    except ValueError:
        # Another plugin already registered the option; safe to ignore.
        pass


def pytest_configure(config):
    """Access timeout setting so the ini option is marked as used."""
    if not _HAS_PYTEST_TIMEOUT:
        # Reading the value ensures pytest considers the ini option consumed.
        config.getini("timeout")
