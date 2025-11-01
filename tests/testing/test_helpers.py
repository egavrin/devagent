import json
import logging
import os
import time

import pytest

from ai_dev_agent.testing import helpers


def test_run_with_timeout_success_and_timeout():
    completed, result = helpers.run_with_timeout(lambda x: x + 1, 0.5, 2)
    assert completed is True and result == 3

    start = time.perf_counter()
    completed, _ = helpers.run_with_timeout(time.sleep, 0.1, 1)
    duration = time.perf_counter() - start
    assert completed is False
    assert duration < 0.5  # returns shortly after timeout


def test_run_with_timeout_propagates_exceptions(monkeypatch):
    class ImmediateThread:
        def __init__(self, target):
            self._target = target
            self.daemon = False

        def start(self):
            self._target()

        def join(self, timeout):
            return None

    monkeypatch.setattr("threading.Thread", ImmediateThread)

    def raises():
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        helpers.run_with_timeout(raises, timeout=0.1)


def test_assert_files_equal_with_whitespace(tmp_path):
    file_a = tmp_path / "a.txt"
    file_b = tmp_path / "b.txt"
    file_a.write_text("hello   world")
    file_b.write_text("hello world")
    helpers.assert_files_equal(file_a, file_b, ignore_whitespace=True)


def test_capture_logs_is_context_manager_and_collects(monkeypatch):
    logger = logging.getLogger("test-logger")

    with helpers.capture_logs("test-logger") as records:
        logger.warning("hello world")

    assert records and records[0]["message"] == "hello world"


def test_temporary_env_sets_and_restores():
    os.environ.pop("DEVAGENT_TEST_ENV", None)
    with helpers.temporary_env(DEVAGENT_TEST_ENV="value"):
        assert os.environ["DEVAGENT_TEST_ENV"] == "value"
    assert "DEVAGENT_TEST_ENV" not in os.environ


def test_wait_for_condition_and_mock_subprocess(monkeypatch):
    flag = {"ready": False}

    def set_flag():
        flag["ready"] = True

    helpers.run_with_timeout(set_flag, 0.5)
    assert helpers.wait_for_condition(lambda: flag["ready"], timeout=0.2)

    mock = helpers.mock_subprocess_run({"echo ok": {"stdout": "ok", "returncode": 0}})
    result = mock(["echo", "ok"])
    assert result.stdout == "ok" and result.returncode == 0

    missing = mock(["unknown"])
    assert missing.returncode == 1


def test_generate_data_compare_json_and_validate_schema(tmp_path):
    strings = helpers.generate_test_data("strings", count=2, length=3)
    assert len(strings) == 2 and all(len(s) == 3 for s in strings)

    numbers = helpers.generate_test_data("numbers", count=2, min=1, max=1)
    assert numbers == [1, 1]

    dates = helpers.generate_test_data("dates", count=1)
    assert len(dates) == 1

    floats = helpers.generate_test_data("floats", count=2, min=0.0, max=1.0)
    assert len(floats) == 2 and all(0.0 <= value <= 1.0 for value in floats)

    dicts = helpers.generate_test_data("dicts", count=1, keys=["x"])
    assert len(dicts) == 1 and set(dicts[0].keys()) == {"x"}

    files = helpers.generate_test_data("files", count=2, extension="py")
    assert files == ["file_0.py", "file_1.py"]

    with pytest.raises(ValueError):
        helpers.generate_test_data("unknown")

    json1 = {"a": 1, "b": [1, 2]}
    json2 = json.dumps({"a": 1, "b": [1, 2, 3], "c": 3})
    assert helpers.compare_json_structures(json1, json2, ignore_keys=["c", "b"])

    schema = {
        "a": {"type": "string", "required": True},
        "b": {"type": "number"},
        "c": {"type": "boolean"},
        "d": {"type": "array"},
        "e": {"type": "object", "properties": {"nested": {"type": "string", "required": True}}},
    }
    is_valid, errors = helpers.validate_schema({"a": "ok"}, schema)
    assert is_valid and errors == []

    invalid_input = {"a": 42, "b": "str", "c": "nope", "d": "not-list", "e": {"nested": None}}
    invalid, errors = helpers.validate_schema(invalid_input, schema)
    assert invalid is False
    assert "a must be a string" in errors
    assert "b must be a number" in errors
    assert "c must be a boolean" in errors
    assert "d must be an array" in errors
    assert "e.nested is required" in errors


def test_cleanup_artifacts(tmp_path):
    file_path = tmp_path / "file.txt"
    file_path.write_text("data")
    dir_path = tmp_path / "dir"
    dir_path.mkdir()
    (dir_path / "nested.txt").write_text("nested")

    helpers.cleanup_test_artifacts(file_path, dir_path)
    assert not file_path.exists()
    assert not dir_path.exists()


def test_measure_test_performance_records_metrics():
    @helpers.measure_test_performance
    def sample_test():
        sample_test.record_assertion()
        assert True
        return "value"

    assert sample_test() == "value"
    metrics = sample_test._test_metrics[-1]
    assert metrics.test_name == "sample_test"
    assert metrics.passed is True
    assert metrics.assertions == 1


def test_measure_test_performance_failure_records():
    @helpers.measure_test_performance
    def broken():
        raise ValueError("fail")

    with pytest.raises(ValueError):
        broken()

    metrics = broken._test_metrics[-1]
    assert metrics.passed is False
    assert metrics.assertions == 0
