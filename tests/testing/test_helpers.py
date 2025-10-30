import json
import logging
import os
import time

from ai_dev_agent.testing import helpers


def test_run_with_timeout_success_and_timeout():
    completed, result = helpers.run_with_timeout(lambda x: x + 1, 0.5, 2)
    assert completed is True and result == 3

    start = time.perf_counter()
    completed, _ = helpers.run_with_timeout(time.sleep, 0.1, 1)
    duration = time.perf_counter() - start
    assert completed is False
    assert duration < 0.5  # returns shortly after timeout


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

    json1 = {"a": 1, "b": 2}
    json2 = json.dumps({"a": 1, "b": 2, "c": 3})
    assert helpers.compare_json_structures(json1, json2, ignore_keys=["c"])

    schema = {"a": {"type": "string", "required": True}}
    is_valid, errors = helpers.validate_schema({"a": "ok"}, schema)
    assert is_valid and errors == []

    invalid, errors = helpers.validate_schema({}, schema)
    assert invalid is False and errors == ["a is required"]


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
        return "value"

    assert sample_test() == "value"
    metrics = sample_test._test_metrics[-1]
    assert metrics.test_name == "sample_test"
    assert metrics.passed is True
