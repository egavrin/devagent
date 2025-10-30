import pytest

from ai_dev_agent.core.failure_detector import FailurePatternDetector


def test_record_failure_and_summary():
    detector = FailurePatternDetector()
    detector.record_failure("grep", "foo", "not found")
    detector.record_failure("read", "bar.py", "reject: binary")

    summary = detector.get_summary()
    assert "Total failures: 2" in summary
    assert "grep" in summary and "read" in summary


def test_should_give_up_same_operation():
    detector = FailurePatternDetector()
    for _ in range(detector.MAX_SAME_FAILURE):
        detector.record_failure("find", "missing.txt", "not found")

    stop, message = detector.should_give_up("find", "missing.txt")
    assert stop is True
    assert "Repeated Failure" in message


def test_should_give_up_grep_related_terms():
    detector = FailurePatternDetector()
    detector.MAX_GREP_FAILURES = 2
    detector.record_failure("grep", "functionA", "no match")
    detector.record_failure("grep", "FunctionA::impl", "no match")

    stop, message = detector.should_give_up("grep", "functiona")
    assert stop is True
    assert "Multiple Search Failures" in message


def test_should_give_up_read_rejections():
    detector = FailurePatternDetector()
    detector.MAX_READ_REJECTIONS = 2
    detector.record_failure("read", "large.bin", "reject: binary")
    detector.record_failure("read", "huge.log", "reject: too large")

    stop, message = detector.should_give_up("read", "large.bin")
    assert stop is True
    assert "Multiple Read Rejections" in message


def test_reset_clears_state():
    detector = FailurePatternDetector()
    detector.record_failure("grep", "foo", "no match")
    detector.reset()
    assert detector.failure_counts == {}
    assert detector.failed_operations == []
