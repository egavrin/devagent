"""Tests for approval policy module."""

from ai_dev_agent.core.approval.policy import ApprovalPolicy


def test_approval_policy_defaults():
    """Test ApprovalPolicy default values."""
    policy = ApprovalPolicy()

    # All should be False by default
    assert policy.auto_approve_plan is False
    assert policy.auto_approve_code is False
    assert policy.auto_approve_shell is False
    assert policy.auto_approve_adr is False
    assert policy.emergency_override is False
    assert policy.audit_file is False


def test_approval_policy_custom_values():
    """Test ApprovalPolicy with custom values."""
    policy = ApprovalPolicy(
        auto_approve_plan=True,
        auto_approve_code=True,
        auto_approve_shell=False,
        auto_approve_adr=True,
        emergency_override=False,
        audit_file=True
    )

    assert policy.auto_approve_plan is True
    assert policy.auto_approve_code is True
    assert policy.auto_approve_shell is False
    assert policy.auto_approve_adr is True
    assert policy.emergency_override is False
    assert policy.audit_file is True


def test_approval_policy_partial_values():
    """Test ApprovalPolicy with partial custom values."""
    policy = ApprovalPolicy(
        auto_approve_plan=True,
        audit_file=True
    )

    # Custom values
    assert policy.auto_approve_plan is True
    assert policy.audit_file is True

    # Default values
    assert policy.auto_approve_code is False
    assert policy.auto_approve_shell is False
    assert policy.auto_approve_adr is False
    assert policy.emergency_override is False


def test_approval_policy_is_dataclass():
    """Test that ApprovalPolicy is a proper dataclass."""
    from dataclasses import is_dataclass, fields

    assert is_dataclass(ApprovalPolicy)

    # Check fields
    field_names = [f.name for f in fields(ApprovalPolicy)]
    assert 'auto_approve_plan' in field_names
    assert 'auto_approve_code' in field_names
    assert 'auto_approve_shell' in field_names
    assert 'auto_approve_adr' in field_names
    assert 'emergency_override' in field_names
    assert 'audit_file' in field_names


def test_approval_policy_equality():
    """Test ApprovalPolicy equality comparison."""
    policy1 = ApprovalPolicy(auto_approve_plan=True)
    policy2 = ApprovalPolicy(auto_approve_plan=True)
    policy3 = ApprovalPolicy(auto_approve_plan=False)

    assert policy1 == policy2
    assert policy1 != policy3