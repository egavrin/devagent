"""Tests for the approval manager module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from ai_dev_agent.core.approval.approvals import ApprovalManager
from ai_dev_agent.core.approval.policy import ApprovalPolicy


class TestApprovalManager:
    """Tests for ApprovalManager class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.policy = ApprovalPolicy()
        self.temp_file = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log")
        self.temp_file.close()
        self.audit_path = Path(self.temp_file.name)
        self.manager = ApprovalManager(self.policy, self.audit_path)

    def teardown_method(self):
        """Clean up temp files."""
        try:
            self.audit_path.unlink()
        except:
            pass

    def test_init_without_audit_file(self):
        """Test initialization without audit file."""
        manager = ApprovalManager(self.policy)
        assert manager.policy == self.policy
        assert manager.audit_file is None

    def test_init_with_audit_file(self):
        """Test initialization with audit file."""
        assert self.manager.policy == self.policy
        assert self.manager.audit_file == self.audit_path

    @patch("ai_dev_agent.core.approval.approvals.click.confirm")
    def test_require_with_user_prompt(self, mock_confirm):
        """Test requiring approval with user prompt."""
        mock_confirm.return_value = True

        result = self.manager.require("test_action")

        assert result is True
        mock_confirm.assert_called_once_with("Approve test_action?", default=False)

        # Check audit log was written
        with Path(self.audit_path).open() as f:
            content = f.read()
            assert "test_action" in content
            assert "True" in content
            assert "prompt" in content

    @patch("ai_dev_agent.core.approval.approvals.click.confirm")
    def test_require_with_custom_prompt(self, mock_confirm):
        """Test requiring approval with custom prompt."""
        mock_confirm.return_value = False

        result = self.manager.require("test_action", prompt="Allow this action?")

        assert result is False
        mock_confirm.assert_called_once_with("Allow this action?", default=False)

    @patch("ai_dev_agent.core.approval.approvals.click.confirm")
    def test_require_with_default_true(self, mock_confirm):
        """Test requiring approval with default=True."""
        mock_confirm.return_value = True

        result = self.manager.require("test_action", default=True)

        assert result is True
        mock_confirm.assert_called_once_with("Approve test_action?", default=True)

    def test_require_with_emergency_override(self):
        """Test requiring approval with emergency override enabled."""
        self.policy.emergency_override = True

        with patch("ai_dev_agent.core.approval.approvals.click.confirm") as mock_confirm:
            result = self.manager.require("test_action")

        assert result is True
        mock_confirm.assert_not_called()  # Should not prompt user

        # Check audit log
        with Path(self.audit_path).open() as f:
            content = f.read()
            assert "emergency_override" in content

    def test_require_with_auto_approve_plan(self):
        """Test auto-approval for plan action."""
        self.policy.auto_approve_plan = True

        with patch("ai_dev_agent.core.approval.approvals.click.confirm") as mock_confirm:
            result = self.manager.require("plan")

        assert result is True
        mock_confirm.assert_not_called()

        # Check audit log
        with Path(self.audit_path).open() as f:
            content = f.read()
            assert "plan" in content
            assert "auto" in content

    def test_require_with_auto_approve_code(self):
        """Test auto-approval for code action."""
        self.policy.auto_approve_code = True

        with patch("ai_dev_agent.core.approval.approvals.click.confirm") as mock_confirm:
            result = self.manager.require("code")

        assert result is True
        mock_confirm.assert_not_called()

    def test_require_with_auto_approve_shell(self):
        """Test auto-approval for shell action."""
        self.policy.auto_approve_shell = True

        with patch("ai_dev_agent.core.approval.approvals.click.confirm") as mock_confirm:
            result = self.manager.require("shell")

        assert result is True
        mock_confirm.assert_not_called()

    def test_require_with_auto_approve_adr(self):
        """Test auto-approval for adr action."""
        self.policy.auto_approve_adr = True

        with patch("ai_dev_agent.core.approval.approvals.click.confirm") as mock_confirm:
            result = self.manager.require("adr")

        assert result is True
        mock_confirm.assert_not_called()

    def test_maybe_auto_with_emergency_override(self):
        """Test maybe_auto with emergency override."""
        self.policy.emergency_override = True

        assert self.manager.maybe_auto("any_action") is True

    def test_maybe_auto_for_unknown_purpose(self):
        """Test maybe_auto for unknown purpose."""
        assert self.manager.maybe_auto("unknown_action") is False

    def test_maybe_auto_for_each_purpose(self):
        """Test maybe_auto for each known purpose."""
        # Test plan
        self.policy.auto_approve_plan = False
        assert self.manager.maybe_auto("plan") is False
        self.policy.auto_approve_plan = True
        assert self.manager.maybe_auto("plan") is True

        # Test code
        self.policy.auto_approve_code = False
        assert self.manager.maybe_auto("code") is False
        self.policy.auto_approve_code = True
        assert self.manager.maybe_auto("code") is True

        # Test shell
        self.policy.auto_approve_shell = False
        assert self.manager.maybe_auto("shell") is False
        self.policy.auto_approve_shell = True
        assert self.manager.maybe_auto("shell") is True

        # Test adr
        self.policy.auto_approve_adr = False
        assert self.manager.maybe_auto("adr") is False
        self.policy.auto_approve_adr = True
        assert self.manager.maybe_auto("adr") is True

    def test_log_with_no_audit_file(self):
        """Test _log method when no audit file is set."""
        manager = ApprovalManager(self.policy)  # No audit file

        # Should not raise error
        manager._log("test", True, "test_reason")

    @patch("ai_dev_agent.core.approval.approvals.datetime")
    def test_log_with_audit_file(self, mock_datetime):
        """Test _log method with audit file."""
        mock_now = MagicMock()
        mock_now.isoformat.return_value = "2024-01-01T12:00:00"
        mock_datetime.utcnow.return_value = mock_now

        self.manager._log("test_purpose", True, "test_reason")

        with Path(self.audit_path).open() as f:
            content = f.read()
            assert "2024-01-01T12:00:00\ttest_purpose\tTrue\ttest_reason\n" in content

    def test_log_creates_parent_directory(self):
        """Test that _log creates parent directory if it doesn't exist."""
        # Use a path that doesn't exist
        new_audit_path = Path(tempfile.mkdtemp()) / "subdir" / "audit.log"
        manager = ApprovalManager(self.policy, new_audit_path)

        assert not new_audit_path.parent.exists()

        manager._log("test", True, "reason")

        assert new_audit_path.parent.exists()
        assert new_audit_path.exists()

        # Cleanup
        try:
            new_audit_path.unlink()
            new_audit_path.parent.rmdir()
            new_audit_path.parent.parent.rmdir()
        except:
            pass

    def test_log_handles_write_error(self):
        """Test that _log handles write errors gracefully."""
        # Create a read-only directory
        import os

        temp_dir = tempfile.mkdtemp()
        audit_file = Path(temp_dir) / "audit.log"

        manager = ApprovalManager(self.policy, audit_file)

        # Make directory read-only
        Path(temp_dir).chmod(0o444)

        try:
            # Should not raise, just log warning
            with patch("ai_dev_agent.core.approval.approvals.LOGGER") as mock_logger:
                manager._log("test", True, "reason")
                mock_logger.warning.assert_called()
        finally:
            # Restore permissions and cleanup
            Path(temp_dir).chmod(0o755)
            try:
                audit_file.unlink()
            except:
                pass
            Path(temp_dir).rmdir()

    @patch("ai_dev_agent.core.approval.approvals.click.confirm")
    def test_full_workflow_with_prompting(self, mock_confirm):
        """Test full workflow with user prompting."""
        mock_confirm.return_value = True

        # First action - should prompt
        result1 = self.manager.require("custom_action", prompt="Allow custom action?")
        assert result1 is True

        # Enable auto-approval for code
        self.policy.auto_approve_code = True

        # Code action - should not prompt
        result2 = self.manager.require("code")
        assert result2 is True

        # Check confirm was only called once
        assert mock_confirm.call_count == 1

        # Check audit log has both entries
        with Path(self.audit_path).open() as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert "custom_action" in lines[0]
            assert "prompt" in lines[0]
            assert "code" in lines[1]
            assert "auto" in lines[1]

    @patch("ai_dev_agent.core.approval.approvals.LOGGER")
    def test_logging_messages(self, mock_logger):
        """Test that appropriate log messages are generated."""
        # Test emergency override logging
        self.policy.emergency_override = True
        self.manager.require("test")
        mock_logger.warning.assert_called_with(
            "Emergency override enabled: auto-approved %s", "test"
        )

        # Reset
        mock_logger.reset_mock()
        self.policy.emergency_override = False
        self.policy.auto_approve_plan = True

        # Test auto-approval logging
        self.manager.require("plan")
        mock_logger.info.assert_any_call("%s automatically approved by policy.", "Plan")

        # Reset
        mock_logger.reset_mock()
        self.policy.auto_approve_plan = False

        # Test prompt decision logging
        with patch("ai_dev_agent.core.approval.approvals.click.confirm") as mock_confirm:
            mock_confirm.return_value = False
            self.manager.require("action")
            mock_logger.info.assert_called_with("Approval for %s: %s", "action", False)
