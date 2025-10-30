"""Local approval manager implementation."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

import click

from ..utils.logger import get_logger

if TYPE_CHECKING:
    from pathlib import Path

    from .policy import ApprovalPolicy

LOGGER = get_logger(__name__)


class ApprovalManager:
    """Handles human-in-the-loop approvals for sensitive actions."""

    def __init__(self, policy: ApprovalPolicy, audit_file: Path | None = None) -> None:
        self.policy = policy
        self.audit_file = audit_file

    def require(
        self,
        purpose: str,
        default: bool = False,
        *,
        prompt: str | None = None,
    ) -> bool:
        """Request approval for the specified purpose."""
        if self.policy.emergency_override:
            LOGGER.warning("Emergency override enabled: auto-approved %s", purpose)
            self._log(purpose, granted=True, reason="emergency_override")
            return True

        if self.maybe_auto(purpose):
            LOGGER.info("%s automatically approved by policy.", purpose.capitalize())
            self._log(purpose, granted=True, reason="auto")
            return True

        prompt_text = prompt or f"Approve {purpose}?"
        decision = click.confirm(prompt_text, default=default)
        LOGGER.info("Approval for %s: %s", purpose, decision)
        self._log(purpose, granted=decision, reason="prompt")
        return decision

    def maybe_auto(self, purpose: str) -> bool:
        if self.policy.emergency_override:
            return True
        if purpose == "plan":
            return self.policy.auto_approve_plan
        if purpose == "code":
            return self.policy.auto_approve_code
        if purpose == "shell":
            return self.policy.auto_approve_shell
        if purpose == "adr":
            return self.policy.auto_approve_adr
        return False

    def _log(self, purpose: str, granted: bool, reason: str) -> None:
        if not self.audit_file:
            return
        try:
            self.audit_file.parent.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.utcnow().isoformat()
            with self.audit_file.open("a", encoding="utf-8") as handle:
                handle.write(f"{timestamp}\t{purpose}\t{granted}\t{reason}\n")
        except OSError as exc:  # pragma: no cover - best effort logging
            LOGGER.warning("Failed to write approval audit entry: %s", exc)


__all__ = ["ApprovalManager"]
