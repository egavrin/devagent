"""Approval policy models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ApprovalPolicy:
    auto_approve_plan: bool = False
    auto_approve_code: bool = False
    auto_approve_shell: bool = False
    auto_approve_adr: bool = False
    emergency_override: bool = False
    audit_file: bool = False


__all__ = ["ApprovalPolicy"]
