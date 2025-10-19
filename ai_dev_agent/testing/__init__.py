"""Testing infrastructure for DevAgent.

This module provides utilities for:
- Code coverage enforcement and reporting
- Test fixtures and mocks
- Integration testing framework
- Test quality metrics and analysis
"""

from .coverage_gate import CoverageGate, check_coverage, enforce_coverage
from .coverage_report import CoverageReporter, generate_report

__all__ = [
    'CoverageGate',
    'CoverageReporter',
    'check_coverage',
    'enforce_coverage',
    'generate_report',
]

__version__ = '1.0.0'