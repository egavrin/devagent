"""Public package interface for the DevAgent toolkit."""

from __future__ import annotations

try:  # pragma: no cover - fallback for very old Python versions
    from importlib import metadata as _metadata
except ImportError:  # pragma: no cover - Python <3.8
    _metadata = None  # type: ignore[assignment]

if _metadata is not None:  # pragma: no cover - importlib metadata availability varies
    try:
        __version__ = _metadata.version("ai-dev-agent")
    except Exception:  # pragma: no cover - fallback when not installed
        __version__ = "0.1.0"
else:  # pragma: no cover - fallback without importlib metadata
    __version__ = "0.1.0"

from . import core, engine, providers, tools
from .core import (
    ARTIFACTS_ROOT,
    ApprovalManager,
    ApprovalPolicy,
    BudgetedLLMClient,
    ContextBudgetConfig,
    DevAgentConfig,
    Settings,
    configure_logging,
    ensure_context_budget,
    get_logger,
    load_devagent_yaml,
    load_settings,
)
from .engine.react import (
    ActionRequest,
    EvaluationResult,
    GateConfig,
    MetricsSnapshot,
    Observation,
    ReactiveExecutor,
    RunResult,
    StepRecord,
    TaskSpec,
)
from .providers.llm import (
    DEEPSEEK_DEFAULT_BASE_URL,
    OPENROUTER_DEFAULT_BASE_URL,
    LLMClient,
    LLMConnectionError,
    LLMError,
    LLMRateLimitError,
    LLMResponseError,
    LLMRetryExhaustedError,
    LLMTimeoutError,
    Message,
    RetryConfig,
    StreamHooks,
    create_client,
)
from .tools.code.code_edit.context import ContextGatherer, ContextGatheringOptions, FileContext
from .tools.code.code_edit.editor import CodeEditor, DiffProposal
from .tools.code.code_edit.tree_sitter_analysis import TreeSitterProjectAnalyzer
from .tools.registry import ToolContext, ToolRegistry, ToolSpec, registry

__all__ = [
    "ARTIFACTS_ROOT",
    "DEEPSEEK_DEFAULT_BASE_URL",
    "OPENROUTER_DEFAULT_BASE_URL",
    "ActionRequest",
    "ApprovalManager",
    "ApprovalPolicy",
    "BudgetedLLMClient",
    "CodeEditor",
    "ContextBudgetConfig",
    "ContextGatherer",
    "ContextGatheringOptions",
    "DevAgentConfig",
    "DiffProposal",
    "EvaluationResult",
    "FileContext",
    "GateConfig",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMRetryExhaustedError",
    "LLMTimeoutError",
    "Message",
    "MetricsSnapshot",
    "Observation",
    "ReactiveExecutor",
    "RetryConfig",
    "RunResult",
    "Settings",
    "StepRecord",
    "StreamHooks",
    "TaskSpec",
    "ToolContext",
    "ToolRegistry",
    "ToolSpec",
    "TreeSitterProjectAnalyzer",
    "__version__",
    "configure_logging",
    "core",
    "create_client",
    "engine",
    "ensure_context_budget",
    "get_logger",
    "load_devagent_yaml",
    "load_settings",
    "providers",
    "registry",
    "tools",
]
