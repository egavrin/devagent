"""Core infrastructure primitives for DevAgent."""

from __future__ import annotations

from .approval import ApprovalManager, ApprovalPolicy
from .tree_sitter import AST_QUERY_TEMPLATES
from .tree_sitter import MANAGER as TREE_SITTER_MANAGER
from .tree_sitter import (
    SUMMARY_QUERY_TEMPLATES,
    TreeSitterManager,
    build_capture_query,
    build_field_capture_query,
)
from .tree_sitter import detect_language
from .tree_sitter import detect_language as detect_tree_sitter_language
from .tree_sitter import ensure_language
from .tree_sitter import ensure_language as ensure_tree_sitter_language
from .tree_sitter import ensure_parser
from .tree_sitter import ensure_parser as ensure_tree_sitter_parser
from .tree_sitter import get_ast_query, get_summary_queries, iter_ast_queries
from .tree_sitter import language_object as tree_sitter_language_object
from .tree_sitter import node_text
from .tree_sitter import node_text as tree_sitter_node_text
from .tree_sitter import normalise_language
from .tree_sitter import normalise_language as normalise_tree_sitter_language
from .tree_sitter import slice_bytes
from .tree_sitter import slice_bytes as tree_sitter_slice_bytes
from .utils import (
    ARTIFACTS_ROOT,
    BudgetedLLMClient,
    ContextBudgetConfig,
    DevAgentConfig,
    InMemoryStateStore,
    PlanSession,
    Settings,
    build_tool_signature,
    canonical_tool_name,
    config_from_settings,
    configure_logging,
    display_tool_name,
    ensure_context_budget,
    estimate_tokens,
    expand_tool_aliases,
    extract_keywords,
    find_config_in_parents,
    get_correlation_id,
    get_logger,
    load_devagent_yaml,
    load_settings,
    prune_messages,
    set_correlation_id,
    summarize_text,
    tool_aliases,
    tool_category,
    tool_signature,
    write_artifact,
)

__all__ = [
    "ARTIFACTS_ROOT",
    "AST_QUERY_TEMPLATES",
    "SUMMARY_QUERY_TEMPLATES",
    "TREE_SITTER_MANAGER",
    "ApprovalManager",
    "ApprovalPolicy",
    "BudgetedLLMClient",
    "ContextBudgetConfig",
    "DevAgentConfig",
    "InMemoryStateStore",
    "PlanSession",
    "Settings",
    "TreeSitterManager",
    "build_capture_query",
    "build_field_capture_query",
    "build_tool_signature",
    "canonical_tool_name",
    "config_from_settings",
    "configure_logging",
    "detect_language",
    "display_tool_name",
    "ensure_context_budget",
    "ensure_language",
    "ensure_parser",
    "estimate_tokens",
    "expand_tool_aliases",
    "extract_keywords",
    "find_config_in_parents",
    "get_ast_query",
    "get_correlation_id",
    "get_logger",
    "get_summary_queries",
    "iter_ast_queries",
    "load_devagent_yaml",
    "load_settings",
    "node_text",
    "normalise_language",
    "prune_messages",
    "set_correlation_id",
    "slice_bytes",
    "summarize_text",
    "tool_aliases",
    "tool_category",
    "tool_signature",
    "tree_sitter_language_object",
    "write_artifact",
]
