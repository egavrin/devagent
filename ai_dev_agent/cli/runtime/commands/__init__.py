"""CLI runtime command registry."""

from .chat import chat_command, execute_chat
from .design import create_design_command, execute_create_design
from .generate_tests import execute_generate_tests, generate_tests_command
from .query import execute_query, query_command
from .review import execute_review, review_command
from .write_code import execute_write_code, write_code_command

__all__ = [
    "query_command",
    "execute_query",
    "review_command",
    "chat_command",
    "execute_chat",
    "create_design_command",
    "generate_tests_command",
    "write_code_command",
    "execute_create_design",
    "execute_generate_tests",
    "execute_write_code",
    "execute_review",
]
