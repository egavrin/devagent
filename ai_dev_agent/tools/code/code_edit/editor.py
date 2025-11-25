"""Enhanced code editing with iterative fix loops and intelligent context gathering."""

from __future__ import annotations

import difflib
import re
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable
from uuid import uuid4

from ai_dev_agent.core.utils.keywords import extract_keywords
from ai_dev_agent.core.utils.logger import get_logger
from ai_dev_agent.providers.llm import LLMClient, LLMError
from ai_dev_agent.session import SessionManager, build_system_messages
from ai_dev_agent.tools.execution.testing.local_tests import TestResult, TestRunner

from .context import ContextGatherer, ContextGatheringOptions, FileContext
from .diff_utils import DiffError, DiffPreview, DiffProcessor

if TYPE_CHECKING:
    from collections.abc import Iterable

    from ai_dev_agent.core.approval.approvals import ApprovalManager

LOGGER = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a precise senior software engineer and code reviewer. Before proposing edits, infer the "
    "repository's conventions, dependency impact, and quality expectations from the provided context. "
    "Return the minimal unified diff that satisfies the task while preserving established patterns. "
    "Think through potential regressions silently; output only the final diff."
)

USER_TEMPLATE = """
Task Description:
{task_description}

Repository Signals:
- Style Guide: {style_profile}
- Dependency Impact: {dependency_profile}
- Observed Patterns: {pattern_notes}
- Quality Considerations: {quality_notes}

Additional Guidance:
{instructions}

File Contexts:
{file_blocks}

Respond with a unified diff enclosed in ```diff fences. Only include files that require changes.
"""

FIX_TEMPLATE = """
The previous implementation failed validation. Diagnose the failure using the signals below and deliver a corrected diff.

Task Context:
{task_description}

Repository Signals:
- Style Guide: {style_profile}
- Dependency Impact: {dependency_profile}
- Observed Patterns: {pattern_notes}
- Quality Considerations: {quality_notes}

Previous Attempt Diff:
{previous_diff}

Test Results:
{test_output}

Current File Contents:
{file_blocks}

Provide an updated unified diff that fixes the regression and adheres to the established conventions.
"""

FILE_BLOCK_TEMPLATE = """
<file path="{path}" relevance="{relevance:.2f}" reason="{reason}">
{content}
</file>
"""


@dataclass
class FixAttempt:
    """Record of a fix attempt with its results and preview details."""

    attempt_number: int
    diff: str
    preview: DiffPreview | None = None
    validation_errors: list[str] = field(default_factory=list)
    test_result: TestResult | None = None
    error_message: str | None = None
    approved: bool | None = None
    applied: bool = False
    timestamp: float = field(default_factory=time.time)


@dataclass
class DiffProposal:
    """Enhanced diff proposal with validation and preview."""

    diff: str
    raw_response: str
    files: list[Path]
    preview: DiffPreview | None = None
    validation_errors: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    fallback_guidance: str | None = None


@dataclass
class IterativeFixConfig:
    """Configuration for iterative fix behavior."""

    max_attempts: int = 3
    run_tests: bool = True
    test_timeout: float = 60.0
    require_test_success: bool = True
    enable_context_expansion: bool = True


class CodeEditor:
    """Enhanced code editor with iterative fix loops and intelligent context gathering."""

    def __init__(
        self,
        repo_root: Path,
        llm_client: LLMClient,
        approvals: ApprovalManager,
        fix_config: IterativeFixConfig | None = None,
    ) -> None:
        self.repo_root = repo_root
        self.llm_client = llm_client
        self.approvals = approvals
        self.fix_config = fix_config or IterativeFixConfig()

        # Enhanced components
        self.context_gatherer = ContextGatherer(repo_root)
        self.diff_processor = DiffProcessor(repo_root)
        self.test_runner = TestRunner(repo_root) if fix_config and fix_config.run_tests else None

        # Fix attempt tracking
        self.fix_attempts: list[FixAttempt] = []

        # Session management
        self._session_manager = SessionManager.get_instance()
        self._session_id = f"editor-{uuid4()}"
        self._base_system_messages = build_system_messages(
            include_react_guidance=False,
            extra_messages=[SYSTEM_PROMPT],
            workspace_root=self.repo_root,
        )
        self._session_manager.ensure_session(
            self._session_id,
            system_messages=self._base_system_messages,
            metadata={
                "mode": "code-editor",
                "repo_root": str(self.repo_root),
            },
        )

    def gather_context(
        self,
        files: Iterable[str],
        task_description: str | None = None,
        options: ContextGatheringOptions | None = None,
    ) -> list[FileContext]:
        """Gather intelligent context for the given files and task."""
        # Extract keywords from task description for better context discovery
        keywords = (
            extract_keywords(task_description, include_special_terms=True)
            if task_description
            else None
        )

        # Convert files to Path objects for RepoMap personalization
        chat_files = [self.repo_root / f for f in files] if files else None

        return self.context_gatherer.gather_contexts(
            files=files,
            task_description=task_description,
            keywords=keywords,
            chat_files=chat_files,  # Pass files being edited for PageRank personalization
        )

    def propose_diff(
        self,
        task_description: str,
        files: Iterable[str],
        extra_instructions: str = "",
        previous_attempts: list[FixAttempt] | None = None,
    ) -> DiffProposal:
        """Propose a diff with enhanced context and validation."""

        # Gather intelligent context
        contexts = self.gather_context(files, task_description)
        code_contexts = self._filter_code_contexts(contexts)
        style_profile = self._build_style_profile(code_contexts)
        dependency_profile = self._build_dependency_summary(code_contexts)
        latest_attempt = previous_attempts[-1] if previous_attempts else None
        pattern_notes = self._identify_common_patterns(code_contexts)
        quality_notes = self._build_quality_notes(code_contexts, task_description, latest_attempt)

        # Build the prompt
        if previous_attempts:
            # This is a fix attempt
            latest_attempt = previous_attempts[-1]
            file_blocks = self._build_file_blocks(contexts)
            prompt = FIX_TEMPLATE.format(
                task_description=task_description,
                previous_diff=latest_attempt.diff,
                test_output=self._format_test_output(latest_attempt.test_result),
                file_blocks=file_blocks,
                style_profile=style_profile,
                dependency_profile=dependency_profile,
                pattern_notes=pattern_notes,
                quality_notes=quality_notes,
            )
        else:
            # Initial implementation
            file_blocks = self._build_file_blocks(contexts)
            prompt = USER_TEMPLATE.format(
                task_description=task_description,
                instructions=extra_instructions or "No additional guidance",
                file_blocks=file_blocks,
                style_profile=style_profile,
                dependency_profile=dependency_profile,
                pattern_notes=pattern_notes,
                quality_notes=quality_notes,
            )

        # Get LLM response
        session = self._session_manager.ensure_session(
            self._session_id,
            system_messages=self._base_system_messages,
            metadata={
                "mode": "code-editor",
                "task": task_description,
            },
        )
        with session.lock:
            session.metadata["last_context_summary"] = {
                "style": style_profile,
                "dependency": dependency_profile,
            }
        self._session_manager.add_user_message(self._session_id, prompt)

        try:
            # Use default temperature (0.0) for reproducible code generation
            response = self.llm_client.complete(
                self._session_manager.compose(self._session_id),
            )
            self._session_manager.add_assistant_message(self._session_id, response)
        except LLMError as exc:
            LOGGER.warning("Diff proposal fallback engaged: %s", exc)
            self._session_manager.add_system_message(
                self._session_id,
                f"Code editor fallback due to error: {exc}",
            )
            guidance = self._build_fallback_guidance(task_description, contexts, str(exc))
            return DiffProposal(
                diff="",
                raw_response="",
                files=[],
                validation_errors=[],
                fallback_reason=str(exc),
                fallback_guidance=guidance,
            )

        # Extract and validate diff
        try:
            diff_text, validation = self.diff_processor.extract_and_validate_diff(response)
            preview = self.diff_processor.create_preview(diff_text)
            files_touched = self._extract_files_from_diff(diff_text)

            return DiffProposal(
                diff=diff_text,
                raw_response=response,
                files=files_touched,
                preview=preview,
                validation_errors=validation.errors,
            )

        except DiffError as exc:
            LOGGER.error("Failed to extract valid diff: %s", exc)
            return DiffProposal(
                diff="", raw_response=response, files=[], validation_errors=[str(exc)]
            )

    def apply_diff_with_fixes(
        self,
        task_description: str,
        files: Iterable[str],
        extra_instructions: str = "",
        test_command: list[str] | None = None,
        *,
        on_proposal: Callable[[DiffProposal, int], None] | None = None,
    ) -> tuple[bool, list[FixAttempt]]:
        """Apply diffs with iterative fixes until tests pass or attempts are exhausted."""

        attempts: list[FixAttempt] = []
        files_list = list(files)

        for attempt_num in range(1, self.fix_config.max_attempts + 1):
            LOGGER.info("Fix attempt %d/%d", attempt_num, self.fix_config.max_attempts)

            proposal = self.propose_diff(
                task_description=task_description,
                files=files_list,
                extra_instructions=extra_instructions,
                previous_attempts=attempts if attempts else None,
            )

            if proposal.fallback_reason:
                message = (
                    proposal.fallback_guidance
                    or "LLM is unavailable; manual intervention required."
                )
                LOGGER.warning(
                    "Stopping iterative fixes due to fallback: %s", proposal.fallback_reason
                )
                attempts.append(
                    FixAttempt(
                        attempt_number=attempt_num,
                        diff=proposal.diff,
                        preview=proposal.preview,
                        validation_errors=list(proposal.validation_errors),
                        error_message=message,
                        approved=None,
                    )
                )
                return False, attempts

            if proposal.validation_errors:
                error_msg = f"Diff validation failed: {'; '.join(proposal.validation_errors)}"
                LOGGER.error(error_msg)
                attempts.append(
                    FixAttempt(
                        attempt_number=attempt_num,
                        diff=proposal.diff,
                        preview=proposal.preview,
                        validation_errors=list(proposal.validation_errors),
                        error_message=error_msg,
                        approved=None,
                    )
                )
                continue

            if proposal.preview:
                LOGGER.info("Diff preview: %s", proposal.preview.summary)
                if proposal.preview.validation_result.warnings:
                    for warning in proposal.preview.validation_result.warnings:
                        LOGGER.warning("Diff warning: %s", warning)

            if on_proposal:
                try:
                    on_proposal(proposal, attempt_num)
                except (
                    Exception
                ) as exc:  # pragma: no cover - callback errors should not break fixes
                    LOGGER.warning("Diff preview callback failed: %s", exc)

            attempt_record = FixAttempt(
                attempt_number=attempt_num,
                diff=proposal.diff,
                preview=proposal.preview,
                validation_errors=list(proposal.validation_errors),
            )

            try:
                approved = self._apply_diff_with_approval(proposal)
            except Exception as exc:
                error_msg = f"Failed to apply diff: {exc}"
                LOGGER.error(error_msg)
                attempt_record.error_message = error_msg
                attempt_record.approved = None
                attempts.append(attempt_record)
                continue

            attempt_record.approved = approved
            if not approved:
                LOGGER.info("Diff application was not approved")
                attempt_record.error_message = "Diff application declined by user."
                attempts.append(attempt_record)
                return False, attempts

            attempt_record.applied = True
            attempts.append(attempt_record)

            should_run_tests = self.fix_config.run_tests and self.test_runner
            if should_run_tests:
                test_result = self._run_tests(test_command)
                attempt_record.test_result = test_result

                if test_result.success:
                    LOGGER.info("Tests passed! Fix successful on attempt %d", attempt_num)
                    return True, attempts

                LOGGER.warning("Tests failed on attempt %d", attempt_num)
                attempt_record.error_message = "Tests failed."

                if attempt_num < self.fix_config.max_attempts:
                    LOGGER.info("Attempting another fix after test failures")
                    continue

                LOGGER.error("Max attempts reached. Tests still failing.")
                return False, attempts

            LOGGER.info("Diff applied successfully without running tests")
            return True, attempts

        return False, attempts

    def propose_diff_simple(
        self, task_description: str, files: Iterable[str], extra_instructions: str = ""
    ) -> DiffProposal:
        """Simple diff proposal for backward compatibility."""
        return self.propose_diff(task_description, files, extra_instructions)

    def apply_diff(self, proposal: DiffProposal) -> None:
        """Apply diff for backward compatibility."""
        if proposal.fallback_reason:
            raise LLMError(
                "Cannot apply diff because the LLM fallback was activated. "
                + (proposal.fallback_guidance or proposal.fallback_reason or "")
            )
        if not self._apply_diff_with_approval(proposal):
            raise PermissionError("Code change not approved by user.")

    def search_and_replace(
        self,
        file_path: str,
        pattern: str,
        replacement: str,
        *,
        regex: bool = False,
        case_sensitive: bool = True,
        whole_word: bool = False,
    ) -> DiffProposal:
        """Perform a search-and-replace edit and return the resulting diff."""
        relative_path = Path(file_path)
        target_path = (self.repo_root / relative_path).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"File not found for search/replace: {relative_path}")

        original_text = target_path.read_text(encoding="utf-8")
        flags = 0 if case_sensitive else re.IGNORECASE

        if regex:
            expression = pattern
        else:
            expression = re.escape(pattern)
            if whole_word:
                expression = r"\b" + expression + r"\b"

        try:
            compiled = re.compile(expression, flags)
        except re.error as exc:
            raise ValueError(f"Invalid search pattern: {exc}") from exc

        new_text, count = compiled.subn(replacement, original_text)
        if count == 0 or new_text == original_text:
            raise ValueError(
                "Search pattern produced no changes; verify the pattern, case, and bounds."
            )

        diff_text = self._build_manual_diff(relative_path, original_text, new_text)
        preview = self.diff_processor.create_preview(diff_text)

        session = self._session_manager.ensure_session(
            self._session_id,
            system_messages=self._base_system_messages,
            metadata={
                "mode": "code-editor",
                "task": f"search-replace:{relative_path}",
            },
        )
        message = (
            "Manual search/replace request\n"
            f"File: {relative_path}\n"
            f"Pattern: {pattern}\n"
            f"Replacement: {replacement}\n"
            f"Options: regex={regex}, case_sensitive={case_sensitive}, whole_word={whole_word}\n"
            f"Matches: {count}"
        )
        self._session_manager.add_user_message(self._session_id, message)
        with session.lock:
            session.metadata["last_context_summary"] = {
                "style": preview.summary,
                "dependency": f"Manual search/replace ({count} matches)",
            }
        self._session_manager.add_assistant_message(
            self._session_id,
            f"Generated manual diff for {relative_path} with {count} replacements.",
        )

        return DiffProposal(
            diff=diff_text,
            raw_response=diff_text,
            files=[relative_path],
            preview=preview,
            validation_errors=list(preview.validation_result.errors),
        )

    def _apply_diff_with_approval(self, proposal: DiffProposal) -> bool:
        """Apply diff with approval check."""
        if proposal.fallback_reason:
            raise LLMError(
                "Diff proposal unavailable: " + (proposal.fallback_reason or "LLM fallback active")
            )
        if not self.approvals.require("code", default=True):
            return False

        try:
            success = self.diff_processor.apply_diff_safely(proposal.diff)
            if success:
                LOGGER.info(
                    "Diff applied for files: %s",
                    ", ".join(str(path) for path in proposal.files) or "<unknown>",
                )
            return success
        except DiffError as exc:
            raise LLMError(f"Failed to apply diff: {exc}") from exc

    def _run_tests(self, test_command: list[str] | None = None) -> TestResult:
        """Run tests and return results."""
        if not self.test_runner:
            raise RuntimeError("Test runner not configured")

        if test_command:
            return self.test_runner.run(test_command)
        else:
            return self.test_runner.run_pytest()

    def _build_file_blocks(self, contexts: list[FileContext]) -> str:
        """Build file blocks for the prompt."""
        return "\n".join(
            FILE_BLOCK_TEMPLATE.format(
                path=context.path.relative_to(self.repo_root),
                content=context.content,
                relevance=context.relevance_score,
                reason=context.reason,
            )
            for context in contexts
        )

    def _filter_code_contexts(self, contexts: list[FileContext]) -> list[FileContext]:
        """Filter out synthetic or empty contexts when deriving repository signals."""
        filtered: list[FileContext] = []
        for context in contexts:
            if (
                context.reason == "project_structure_summary"
                or context.path.name == "__project_structure__.md"
            ):
                continue
            if not context.content.strip():
                continue
            filtered.append(context)
        return filtered

    def _build_style_profile(self, contexts: list[FileContext]) -> str:
        """Infer indentation, quoting, and structural conventions from the provided files."""

        if not contexts:
            return "No dominant style detected from provided files; mirror local context."

        indent_counter: Counter[str] = Counter()
        space_widths: Counter[int] = Counter()
        quote_counter: Counter[str] = Counter()
        uses_type_hints = False
        uses_dataclasses = False
        uses_async = False

        for context in contexts:
            text = context.content
            uses_dataclasses = uses_dataclasses or "@dataclass" in text
            uses_type_hints = uses_type_hints or bool(re.search(r"def\s+\w+\(.*\)\s*->", text))
            uses_async = uses_async or "async def" in text

            quote_counter["double"] += text.count('"')
            quote_counter["single"] += text.count("'")

            for line in text.splitlines():
                stripped = line.lstrip(" \t")
                if not stripped:
                    continue
                prefix_length = len(line) - len(stripped)
                prefix = line[:prefix_length]
                if not prefix:
                    continue
                if "\t" in prefix:
                    indent_counter["tabs"] += 1
                else:
                    indent_counter["spaces"] += 1
                    space_widths[len(prefix)] += 1

        style_fragments: list[str] = []

        if indent_counter:
            tab_count = indent_counter.get("tabs", 0)
            space_count = indent_counter.get("spaces", 0)
            if tab_count > space_count * 1.5:
                style_fragments.append("Tab-indented codebase")
            elif space_count:
                common_width = max(space_widths, key=space_widths.get, default=4)
                style_fragments.append(f"{common_width}-space indentation")

        double_quotes = quote_counter.get("double", 0)
        single_quotes = quote_counter.get("single", 0)
        if double_quotes > single_quotes * 1.3:
            style_fragments.append("Prefers double quotes")
        elif single_quotes > double_quotes * 1.3:
            style_fragments.append("Prefers single quotes")

        if uses_type_hints:
            style_fragments.append("Type hints in function signatures")
        if uses_dataclasses:
            style_fragments.append("Dataclass-heavy modules")
        if uses_async:
            style_fragments.append("Async/await usage present")

        return (
            "; ".join(style_fragments)
            if style_fragments
            else "No dominant style detected from provided files; mirror local context."
        )

    def _build_dependency_summary(self, contexts: list[FileContext]) -> str:
        """Summarize external module usage for awareness of ripple effects."""

        dependency_map: dict[str, Counter[str]] = {}

        for context in contexts:
            try:
                rel_path = str(context.path.relative_to(self.repo_root))
            except ValueError:
                rel_path = str(context.path)

            counter = dependency_map.setdefault(rel_path, Counter())
            suffix = context.path.suffix.lower()

            for line in context.content.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue

                dependency: str | None = None
                if suffix in {".py", ".pyi"}:
                    if stripped.startswith("from ") and " import " in stripped:
                        dependency = stripped.split(" import ", 1)[0].replace("from", "", 1).strip()
                    elif stripped.startswith("import "):
                        dependency = stripped[len("import ") :].split(" as ", 1)[0].strip()
                elif suffix in {".js", ".jsx", ".ts", ".tsx"}:
                    if stripped.startswith("import "):
                        if " from " in stripped:
                            dependency = (
                                stripped.split(" from ", 1)[1].strip().rstrip(";").strip("'\"`")
                            )
                        else:
                            dependency = stripped[len("import ") :].strip().rstrip(";")
                    elif "require(" in stripped:
                        match = re.search(r"require\(([^)]+)\)", stripped)
                        if match:
                            dependency = match.group(1).strip("'\"`")
                elif suffix == ".go":
                    if stripped.startswith("import "):
                        dependency = stripped[len("import ") :].strip().strip('"')
                elif suffix == ".rs" and stripped.startswith("use "):
                    dependency = stripped[4:].split(" as ", 1)[0].rstrip(";")

                if dependency:
                    counter[dependency] += 1

        summaries: list[str] = []
        for rel_path, counter in dependency_map.items():
            if not counter:
                continue
            top_dependencies = ", ".join(dep for dep, _ in counter.most_common(3))
            summaries.append(f"{rel_path}: {top_dependencies}")

        if not summaries:
            return "No external dependencies detected in the provided snippets."

        return " | ".join(summaries[:4])

    def _identify_common_patterns(self, contexts: list[FileContext]) -> str:
        """Call out prevalent implementation patterns to guide the diff."""

        if not contexts:
            return "No dominant implementation patterns detected."

        patterns: set[str] = set()
        for context in contexts:
            text = context.content
            path_str = str(context.path)

            if "@dataclass" in text:
                patterns.add("dataclasses")
            if "pydantic" in text:
                patterns.add("pydantic models")
            if "logging.getLogger" in text or "LOGGER =" in text:
                patterns.add("centralized logging")
            if "async def" in text:
                patterns.add("async routines")
            if "pytest" in text or ("assert " in text and "tests" in path_str):
                patterns.add("pytest-style tests")
            if "click." in text:
                patterns.add("Click CLI patterns")
            if "Path(" in text:
                patterns.add("pathlib usage")
            if "typing." in text or "TypeVar" in text:
                patterns.add("typing utilities")

        return (
            "; ".join(sorted(patterns))
            if patterns
            else "No dominant implementation patterns detected."
        )

    def _build_quality_notes(
        self,
        contexts: list[FileContext],
        task_description: str | None,
        latest_attempt: FixAttempt | None,
    ) -> str:
        """Surface quality considerations, related tests, and recent failures."""

        notes: list[str] = []

        test_paths: list[str] = []
        for context in contexts:
            try:
                rel_path = context.path.relative_to(self.repo_root)
            except ValueError:
                rel_path = context.path
            rel_str = str(rel_path)
            if (
                "test" in rel_path.parts
                or rel_str.startswith("tests/")
                or rel_path.name.startswith("test_")
            ):
                test_paths.append(rel_str)

            if "TODO" in context.content or "FIXME" in context.content:
                notes.append(f"Outstanding TODO/FIXME in {rel_str}; verify if task addresses it.")

        if test_paths:
            display = ", ".join(test_paths[:3])
            if len(test_paths) > 3:
                display += ", …"
            notes.append(f"Relevant tests: {display}")

        if task_description:
            lowered = task_description.lower()
            if any(keyword in lowered for keyword in ("performance", "latency", "optimiz")):
                notes.append("Assess runtime impact; add benchmarks or profiling if necessary.")
            if any(keyword in lowered for keyword in ("security", "auth", "permission")):
                notes.append("Validate security invariants and sanitise inputs.")

        if latest_attempt and latest_attempt.test_result:
            result = latest_attempt.test_result
            try:
                command_preview = " ".join(result.command)
            except Exception:  # pragma: no cover - defensive
                command_preview = "tests"
            exit_code = getattr(result, "returncode", getattr(result, "exit_code", None))
            if exit_code and exit_code != 0:
                notes.append(f"Last test run `{command_preview}` failed (exit code {exit_code}).")

        return (
            " | ".join(notes)
            if notes
            else "No explicit quality blockers detected; still add tests when appropriate."
        )

    def _format_test_output(self, test_result: TestResult | None) -> str:
        """Format test output for inclusion in fix prompts."""
        if not test_result:
            return "No test results available"

        output_parts = []

        exit_code = getattr(test_result, "returncode", getattr(test_result, "exit_code", 0))
        if exit_code != 0:
            output_parts.append(f"Exit code: {exit_code}")

        if test_result.stderr:
            output_parts.append(f"STDERR:\n{test_result.stderr[:2000]}...")  # Limit length

        if test_result.stdout:
            output_parts.append(f"STDOUT:\n{test_result.stdout[:2000]}...")

        return "\n".join(output_parts) if output_parts else "Tests failed but no output captured"

    def _extract_files_from_diff(self, diff_text: str) -> list[Path]:
        """Extract file paths from diff."""
        files: list[Path] = []
        for line in diff_text.splitlines():
            if line.startswith("+++"):
                path = line[4:].strip()
                if path.startswith("b/"):
                    path = path[2:]
                if path != "/dev/null":
                    files.append(Path(path))
        return files

    def _build_fallback_guidance(
        self,
        task_description: str,
        contexts: list[FileContext],
        reason: str,
    ) -> str:
        if contexts:
            suggestions = []
            for context in contexts[:5]:
                try:
                    rel_path = context.path.relative_to(self.repo_root)
                except ValueError:
                    rel_path = context.path
                suggestions.append(f"- {rel_path}: {context.reason or 'Relevant code context'}")
            suggestion_block = "\n".join(suggestions)
            return (
                "LLM-generated diff unavailable; manual edits required.\n"
                f"Reason: {reason}\n"
                "Review these files and implement the task manually, then rerun with tests:\n"
                f"{suggestion_block}"
            )
        details = task_description.strip() or "No task description provided"
        return (
            "LLM-generated diff unavailable; manual edits required.\n"
            f"Reason: {reason}\n"
            f"Task context: {details}"
        )

    def _build_manual_diff(self, relative_path: Path, old_text: str, new_text: str) -> str:
        """Construct a unified diff for manual edits."""
        old_lines = old_text.splitlines()
        new_lines = new_text.splitlines()
        diff_lines = list(
            difflib.unified_diff(
                old_lines,
                new_lines,
                fromfile=f"a/{relative_path.as_posix()}",
                tofile=f"b/{relative_path.as_posix()}",
                lineterm="",
            )
        )
        if not diff_lines:
            raise ValueError("Search/replace did not produce any textual differences.")
        return "\n".join(diff_lines) + "\n"


__all__ = ["CodeEditor", "DiffProposal", "FixAttempt", "IterativeFixConfig"]
