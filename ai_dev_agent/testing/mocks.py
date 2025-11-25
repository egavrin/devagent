"""Mock objects and utilities for DevAgent testing.

This module provides mock implementations of external dependencies
to enable isolated unit testing.
"""

import asyncio
from typing import Any, Optional, Union
from unittest.mock import Mock

from ai_dev_agent.core.utils.constants import LLM_DEFAULT_TEMPERATURE


class MockLLM:
    """Mock LLM implementation for testing without API calls."""

    def __init__(
        self, default_response: str = "Mock LLM response", fail_after: Optional[int] = None
    ):
        """Initialize mock LLM.

        Args:
            default_response: Default response text
            fail_after: Fail after N calls (for testing error handling)
        """
        self.default_response = default_response
        self.fail_after = fail_after
        self.call_count = 0
        self.history = []

    def complete(
        self,
        prompt: str,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        max_tokens: int = 2000,
        **kwargs,
    ) -> dict:
        """Mock completion endpoint.

        Args:
            prompt: Input prompt
            temperature: Temperature setting
            max_tokens: Maximum tokens
            **kwargs: Additional parameters

        Returns:
            Mock completion response
        """
        self.call_count += 1
        self.history.append({"prompt": prompt, "kwargs": kwargs})

        if self.fail_after and self.call_count > self.fail_after:
            raise Exception("Mock LLM failure")

        # Generate response based on prompt content
        response = self._generate_response(prompt)

        return {
            "content": response,
            "model": "mock-gpt-4",
            "usage": {
                "prompt_tokens": len(prompt.split()),
                "completion_tokens": len(response.split()),
                "total_tokens": len(prompt.split()) + len(response.split()),
            },
            "finish_reason": "stop",
        }

    def _generate_response(self, prompt: str) -> str:
        """Generate contextual mock response based on prompt.

        Args:
            prompt: Input prompt

        Returns:
            Contextual mock response
        """
        prompt_lower = prompt.lower()

        if "test" in prompt_lower:
            return "def test_example():\n    assert True"
        elif "implement" in prompt_lower:
            return "def implement_feature():\n    return 'implemented'"
        elif "review" in prompt_lower:
            return "Code review: Looks good, minor suggestions for improvement."
        elif "plan" in prompt_lower:
            return "1. Analyze requirements\n2. Design solution\n3. Implement\n4. Test"
        elif "error" in prompt_lower or "fix" in prompt_lower:
            return "Fixed the error by updating the logic."
        else:
            return self.default_response

    async def aComplete(self, prompt: str, **kwargs) -> dict:
        """Async version of complete method."""
        await asyncio.sleep(0.01)  # Simulate network delay
        return self.complete(prompt, **kwargs)

    def stream_complete(self, prompt: str, **kwargs):
        """Mock streaming completion.

        Yields:
            Chunks of the response
        """
        response = self._generate_response(prompt)
        words = response.split()

        for word in words:
            yield {"delta": word + " ", "finish_reason": None}

        yield {"delta": "", "finish_reason": "stop"}


class MockFileSystem:
    """Mock file system for testing file operations."""

    def __init__(self):
        """Initialize mock file system."""
        self.files = {}
        self.directories = {"/", "/src", "/tests", "/docs"}

    def read(self, path: str) -> str:
        """Mock file read operation.

        Args:
            path: File path

        Returns:
            File content

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if path not in self.files:
            raise FileNotFoundError(f"Mock file not found: {path}")
        return self.files[path]

    def write(self, path: str, content: str) -> None:
        """Mock file write operation.

        Args:
            path: File path
            content: Content to write
        """
        # Add parent directories
        parts = path.split("/")
        for i in range(1, len(parts)):
            dir_path = "/".join(parts[:i])
            if dir_path:
                self.directories.add(dir_path)

        self.files[path] = content

    def exists(self, path: str) -> bool:
        """Check if file exists.

        Args:
            path: File path

        Returns:
            True if file exists
        """
        return path in self.files or path in self.directories

    def list_dir(self, path: str) -> list[str]:
        """List directory contents.

        Args:
            path: Directory path

        Returns:
            List of files and directories
        """
        if path not in self.directories:
            raise NotADirectoryError(f"Not a directory: {path}")

        items = []
        path = path.rstrip("/")

        # Find direct children
        for file_path in self.files:
            if file_path.startswith(path + "/"):
                relative = file_path[len(path) + 1 :]
                if "/" not in relative:
                    items.append(relative)

        for dir_path in self.directories:
            if dir_path.startswith(path + "/"):
                relative = dir_path[len(path) + 1 :]
                if "/" not in relative:
                    items.append(relative + "/")

        return sorted(items)

    def delete(self, path: str) -> None:
        """Delete a file.

        Args:
            path: File path
        """
        if path in self.files:
            del self.files[path]
        else:
            raise FileNotFoundError(f"File not found: {path}")


class MockGitRepo:
    """Mock git repository for testing git operations."""

    def __init__(self):
        """Initialize mock git repo."""
        self.commits = []
        self.branches = {"main": True, "develop": False}
        self.current_branch = "main"
        self.staged = set()
        self.modified = set()

    def status(self) -> dict:
        """Get repository status.

        Returns:
            Status dictionary
        """
        return {
            "branch": self.current_branch,
            "staged": list(self.staged),
            "modified": list(self.modified),
            "untracked": [],
        }

    def add(self, files: Union[str, list[str]]) -> None:
        """Stage files for commit.

        Args:
            files: File(s) to stage
        """
        if isinstance(files, str):
            files = [files]
        self.staged.update(files)

    def commit(self, message: str) -> str:
        """Create a commit.

        Args:
            message: Commit message

        Returns:
            Commit hash
        """
        import hashlib

        commit_hash = hashlib.sha1(message.encode()).hexdigest()[:7]

        self.commits.append({"hash": commit_hash, "message": message, "files": list(self.staged)})

        self.staged.clear()
        return commit_hash

    def checkout(self, branch: str, create: bool = False) -> None:
        """Checkout a branch.

        Args:
            branch: Branch name
            create: Create new branch if True
        """
        if create:
            self.branches[branch] = False
        elif branch not in self.branches:
            raise ValueError(f"Branch not found: {branch}")

        self.current_branch = branch

    def log(self, n: int = 10) -> list[dict]:
        """Get commit log.

        Args:
            n: Number of commits to return

        Returns:
            List of commits
        """
        return self.commits[-n:]


class MockHTTPClient:
    """Mock HTTP client for testing web operations."""

    def __init__(self):
        """Initialize mock HTTP client."""
        self.responses = {}
        self.history = []

    def set_response(self, url: str, response: dict) -> None:
        """Set mock response for URL.

        Args:
            url: URL pattern
            response: Response to return
        """
        self.responses[url] = response

    def get(self, url: str, **kwargs) -> dict:
        """Mock GET request.

        Args:
            url: Request URL
            **kwargs: Additional parameters

        Returns:
            Mock response
        """
        self.history.append({"method": "GET", "url": url, "kwargs": kwargs})

        for pattern, response in self.responses.items():
            if pattern in url:
                return response

        return {"status": 404, "content": "Not found", "headers": {}}

    def post(self, url: str, data: Any = None, **kwargs) -> dict:
        """Mock POST request.

        Args:
            url: Request URL
            data: Request data
            **kwargs: Additional parameters

        Returns:
            Mock response
        """
        self.history.append({"method": "POST", "url": url, "data": data, "kwargs": kwargs})

        for pattern, response in self.responses.items():
            if pattern in url:
                return response

        return {"status": 200, "content": {"message": "Success"}, "headers": {}}


class MockDatabase:
    """Mock database for testing data operations."""

    def __init__(self):
        """Initialize mock database."""
        self.tables = {}
        self.query_count = 0

    def create_table(self, name: str, schema: dict) -> None:
        """Create a table.

        Args:
            name: Table name
            schema: Table schema
        """
        self.tables[name] = {"schema": schema, "data": []}

    def insert(self, table: str, record: dict) -> int:
        """Insert a record.

        Args:
            table: Table name
            record: Record to insert

        Returns:
            Record ID
        """
        if table not in self.tables:
            raise ValueError(f"Table not found: {table}")

        self.query_count += 1
        record_id = len(self.tables[table]["data"])
        record["id"] = record_id
        self.tables[table]["data"].append(record)
        return record_id

    def select(self, table: str, where: Optional[dict] = None) -> list[dict]:
        """Select records.

        Args:
            table: Table name
            where: Filter conditions

        Returns:
            List of matching records
        """
        if table not in self.tables:
            raise ValueError(f"Table not found: {table}")

        self.query_count += 1
        data = self.tables[table]["data"]

        if not where:
            return data

        # Simple filtering
        results = []
        for record in data:
            match = all(record.get(k) == v for k, v in where.items())
            if match:
                results.append(record)

        return results

    def update(self, table: str, record_id: int, updates: dict) -> bool:
        """Update a record.

        Args:
            table: Table name
            record_id: Record ID
            updates: Fields to update

        Returns:
            True if updated
        """
        if table not in self.tables:
            raise ValueError(f"Table not found: {table}")

        self.query_count += 1
        for record in self.tables[table]["data"]:
            if record.get("id") == record_id:
                record.update(updates)
                return True

        return False

    def delete(self, table: str, record_id: int) -> bool:
        """Delete a record.

        Args:
            table: Table name
            record_id: Record ID

        Returns:
            True if deleted
        """
        if table not in self.tables:
            raise ValueError(f"Table not found: {table}")

        self.query_count += 1
        original_len = len(self.tables[table]["data"])
        self.tables[table]["data"] = [
            r for r in self.tables[table]["data"] if r.get("id") != record_id
        ]
        return len(self.tables[table]["data"]) < original_len


class MockCache:
    """Mock cache for testing caching operations."""

    def __init__(self):
        """Initialize mock cache."""
        self.data = {}
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        if key in self.data:
            self.hits += 1
            return self.data[key]
        else:
            self.misses += 1
            return None

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (ignored in mock)
        """
        self.data[key] = value

    def delete(self, key: str) -> bool:
        """Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted
        """
        if key in self.data:
            del self.data[key]
            return True
        return False

    def clear(self) -> None:
        """Clear all cache entries."""
        self.data.clear()

    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate.

        Returns:
            Hit rate percentage
        """
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return (self.hits / total) * 100


# Factory functions for creating configured mocks
def create_mock_agent(name: str, agent_type: str) -> Mock:
    """Create a mock agent.

    Args:
        name: Agent name
        agent_type: Agent type

    Returns:
        Configured mock agent
    """
    agent = Mock()
    agent.name = name
    agent.type = agent_type
    agent.status = "ready"
    agent.execute = Mock(return_value={"status": "success", "result": f"Task completed by {name}"})
    return agent


def create_mock_tool(name: str, description: str) -> Mock:
    """Create a mock tool.

    Args:
        name: Tool name
        description: Tool description

    Returns:
        Configured mock tool
    """
    tool = Mock()
    tool.name = name
    tool.description = description
    tool.execute = Mock(return_value={"status": "success", "output": f"Tool {name} executed"})
    return tool
