"""Tests for the artifacts utility module."""

import hashlib
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import tempfile
import shutil

from ai_dev_agent.core.utils.artifacts import write_artifact, ARTIFACTS_ROOT


class TestWriteArtifact:
    """Tests for the write_artifact function."""

    def setup_method(self):
        """Set up a temporary directory for testing."""
        self.temp_dir = tempfile.mkdtemp()
        self.temp_path = Path(self.temp_dir)

    def teardown_method(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_write_artifact_basic(self):
        """Test basic artifact writing."""
        content = "Test content for artifact"

        # Write artifact to temp directory
        artifact_path = write_artifact(content, root=self.temp_path)

        # Check file was created
        assert artifact_path.exists()
        assert artifact_path.parent == self.temp_path / ARTIFACTS_ROOT

        # Check content was written correctly
        written_content = artifact_path.read_text(encoding="utf-8")
        assert written_content == content

        # Check filename format
        assert artifact_path.name.startswith("artifact_")
        assert artifact_path.suffix == ".txt"

    def test_write_artifact_custom_suffix(self):
        """Test writing artifact with custom suffix."""
        content = '{"key": "value"}'

        artifact_path = write_artifact(content, suffix=".json", root=self.temp_path)

        assert artifact_path.suffix == ".json"
        assert artifact_path.read_text(encoding="utf-8") == content

    def test_write_artifact_creates_directory(self):
        """Test that write_artifact creates the artifacts directory if it doesn't exist."""
        # Use a new temp directory that definitely doesn't have the artifacts dir
        new_temp = Path(tempfile.mkdtemp())
        try:
            artifacts_dir = new_temp / ARTIFACTS_ROOT
            assert not artifacts_dir.exists()

            write_artifact("test", root=new_temp)

            assert artifacts_dir.exists()
            assert artifacts_dir.is_dir()
        finally:
            shutil.rmtree(new_temp, ignore_errors=True)

    def test_write_artifact_unicode_handling(self):
        """Test writing artifact with unicode content."""
        content = "Unicode test: 你好世界 🌍 émojis"

        artifact_path = write_artifact(content, root=self.temp_path)

        written_content = artifact_path.read_text(encoding="utf-8")
        assert written_content == content

    def test_write_artifact_encoding_errors(self):
        """Test handling of encoding errors with replacement."""
        # Create content with invalid unicode sequences
        content = "Test \udcff invalid unicode"

        # Should not raise an error, will use 'replace' error handling
        artifact_path = write_artifact(content, root=self.temp_path)

        assert artifact_path.exists()
        # The invalid character should be replaced
        written_bytes = artifact_path.read_bytes()
        assert len(written_bytes) > 0

    @patch('ai_dev_agent.core.utils.artifacts.datetime')
    def test_write_artifact_filename_format(self, mock_datetime):
        """Test the filename format with mocked datetime."""
        # Mock datetime to have consistent timestamp
        mock_now = MagicMock()
        mock_now.strftime.return_value = "20240101_120000"
        mock_datetime.utcnow.return_value = mock_now

        content = "Test content"
        artifact_path = write_artifact(content, root=self.temp_path)

        # Calculate expected hash
        data = content.encode("utf-8", errors="replace")
        expected_hash = hashlib.sha1(data).hexdigest()[:12]

        expected_filename = f"artifact_20240101_120000_{expected_hash}.txt"
        assert artifact_path.name == expected_filename

    def test_write_artifact_same_content_different_files(self):
        """Test that same content creates different files due to timestamp."""
        content = "Same content"

        # Write twice with a small delay
        path1 = write_artifact(content, root=self.temp_path)

        # Mock different timestamp for second write
        with patch('ai_dev_agent.core.utils.artifacts.datetime') as mock_dt:
            mock_now = MagicMock()
            mock_now.strftime.return_value = "20240101_120001"  # Different timestamp
            mock_dt.utcnow.return_value = mock_now

            path2 = write_artifact(content, root=self.temp_path)

        # Should be different files despite same content
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()
        assert path1.read_text() == path2.read_text()

    def test_write_artifact_no_root_specified(self):
        """Test writing artifact without specifying root (uses cwd)."""
        # Change to temp directory
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(self.temp_dir)

            content = "Test without root"
            artifact_path = write_artifact(content)

            # Should be under current directory
            assert artifact_path.exists()
            expected_parent = Path.cwd() / ARTIFACTS_ROOT
            assert artifact_path.parent == expected_parent

        finally:
            os.chdir(original_cwd)

    def test_write_artifact_empty_content(self):
        """Test writing empty content."""
        content = ""

        artifact_path = write_artifact(content, root=self.temp_path)

        assert artifact_path.exists()
        assert artifact_path.read_text() == ""

        # Even empty content gets a hash
        assert "_" in artifact_path.name

    def test_write_artifact_large_content(self):
        """Test writing large content."""
        # Create 10MB of content
        content = "x" * (10 * 1024 * 1024)

        artifact_path = write_artifact(content, root=self.temp_path)

        assert artifact_path.exists()
        assert artifact_path.stat().st_size == len(content)

    def test_write_artifact_path_traversal_safety(self):
        """Test that the function is safe from path traversal attacks."""
        # Try to use path traversal in suffix (should be safe)
        content = "Test"

        # Suffix with path traversal attempt
        malicious_suffix = "/../../../evil.txt"

        artifact_path = write_artifact(content, suffix=malicious_suffix, root=self.temp_path)

        # Should still be under artifacts directory, not escaped
        assert str(self.temp_path / ARTIFACTS_ROOT) in str(artifact_path.parent)

    def test_artifacts_root_constant(self):
        """Test the ARTIFACTS_ROOT constant."""
        assert ARTIFACTS_ROOT == Path(".devagent") / "artifacts"
        assert isinstance(ARTIFACTS_ROOT, Path)