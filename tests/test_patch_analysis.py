"""Tests for patch analysis utilities."""

from __future__ import annotations

from ai_dev_agent.tools.patch_analysis import PatchParser

SIMPLE_PATCH = """diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,4 @@
 def foo():
+    print("added")
     pass
"""

MULTI_HUNK_PATCH = """diff --git a/bar.py b/bar.py
--- a/bar.py
+++ b/bar.py
@@ -1,5 +1,6 @@
 def bar():
+    print("first addition")
     pass

 def baz():
@@ -10,3 +11,4 @@
     return 42

 def qux():
+    print("second addition")
     pass
"""

MULTIPLE_FILES_PATCH = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,2 +1,3 @@
 def one():
+    print("file1")
     pass
diff --git a/file2.js b/file2.js
--- a/file2.js
+++ b/file2.js
@@ -1,2 +1,3 @@
 function two() {
+    console.log("file2");
 }
"""

NEW_FILE_PATCH = """diff --git a/new.py b/new.py
new file mode 100644
--- /dev/null
+++ b/new.py
@@ -0,0 +1,3 @@
+def new_function():
+    return True
+
"""

DELETED_FILE_PATCH = """diff --git a/old.py b/old.py
deleted file mode 100644
--- a/old.py
+++ /dev/null
@@ -1,3 +0,0 @@
-def old_function():
-    return False
-
"""

RENAME_PATCH = """diff --git a/old_name.py b/new_name.py
similarity index 100%
rename from old_name.py
rename to new_name.py
--- a/old_name.py
+++ b/new_name.py
@@ -1,2 +1,3 @@
 def renamed():
+    print("renamed")
     pass
"""

GIT_FORMAT_PATCH = """From 2e573ddf5a0b895e5d904cad1536421b00e04db1 Mon Sep 17 00:00:00 2001
From: John Doe <john@example.com>
Date: Wed, 10 Jan 2024 10:00:00 +0000
Subject: Add new feature

This is a test commit.
---
 test.py | 3 ++-
 1 file changed, 2 insertions(+), 1 deletion(-)

diff --git a/test.py b/test.py
--- a/test.py
+++ b/test.py
@@ -1,3 +1,4 @@
 def test():
+    print("feature")
     pass
"""

COMPLEX_PATCH_WITH_REMOVALS = """diff --git a/complex.py b/complex.py
--- a/complex.py
+++ b/complex.py
@@ -1,10 +1,8 @@
 def complex():
-    old_line_1 = True
-    old_line_2 = False
+    new_line_1 = 42
     keep_this = "unchanged"
+    new_line_2 = 99
-    removed_line = None
     return True
"""


class TestPatchParser:
    """Test the PatchParser class."""

    def test_parse_simple_addition(self):
        """Test parsing a simple patch with one addition."""
        parser = PatchParser(SIMPLE_PATCH)
        result = parser.parse()

        assert result["files"]
        assert len(result["files"]) == 1

        file_entry = result["files"][0]
        assert file_entry["path"] == "foo.py"
        assert file_entry["change_type"] == "modified"
        assert file_entry["language"] == "python"
        assert file_entry["stats"]["additions"] == 1
        assert file_entry["stats"]["deletions"] == 0

        assert len(file_entry["hunks"]) == 1
        hunk = file_entry["hunks"][0]
        assert hunk["new_start"] == 1
        assert len(hunk["added_lines"]) == 1
        assert hunk["added_lines"][0]["content"] == '    print("added")'
        assert hunk["added_lines"][0]["line_number"] == 2

    def test_parse_multi_hunk(self):
        """Test parsing a patch with multiple hunks in one file."""
        parser = PatchParser(MULTI_HUNK_PATCH)
        result = parser.parse()

        assert len(result["files"]) == 1
        file_entry = result["files"][0]

        assert len(file_entry["hunks"]) == 2
        assert file_entry["stats"]["additions"] == 2

        # First hunk
        assert file_entry["hunks"][0]["added_lines"][0]["content"] == '    print("first addition")'

        # Second hunk
        assert file_entry["hunks"][1]["added_lines"][0]["content"] == '    print("second addition")'

    def test_parse_multiple_files(self):
        """Test parsing a patch with multiple files."""
        parser = PatchParser(MULTIPLE_FILES_PATCH)
        result = parser.parse()

        assert len(result["files"]) == 2

        # First file
        assert result["files"][0]["path"] == "file1.py"
        assert result["files"][0]["language"] == "python"
        assert result["files"][0]["stats"]["additions"] == 1

        # Second file
        assert result["files"][1]["path"] == "file2.js"
        assert result["files"][1]["language"] == "javascript"
        assert result["files"][1]["stats"]["additions"] == 1

    def test_parse_new_file(self):
        """Test parsing a patch that creates a new file."""
        parser = PatchParser(NEW_FILE_PATCH)
        result = parser.parse()

        assert len(result["files"]) == 1
        file_entry = result["files"][0]

        assert file_entry["path"] == "new.py"
        assert file_entry["change_type"] == "added"
        assert file_entry["stats"]["additions"] == 3
        assert len(file_entry["hunks"][0]["added_lines"]) == 3

    def test_parse_deleted_file(self):
        """Test parsing a patch that deletes a file."""
        parser = PatchParser(DELETED_FILE_PATCH)
        result = parser.parse()

        assert len(result["files"]) == 1
        file_entry = result["files"][0]

        assert file_entry["path"] == "old.py"
        assert file_entry["change_type"] == "deleted"
        assert file_entry["stats"]["deletions"] == 3
        assert len(file_entry["hunks"][0]["removed_lines"]) == 3

    def test_parse_renamed_file(self):
        """Test parsing a patch that renames a file."""
        parser = PatchParser(RENAME_PATCH)
        result = parser.parse()

        assert len(result["files"]) == 1
        file_entry = result["files"][0]

        assert file_entry["path"] == "new_name.py"
        assert file_entry["old_path"] == "old_name.py"
        assert file_entry["change_type"] == "renamed"

    def test_parse_git_format_patch(self):
        """Test parsing a git format-patch with commit metadata."""
        parser = PatchParser(GIT_FORMAT_PATCH)
        result = parser.parse()

        # Check commit metadata
        assert result["patch_info"]["commit"] == "2e573ddf5a0b895e5d904cad1536421b00e04db1"
        assert result["patch_info"]["author"] == "John Doe <john@example.com>"
        assert "Add new feature" in result["patch_info"]["message"]

        # Check file changes
        assert len(result["files"]) == 1
        assert result["files"][0]["path"] == "test.py"

    def test_parse_with_removals(self):
        """Test parsing a patch with both additions and deletions."""
        parser = PatchParser(COMPLEX_PATCH_WITH_REMOVALS)
        result = parser.parse()

        assert len(result["files"]) == 1
        file_entry = result["files"][0]

        hunk = file_entry["hunks"][0]
        assert len(hunk["added_lines"]) == 2
        assert len(hunk["removed_lines"]) == 3

        # Check specific additions
        added_contents = [line["content"] for line in hunk["added_lines"]]
        assert "    new_line_1 = 42" in added_contents
        assert "    new_line_2 = 99" in added_contents

    def test_filter_by_pattern(self):
        """Test filtering files by regex pattern."""
        parser = PatchParser(MULTIPLE_FILES_PATCH)

        # Filter for only .py files
        result = parser.parse(filter_pattern=r".*\.py$")
        assert len(result["files"]) == 1
        assert result["files"][0]["path"] == "file1.py"

        # Filter for only .js files
        parser2 = PatchParser(MULTIPLE_FILES_PATCH)
        result2 = parser2.parse(filter_pattern=r".*\.js$")
        assert len(result2["files"]) == 1
        assert result2["files"][0]["path"] == "file2.js"

    def test_include_context(self):
        """Test including context lines."""
        parser = PatchParser(SIMPLE_PATCH, include_context=True)
        result = parser.parse()

        hunk = result["files"][0]["hunks"][0]
        assert "context_lines" in hunk
        assert len(hunk["context_lines"]) >= 1

    def test_summary_statistics(self):
        """Test summary statistics calculation."""
        parser = PatchParser(MULTIPLE_FILES_PATCH)
        result = parser.parse()

        summary = result["summary"]
        assert summary["total_files"] == 2
        assert summary["files_modified"] == 2
        assert summary["total_additions"] == 2
        assert summary["total_deletions"] == 0

    def test_indentation_detection(self):
        """Test that indentation is correctly detected."""
        parser = PatchParser(SIMPLE_PATCH)
        result = parser.parse()

        added_line = result["files"][0]["hunks"][0]["added_lines"][0]
        assert added_line["indentation"] == "    "

    def test_language_detection(self):
        """Test language detection from file extensions."""
        test_cases = [
            ("test.py", "python"),
            ("test.js", "javascript"),
            ("test.ts", "typescript"),
            ("test.java", "java"),
            ("test.cpp", "cpp"),
            ("test.h", "c"),
            ("test.go", "go"),
            ("test.rs", "rust"),
            ("test.rb", "ruby"),
        ]

        for filename, expected_lang in test_cases:
            assert PatchParser._detect_language(filename) == expected_lang

    def test_empty_patch(self):
        """Test parsing an empty patch."""
        parser = PatchParser("")
        result = parser.parse()

        assert result["files"] == []
        assert result["summary"]["total_files"] == 0
        assert result["summary"]["total_additions"] == 0
        assert result["summary"]["total_deletions"] == 0

    def test_invalid_patch_content(self):
        """Test parsing content that is not a valid patch."""
        parser = PatchParser("This is not a patch")
        result = parser.parse()

        assert result["files"] == []
        assert result["summary"]["total_files"] == 0

    def test_line_number_tracking(self):
        """Test that line numbers are correctly tracked."""
        parser = PatchParser(MULTI_HUNK_PATCH)
        result = parser.parse()

        hunks = result["files"][0]["hunks"]

        # First hunk starts at line 1
        assert hunks[0]["new_start"] == 1
        assert hunks[0]["added_lines"][0]["line_number"] == 2

        # Second hunk starts at line 11
        assert hunks[1]["new_start"] == 11
