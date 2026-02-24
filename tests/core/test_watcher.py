"""Tests for the watch mode module (watcher.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from axon_pro.core.ingestion.pipeline import reindex_files, run_pipeline
from axon_pro.core.ingestion.watcher import _reindex_files
from axon_pro.core.ingestion.walker import FileEntry, read_file
from axon_pro.core.storage.kuzu_backend import KuzuBackend


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_repo(tmp_path: Path) -> Path:
    """Create a small Python repository for watcher tests."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "app.py").write_text(
        "def hello():\n"
        "    return 'hello'\n",
        encoding="utf-8",
    )

    (src / "utils.py").write_text(
        "def helper():\n"
        "    pass\n",
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture()
def storage(tmp_path: Path) -> KuzuBackend:
    """Provide an initialised KuzuBackend for testing."""
    db_path = tmp_path / "test_db"
    backend = KuzuBackend()
    backend.initialize(db_path)
    yield backend
    backend.close()


# ---------------------------------------------------------------------------
# Tests: _read_file_entry
# ---------------------------------------------------------------------------


class TestReadFileEntry:
    """_read_file_entry reads a file and returns a FileEntry."""

    def test_reads_python_file(self, tmp_repo: Path) -> None:
        entry = read_file(tmp_repo, tmp_repo / "src" / "app.py")

        assert entry is not None
        assert entry.path == "src/app.py"
        assert entry.language == "python"
        assert "hello" in entry.content

    def test_returns_none_for_unsupported(self, tmp_repo: Path) -> None:
        readme = tmp_repo / "README.md"
        readme.write_text("# readme", encoding="utf-8")

        entry = read_file(tmp_repo, readme)

        assert entry is None

    def test_returns_none_for_missing(self, tmp_repo: Path) -> None:
        entry = read_file(tmp_repo, tmp_repo / "nonexistent.py")

        assert entry is None

    def test_returns_none_for_empty(self, tmp_repo: Path) -> None:
        empty = tmp_repo / "empty.py"
        empty.write_text("", encoding="utf-8")

        entry = read_file(tmp_repo, empty)

        assert entry is None


# ---------------------------------------------------------------------------
# Tests: reindex_files (pipeline function)
# ---------------------------------------------------------------------------


class TestReindexFiles:
    """reindex_files() correctly removes old nodes and adds new ones."""

    def test_reindex_updates_content(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)

        # Verify initial node exists.
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "hello" in node.content

        # Modify the file.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n"
            "    return 'goodbye'\n",
            encoding="utf-8",
        )

        # Re-read and reindex.
        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        reindex_files([entry], tmp_repo, storage)

        # Verify updated node.
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "goodbye" in node.content

    def test_reindex_handles_new_symbols(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)

        # Add a new function to the file.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n"
            "    return 'hello'\n"
            "\n"
            "def world():\n"
            "    return 'world'\n",
            encoding="utf-8",
        )

        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        reindex_files([entry], tmp_repo, storage)

        # Both symbols should exist.
        assert storage.get_node("function:src/app.py:hello") is not None
        assert storage.get_node("function:src/app.py:world") is not None

    def test_reindex_removes_deleted_symbols(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        # Initial full index.
        run_pipeline(tmp_repo, storage)
        assert storage.get_node("function:src/app.py:hello") is not None

        # Remove the function.
        (tmp_repo / "src" / "app.py").write_text(
            "# empty file\nX = 1\n",
            encoding="utf-8",
        )

        entry = FileEntry(
            path="src/app.py",
            content=(tmp_repo / "src" / "app.py").read_text(),
            language="python",
        )
        reindex_files([entry], tmp_repo, storage)

        # Old symbol should be gone.
        assert storage.get_node("function:src/app.py:hello") is None


# ---------------------------------------------------------------------------
# Tests: _reindex_files (watcher helper)
# ---------------------------------------------------------------------------


class TestWatcherReindexFiles:
    """_reindex_files filters and processes changed paths."""

    def test_reindexes_changed_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Modify a file.
        app_path = tmp_repo / "src" / "app.py"
        app_path.write_text(
            "def hello():\n    return 'updated'\n",
            encoding="utf-8",
        )

        count = _reindex_files([app_path], tmp_repo, storage)

        assert count == 1
        node = storage.get_node("function:src/app.py:hello")
        assert node is not None
        assert "updated" in node.content

    def test_skips_ignored_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Create a file in an ignored directory.
        cache_dir = tmp_repo / "__pycache__"
        cache_dir.mkdir()
        cached = cache_dir / "module.cpython-311.pyc"
        cached.write_bytes(b"\x00")

        count = _reindex_files([cached], tmp_repo, storage)

        assert count == 0

    def test_skips_unsupported_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        readme = tmp_repo / "README.md"
        readme.write_text("# hello", encoding="utf-8")

        count = _reindex_files([readme], tmp_repo, storage)

        assert count == 0

    def test_handles_deleted_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # File exists in storage but is now deleted from disk.
        deleted_path = tmp_repo / "src" / "app.py"
        assert storage.get_node("file:src/app.py:") is not None

        deleted_path.unlink()

        count = _reindex_files([deleted_path], tmp_repo, storage)

        # Returns 0 because file no longer exists (was handled as deletion).
        assert count == 0

    def test_handles_multiple_files(
        self, tmp_repo: Path, storage: KuzuBackend
    ) -> None:
        run_pipeline(tmp_repo, storage)

        # Modify both files.
        (tmp_repo / "src" / "app.py").write_text(
            "def hello():\n    return 'v2'\n",
            encoding="utf-8",
        )
        (tmp_repo / "src" / "utils.py").write_text(
            "def helper():\n    return 42\n",
            encoding="utf-8",
        )

        count = _reindex_files(
            [tmp_repo / "src" / "app.py", tmp_repo / "src" / "utils.py"],
            tmp_repo,
            storage,
        )

        assert count == 2
