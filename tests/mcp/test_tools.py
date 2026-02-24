"""Tests for Axon Pro MCP tool handlers.

All tests mock the storage backend to avoid needing a real database.
Each tool handler is tested for both success and edge-case paths.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from axon_pro.core.graph.model import GraphNode, NodeLabel
from axon_pro.core.storage.base import SearchResult
from axon_pro.mcp.tools import (
    handle_context,
    handle_cypher,
    handle_dead_code,
    handle_detect_changes,
    handle_impact,
    handle_list_repos,
    handle_query,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_storage():
    """Create a mock storage backend with common default return values."""
    storage = MagicMock()
    storage.fts_search.return_value = [
        SearchResult(
            node_id="function:src/auth.py:validate",
            score=1.0,
            node_name="validate",
            file_path="src/auth.py",
            label="function",
            snippet="def validate(user): ...",
        ),
    ]
    storage.get_node.return_value = GraphNode(
        id="function:src/auth.py:validate",
        label=NodeLabel.FUNCTION,
        name="validate",
        file_path="src/auth.py",
        start_line=10,
        end_line=30,
    )
    storage.get_callers.return_value = []
    storage.get_callees.return_value = []
    storage.get_type_refs.return_value = []
    storage.vector_search.return_value = []
    storage.traverse.return_value = []
    storage.execute_raw.return_value = []
    return storage


@pytest.fixture
def mock_storage_with_relations(mock_storage):
    """Storage mock with callers, callees, and type refs populated."""
    mock_storage.get_callers.return_value = [
        GraphNode(
            id="function:src/routes/auth.py:login_handler",
            label=NodeLabel.FUNCTION,
            name="login_handler",
            file_path="src/routes/auth.py",
            start_line=12,
            end_line=40,
        ),
    ]
    mock_storage.get_callees.return_value = [
        GraphNode(
            id="function:src/auth/crypto.py:hash_password",
            label=NodeLabel.FUNCTION,
            name="hash_password",
            file_path="src/auth/crypto.py",
            start_line=5,
            end_line=20,
        ),
    ]
    mock_storage.get_type_refs.return_value = [
        GraphNode(
            id="class:src/models.py:User",
            label=NodeLabel.CLASS,
            name="User",
            file_path="src/models.py",
            start_line=1,
            end_line=50,
        ),
    ]
    return mock_storage


# ---------------------------------------------------------------------------
# 1. axon_list_repos
# ---------------------------------------------------------------------------


class TestHandleListRepos:
    def test_no_registry_dir(self, tmp_path):
        """Returns 'no repos' message when registry directory does not exist."""
        result = handle_list_repos(registry_dir=tmp_path / "nonexistent")
        assert "No indexed repositories found" in result

    def test_empty_registry_dir(self, tmp_path):
        """Returns 'no repos' message when registry directory is empty."""
        registry = tmp_path / "repos"
        registry.mkdir()
        result = handle_list_repos(registry_dir=registry)
        assert "No indexed repositories found" in result

    def test_with_repos(self, tmp_path):
        """Returns formatted repo list when meta.json files are present."""
        registry = tmp_path / "repos"
        repo_dir = registry / "my-project"
        repo_dir.mkdir(parents=True)
        meta = {
            "name": "my-project",
            "path": "/home/user/my-project",
            "stats": {
                "files": 25,
                "symbols": 150,
                "relationships": 200,
            },
        }
        (repo_dir / "meta.json").write_text(json.dumps(meta))

        result = handle_list_repos(registry_dir=registry)
        assert "my-project" in result
        assert "150" in result
        assert "200" in result
        assert "Indexed repositories (1)" in result


# ---------------------------------------------------------------------------
# 2. axon_query
# ---------------------------------------------------------------------------


class TestHandleQuery:
    def test_returns_results(self, mock_storage):
        """Successful query returns formatted results."""
        result = handle_query(mock_storage, "validate")
        assert "validate" in result
        assert "Function" in result
        assert "src/auth.py" in result
        assert "Next:" in result

    def test_no_results(self, mock_storage):
        """Empty search returns no-results message."""
        mock_storage.fts_search.return_value = []
        mock_storage.vector_search.return_value = []
        result = handle_query(mock_storage, "nonexistent")
        assert "No results found" in result

    def test_snippet_included(self, mock_storage):
        """Search results include snippet text."""
        result = handle_query(mock_storage, "validate")
        assert "def validate" in result

    def test_custom_limit(self, mock_storage):
        """Limit parameter is passed through to hybrid_search."""
        handle_query(mock_storage, "validate", limit=5)
        # hybrid_search calls fts_search with candidate_limit = limit * 3
        mock_storage.fts_search.assert_called_once_with("validate", limit=15)


# ---------------------------------------------------------------------------
# 3. axon_context
# ---------------------------------------------------------------------------


class TestHandleContext:
    def test_basic_context(self, mock_storage):
        """Returns symbol name, file, and line range."""
        result = handle_context(mock_storage, "validate")
        assert "Symbol: validate (Function)" in result
        assert "src/auth.py:10-30" in result
        assert "Next:" in result

    def test_not_found_fts_empty(self, mock_storage):
        """Returns not-found message when FTS returns nothing."""
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.return_value = []
        result = handle_context(mock_storage, "nonexistent")
        assert "not found" in result.lower()

    def test_not_found_node_none(self, mock_storage):
        """Returns not-found message when get_node returns None."""
        mock_storage.get_node.return_value = None
        result = handle_context(mock_storage, "validate")
        assert "not found" in result.lower()

    def test_with_callers_callees_type_refs(self, mock_storage_with_relations):
        """Full context includes callers, callees, and type refs."""
        result = handle_context(mock_storage_with_relations, "validate")
        assert "Callers (1):" in result
        assert "login_handler" in result
        assert "Callees (1):" in result
        assert "hash_password" in result
        assert "Type references (1):" in result
        assert "User" in result

    def test_dead_code_flag(self, mock_storage):
        """Dead code status is shown when is_dead is True."""
        mock_storage.get_node.return_value = GraphNode(
            id="function:src/old.py:deprecated",
            label=NodeLabel.FUNCTION,
            name="deprecated",
            file_path="src/old.py",
            start_line=1,
            end_line=5,
            is_dead=True,
        )
        result = handle_context(mock_storage, "deprecated")
        assert "DEAD CODE" in result


# ---------------------------------------------------------------------------
# 4. axon_impact
# ---------------------------------------------------------------------------


class TestHandleImpact:
    def test_no_downstream(self, mock_storage):
        """Returns no-dependencies message when traverse is empty."""
        result = handle_impact(mock_storage, "validate")
        assert "No upstream callers found" in result or "No downstream dependencies" in result

    def test_with_affected_symbols(self, mock_storage):
        """Returns formatted impact list when traverse finds nodes."""
        mock_storage.traverse.return_value = [
            GraphNode(
                id="function:src/api.py:login",
                label=NodeLabel.FUNCTION,
                name="login",
                file_path="src/api.py",
                start_line=5,
                end_line=20,
            ),
            GraphNode(
                id="function:src/api.py:register",
                label=NodeLabel.FUNCTION,
                name="register",
                file_path="src/api.py",
                start_line=25,
                end_line=50,
            ),
        ]
        result = handle_impact(mock_storage, "validate", depth=2)
        assert "Impact analysis for: validate" in result
        assert "Total affected symbols: 2" in result
        assert "login" in result
        assert "register" in result
        assert "Depth: 2" in result

    def test_symbol_not_found(self, mock_storage):
        """Returns not-found when symbol does not exist."""
        mock_storage.exact_name_search.return_value = []
        mock_storage.fts_search.return_value = []
        result = handle_impact(mock_storage, "nonexistent")
        assert "not found" in result.lower()


# ---------------------------------------------------------------------------
# 5. axon_dead_code
# ---------------------------------------------------------------------------


class TestHandleDeadCode:
    def test_no_dead_code(self, mock_storage):
        """Returns clean message when no dead code found."""
        result = handle_dead_code(mock_storage)
        assert "No dead code detected" in result

    def test_with_dead_code(self, mock_storage):
        """Returns formatted dead code list (delegates to get_dead_code_list)."""
        mock_storage.execute_raw.return_value = [
            ["unused_func", "src/old.py", 10],
            ["DeprecatedModel", "src/models.py", 5],
        ]
        result = handle_dead_code(mock_storage)
        assert "Dead Code Report (2 symbols)" in result
        assert "unused_func" in result
        assert "DeprecatedModel" in result

    def test_execute_raw_exception(self, mock_storage):
        """Gracefully handles storage errors."""
        mock_storage.execute_raw.side_effect = RuntimeError("DB error")
        result = handle_dead_code(mock_storage)
        assert "Could not retrieve dead code list" in result


# ---------------------------------------------------------------------------
# 6. axon_detect_changes
# ---------------------------------------------------------------------------


SAMPLE_DIFF = """\
diff --git a/src/auth.py b/src/auth.py
index abc1234..def5678 100644
--- a/src/auth.py
+++ b/src/auth.py
@@ -10,5 +10,7 @@ def validate(user):
     if not user:
         return False
+    # Added new validation
+    check_permissions(user)
     return True
"""


class TestHandleDetectChanges:
    def test_parses_diff(self, mock_storage):
        """Successfully parses diff and identifies changed files."""
        # handle_detect_changes now uses execute_raw() with a Cypher query
        # to find symbols in the changed file.
        mock_storage.execute_raw.return_value = [
            ["function:src/auth.py:validate", "validate", "src/auth.py", 10, 30],
        ]

        result = handle_detect_changes(mock_storage, SAMPLE_DIFF)
        assert "src/auth.py" in result
        assert "validate" in result
        assert "Total affected symbols:" in result

    def test_empty_diff(self, mock_storage):
        """Returns message for empty diff input."""
        result = handle_detect_changes(mock_storage, "")
        assert "Empty diff provided" in result

    def test_unparseable_diff(self, mock_storage):
        """Returns message when diff contains no recognisable hunks."""
        result = handle_detect_changes(mock_storage, "just some random text")
        assert "Could not parse" in result

    def test_no_symbols_in_changed_lines(self, mock_storage):
        """Reports file but no symbols when nothing overlaps."""
        mock_storage.execute_raw.return_value = []
        result = handle_detect_changes(mock_storage, SAMPLE_DIFF)
        assert "src/auth.py" in result
        assert "no indexed symbols" in result


# ---------------------------------------------------------------------------
# 7. axon_cypher
# ---------------------------------------------------------------------------


class TestHandleCypher:
    def test_returns_results(self, mock_storage):
        """Formats raw query results."""
        mock_storage.execute_raw.return_value = [
            ["validate", "src/auth.py", 10],
            ["login", "src/api.py", 5],
        ]
        result = handle_cypher(mock_storage, "MATCH (n) RETURN n.name, n.file_path, n.start_line")
        assert "Results (2 rows)" in result
        assert "validate" in result
        assert "src/api.py" in result

    def test_no_results(self, mock_storage):
        """Returns no-results message for empty query output."""
        result = handle_cypher(mock_storage, "MATCH (n:Nonexistent) RETURN n")
        assert "no results" in result.lower()

    def test_query_error(self, mock_storage):
        """Returns error message when query execution fails."""
        mock_storage.execute_raw.side_effect = RuntimeError("Syntax error")
        result = handle_cypher(mock_storage, "INVALID QUERY")
        assert "failed" in result.lower()
        assert "Syntax error" in result


# ---------------------------------------------------------------------------
# Resource handlers
# ---------------------------------------------------------------------------


class TestResources:
    def test_get_schema(self):
        """Schema resource returns static schema text."""
        from axon_pro.mcp.resources import get_schema

        result = get_schema()
        assert "Node Labels:" in result
        assert "Relationship Types:" in result
        assert "CALLS" in result
        assert "Function" in result

    def test_get_overview(self, mock_storage):
        """Overview resource queries storage for stats."""
        from axon_pro.mcp.resources import get_overview

        mock_storage.execute_raw.return_value = [["Function", 42]]
        result = get_overview(mock_storage)
        assert "Axon Pro Codebase Overview" in result

    def test_get_dead_code_list(self, mock_storage):
        """Dead code resource returns formatted report."""
        from axon_pro.mcp.resources import get_dead_code_list

        mock_storage.execute_raw.return_value = [
            ["old_func", "src/old.py", 10],
        ]
        result = get_dead_code_list(mock_storage)
        assert "Dead Code Report" in result
        assert "old_func" in result

    def test_get_dead_code_list_empty(self, mock_storage):
        """Dead code resource returns clean message when empty."""
        from axon_pro.mcp.resources import get_dead_code_list

        result = get_dead_code_list(mock_storage)
        assert "No dead code detected" in result
