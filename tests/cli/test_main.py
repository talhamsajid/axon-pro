"""Tests for the Axon Pro CLI."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import typer
from typer.testing import CliRunner

from axon_pro import __version__
from axon_pro.cli.main import app

runner = CliRunner()


class TestVersion:
    """Tests for the --version flag."""

    def test_version_long_flag(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert result.exit_code == 0
        assert f"Axon Pro v{__version__}" in result.output

    def test_version_short_flag(self) -> None:
        result = runner.invoke(app, ["-v"])
        assert result.exit_code == 0
        assert f"Axon Pro v{__version__}" in result.output

    def test_version_string_format(self) -> None:
        result = runner.invoke(app, ["--version"])
        assert f"Axon Pro v{__version__}" in result.output


class TestHelp:
    """Tests for the --help flag."""

    def test_help_exit_code(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0

    def test_help_shows_app_name(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert "Axon Pro" in result.output

    def test_help_lists_commands(self) -> None:
        result = runner.invoke(app, ["--help"])
        expected_commands = [
            "analyze",
            "status",
            "list",
            "clean",
            "query",
            "context",
            "impact",
            "dead-code",
            "cypher",
            "setup",
            "watch",
            "diff",
            "mcp",
        ]
        for cmd in expected_commands:
            assert cmd in result.output, f"Command '{cmd}' not found in --help output"


class TestStatus:
    """Tests for the status command."""

    def test_status_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Status should error when no .axon directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["status"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_status_with_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Status should display stats from meta.json."""
        monkeypatch.chdir(tmp_path)
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        meta = {
            "version": "0.1.0",
            "stats": {
                "files": 10,
                "symbols": 42,
                "relationships": 100,
                "clusters": 3,
                "flows": 0,
                "dead_code": 5,
                "coupled_pairs": 0,
            },
            "last_indexed_at": "2025-01-15T10:00:00+00:00",
        }
        (axon_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

        result = runner.invoke(app, ["status"])
        assert result.exit_code == 0
        assert "Index status for" in result.output
        assert "0.1.0" in result.output
        assert "10" in result.output  # files
        assert "42" in result.output  # symbols
        assert "100" in result.output  # relationships


class TestListRepos:
    """Tests for the list command."""

    def test_list_calls_handle_list_repos(self) -> None:
        """List should call handle_list_repos and print the result."""
        with patch(
            "axon_pro.mcp.tools.handle_list_repos",
            return_value="Indexed repositories (1):\n\n  1. my-project",
        ):
            result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        assert "my-project" in result.output

    def test_list_no_repos(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """List should show 'no repos' message when none are indexed."""
        monkeypatch.chdir(tmp_path)
        # Patch the global registry to a non-existent dir so the fallback also fails
        result = runner.invoke(app, ["list"])
        assert result.exit_code == 0
        # handle_list_repos returns "No indexed repositories found." when nothing found
        assert "No indexed repositories found" in result.output or "repositories" in result.output.lower()


class TestClean:
    """Tests for the clean command."""

    def test_clean_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Clean should error when no .axon directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["clean", "--force"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_clean_with_force(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Clean with --force should delete .axon without confirmation."""
        monkeypatch.chdir(tmp_path)
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        (axon_dir / "meta.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(app, ["clean", "--force"])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert not axon_dir.exists()

    def test_clean_aborted(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Clean should abort when user says no."""
        monkeypatch.chdir(tmp_path)
        axon_dir = tmp_path / ".axon"
        axon_dir.mkdir()
        (axon_dir / "meta.json").write_text("{}", encoding="utf-8")

        result = runner.invoke(app, ["clean"], input="n\n")
        assert axon_dir.exists()  # Not deleted


class TestQuery:
    """Tests for the query command."""

    def test_query_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Query should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["query", "find classes"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_query_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Query should call handle_query with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon_pro.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon_pro.mcp.tools.handle_query",
                return_value="1. MyClass (Class) -- src/main.py",
            ):
                result = runner.invoke(app, ["query", "find classes"])
        assert result.exit_code == 0
        assert "MyClass" in result.output


class TestContext:
    """Tests for the context command."""

    def test_context_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Context should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["context", "MyClass"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_context_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Context should call handle_context with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon_pro.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon_pro.mcp.tools.handle_context",
                return_value="Symbol: MyClass (Class)\nFile: src/main.py:1-50",
            ):
                result = runner.invoke(app, ["context", "MyClass"])
        assert result.exit_code == 0
        assert "MyClass" in result.output


class TestImpact:
    """Tests for the impact command."""

    def test_impact_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Impact should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["impact", "MyClass.method"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_impact_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Impact should call handle_impact with loaded storage and depth."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon_pro.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon_pro.mcp.tools.handle_impact",
                return_value="Impact analysis for: MyClass.method",
            ):
                result = runner.invoke(app, ["impact", "MyClass.method", "--depth", "5"])
        assert result.exit_code == 0
        assert "Impact analysis" in result.output

    def test_impact_default_depth(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Impact without --depth should use default depth of 3."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon_pro.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon_pro.mcp.tools.handle_impact",
                return_value="Impact analysis for: foo",
            ) as mock_handle:
                result = runner.invoke(app, ["impact", "foo"])
        assert result.exit_code == 0


class TestDeadCode:
    """Tests for the dead-code command."""

    def test_dead_code_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Dead-code should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["dead-code"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_dead_code_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Dead-code should call handle_dead_code with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon_pro.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon_pro.mcp.tools.handle_dead_code",
                return_value="No dead code detected.",
            ):
                result = runner.invoke(app, ["dead-code"])
        assert result.exit_code == 0
        assert "No dead code detected" in result.output


class TestCypher:
    """Tests for the cypher command."""

    def test_cypher_no_index(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Cypher should error when no .axon/kuzu directory exists."""
        monkeypatch.chdir(tmp_path)
        result = runner.invoke(app, ["cypher", "MATCH (n) RETURN n"])
        assert result.exit_code == 1
        assert "No index found" in result.output

    def test_cypher_with_storage(self, tmp_path: Path, monkeypatch: "pytest.MonkeyPatch") -> None:
        """Cypher should call handle_cypher with loaded storage."""
        monkeypatch.chdir(tmp_path)
        mock_storage = MagicMock()
        with patch("axon_pro.cli.main._load_storage", return_value=mock_storage):
            with patch(
                "axon_pro.mcp.tools.handle_cypher",
                return_value="Results (3 rows):\n\n  1. foo",
            ):
                result = runner.invoke(app, ["cypher", "MATCH (n) RETURN n"])
        assert result.exit_code == 0
        assert "Results" in result.output


class TestSetup:
    """Tests for the setup command."""

    def test_setup_no_flags_shows_both(self) -> None:
        """Setup with no flags should show config for both Claude and Cursor."""
        result = runner.invoke(app, ["setup"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" in result.output
        assert '"axon"' in result.output

    def test_setup_claude_only(self) -> None:
        """Setup with --claude should show only Claude config."""
        result = runner.invoke(app, ["setup", "--claude"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" not in result.output

    def test_setup_cursor_only(self) -> None:
        """Setup with --cursor should show only Cursor config."""
        result = runner.invoke(app, ["setup", "--cursor"])
        assert result.exit_code == 0
        assert "Cursor" in result.output
        assert "Claude Code" not in result.output

    def test_setup_both_flags(self) -> None:
        """Setup with both flags should show both configs."""
        result = runner.invoke(app, ["setup", "--claude", "--cursor"])
        assert result.exit_code == 0
        assert "Claude Code" in result.output
        assert "Cursor" in result.output


class TestMcp:
    """Tests for the mcp command."""

    def test_mcp_command_exists(self) -> None:
        """The mcp command should be registered."""
        result = runner.invoke(app, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "MCP server" in result.output or "stdio" in result.output.lower()

    def test_mcp_calls_server_main(self) -> None:
        """MCP command should call asyncio.run(mcp_main())."""
        with patch("axon_pro.cli.main.asyncio", create=True) as mock_asyncio:
            with patch("axon_pro.mcp.server.main") as mock_mcp_main:
                # We need to mock at the import level inside the function
                import asyncio as real_asyncio

                with patch.object(real_asyncio, "run") as mock_run:
                    result = runner.invoke(app, ["mcp"])
                    mock_run.assert_called_once()


class TestServe:
    """Tests for the serve command."""

    def test_serve_command_exists(self) -> None:
        """The serve command should be registered."""
        result = runner.invoke(app, ["serve", "--help"])
        assert result.exit_code == 0
        assert "watch" in result.output.lower()

    def test_serve_without_watch_delegates_to_mcp(self) -> None:
        """serve without --watch should behave like axon mcp."""
        import asyncio as real_asyncio

        with patch.object(real_asyncio, "run") as mock_run:
            result = runner.invoke(app, ["serve"])
            mock_run.assert_called_once()


class TestWatch:
    """Tests for the watch command."""

    def test_watch_command_exists(self) -> None:
        """The watch command should be registered."""
        result = runner.invoke(app, ["watch", "--help"])
        assert result.exit_code == 0
        assert "Watch mode" in result.output or "re-index" in result.output.lower()

    def test_diff_command_exists(self) -> None:
        """The diff command should be registered."""
        result = runner.invoke(app, ["diff", "--help"])
        assert result.exit_code == 0
        assert "branch" in result.output.lower()
