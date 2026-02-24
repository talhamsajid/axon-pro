"""Axon Pro CLI — Graph-powered code intelligence engine."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from axon_pro import __version__

console = Console()

def _load_storage(repo_path: Path | None = None) -> "KuzuBackend":  # noqa: F821
    """Load the KuzuDB backend for the given or current repo."""
    from axon_pro.core.storage.kuzu_backend import KuzuBackend

    target = (repo_path or Path.cwd()).resolve()
    db_path = target / ".axon-pro" / "kuzu"
    if not db_path.exists():
        console.print(
            f"[red]Error:[/red] No index found at {target}. Run 'axon analyze' first."
        )
        raise typer.Exit(code=1)

    storage = KuzuBackend()
    storage.initialize(db_path, read_only=True)
    return storage

app = typer.Typer(
    name="axon-pro",
    help="Axon Pro — Graph-powered code intelligence engine.",
    no_args_is_help=True,
)

def _version_callback(value: bool) -> None:
    """Print the version and exit."""
    if value:
        console.print(f"Axon Pro v{__version__}")
        raise typer.Exit()

@app.callback()
def main(
    version: Optional[bool] = typer.Option(  # noqa: N803
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=_version_callback,
        is_eager=True,
    ),
) -> None:
    """Axon Pro — Graph-powered code intelligence engine."""

@app.command()
def analyze(
    path: Path = typer.Argument(Path("."), help="Path to the repository to index."),
    full: bool = typer.Option(False, "--full", help="Perform a full re-index."),
    neo4j: bool = typer.Option(False, "--neo4j", help="Export to a Neo4j database."),
    neo4j_url: str = typer.Option("bolt://localhost:7687", "--neo4j-url", help="Neo4j connection URI."),
    neo4j_user: str = typer.Option("neo4j", "--neo4j-user", help="Neo4j username."),
    neo4j_pass: str = typer.Option("password", "--neo4j-pass", help="Neo4j password."),
) -> None:
    """Index a repository into a knowledge graph."""
    from axon_pro.core.ingestion.pipeline import PipelineResult, run_pipeline
    from axon_pro.core.storage.kuzu_backend import KuzuBackend

    repo_path = path.resolve()
    if not repo_path.is_dir():
        console.print(f"[red]Error:[/red] {repo_path} is not a directory.")
        raise typer.Exit(code=1)

    console.print(f"[bold]Indexing[/bold] {repo_path}")

    axon_dir = repo_path / ".axon-pro"
    axon_dir.mkdir(parents=True, exist_ok=True)
    db_path = axon_dir / "kuzu"

    if neo4j:
        from axon_pro.core.storage.neo4j_backend import Neo4jBackend
        storage = Neo4jBackend(uri=neo4j_url, user=neo4j_user, password=neo4j_pass)
        console.print(f"[green]Using Neo4j backend at {neo4j_url}[/green]")
    else:
        storage = KuzuBackend()
        storage.initialize(db_path)

    result: PipelineResult | None = None
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        task = progress.add_task("Starting...", total=None)

        def on_progress(phase: str, pct: float) -> None:
            progress.update(task, description=f"{phase} ({pct:.0%})")

        _, result = run_pipeline(
            repo_path=repo_path,
            storage=storage,
            full=full,
            progress_callback=on_progress,
        )

    meta = {
        "version": __version__,
        "name": repo_path.name,
        "path": str(repo_path),
        "stats": {
            "files": result.files,
            "symbols": result.symbols,
            "relationships": result.relationships,
            "clusters": result.clusters,
            "flows": result.processes,
            "dead_code": result.dead_code,
            "coupled_pairs": result.coupled_pairs,
        },
        "last_indexed_at": datetime.now(tz=timezone.utc).isoformat(),
    }
    meta_path = axon_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    console.print()
    console.print("[bold green]Indexing complete.[/bold green]")
    console.print(f"  Files:          {result.files}")
    console.print(f"  Symbols:        {result.symbols}")
    console.print(f"  Relationships:  {result.relationships}")
    if result.clusters > 0:
        console.print(f"  Clusters:       {result.clusters}")
    if result.processes > 0:
        console.print(f"  Flows:          {result.processes}")
    if result.dead_code > 0:
        console.print(f"  Dead code:      {result.dead_code}")
    if result.coupled_pairs > 0:
        console.print(f"  Coupled pairs:  {result.coupled_pairs}")
    console.print(f"  Duration:       {result.duration_seconds:.2f}s")

    storage.close()

@app.command()
def status() -> None:
    """Show index status for current repository."""
    repo_path = Path.cwd().resolve()
    meta_path = repo_path / ".axon-pro" / "meta.json"

    if not meta_path.exists():
        console.print(
            f"[red]Error:[/red] No index found at {repo_path}. Run 'axon analyze' first."
        )
        raise typer.Exit(code=1)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    stats = meta.get("stats", {})

    console.print(f"[bold]Index status for[/bold] {repo_path}")
    console.print(f"  Version:        {meta.get('version', '?')}")
    console.print(f"  Last indexed:   {meta.get('last_indexed_at', '?')}")
    console.print(f"  Files:          {stats.get('files', '?')}")
    console.print(f"  Symbols:        {stats.get('symbols', '?')}")
    console.print(f"  Relationships:  {stats.get('relationships', '?')}")

    if stats.get("clusters", 0) > 0:
        console.print(f"  Clusters:       {stats['clusters']}")
    if stats.get("flows", 0) > 0:
        console.print(f"  Flows:          {stats['flows']}")
    if stats.get("dead_code", 0) > 0:
        console.print(f"  Dead code:      {stats['dead_code']}")
    if stats.get("coupled_pairs", 0) > 0:
        console.print(f"  Coupled pairs:  {stats['coupled_pairs']}")

@app.command()
def check() -> None:
    """Run architectural linting rules against the knowledge graph."""
    from axon_pro.core.storage.kuzu_backend import KuzuBackend
    
    repo_path = Path.cwd().resolve()
    meta_path = repo_path / ".axon-pro" / "meta.json"
    if not meta_path.exists():
        console.print("[red]Error:[/red] No index found. Run 'axon-pro analyze' first.")
        raise typer.Exit(1)

    db_path = repo_path / ".axon-pro" / "kuzu"
    storage = KuzuBackend()
    storage.initialize(db_path)

    console.print("[bold]Running Architectural Audit...[/bold]\n")
    
    violations = 0
    
    # Rule 1: No Direct DB calls in Controllers (Architecture: Use Services/Repositories)
    # We query for calls where receiver is 'DB' and the caller class ends with 'Controller'
    db_calls = storage.query("MATCH (c:class)-[:defines]->(m:method)-[:calls]->(call) WHERE c.name ENDS WITH 'Controller' AND call.receiver = 'DB' RETURN c.name AS controller, call.name AS method, call.line AS line")
    if db_calls:
        console.print("[yellow]Violation:[/yellow] Direct DB Facade usage found in Controllers (Architecture: Use Services/Repositories)")
        for call in db_calls:
            console.print(f"  - {call['controller']} uses DB::{call['method']}() at line {call['line']}")
            violations += 1

    # Rule 2: N+1 Query Warnings
    n_plus_ones = storage.query("MATCH (n) WHERE n.n_plus_one_warnings IS NOT NULL RETURN n.name AS name, n.file_path AS file, n.n_plus_one_warnings AS warnings")
    if n_plus_ones:
        console.print("\n[red]Warning:[/red] Potential N+1 queries detected in loops")
        for n in n_plus_ones:
            import json
            ws = json.loads(n['warnings']) if isinstance(n['warnings'], str) else n['warnings']
            for w in ws:
                console.print(f"  - {n['name']} ({n['file']}): Loop calls {w['method']}() at line {w['line']}")
                violations += 1

    if violations == 0:
        console.print("[green]Success:[/green] No architectural violations found.")
    else:
        console.print(f"\n[bold red]Audit Failed:[/bold red] Found {violations} violations.")
        # We don't necessarily want to fail the command during interactive use, 
        # but we could exit with 1 for CI/CD usage.
        # raise typer.Exit(1)

    storage.close()

@app.command()
def brief() -> None:
    """Generate a high-level architectural brief of the codebase."""
    from axon_pro.core.storage.kuzu_backend import KuzuBackend
    
    repo_path = Path.cwd().resolve()
    meta_path = repo_path / ".axon-pro" / "meta.json"
    if not meta_path.exists():
        console.print("[red]Error:[/red] No index found. Run 'axon-pro analyze' first.")
        raise typer.Exit(1)

    db_path = repo_path / ".axon-pro" / "kuzu"
    storage = KuzuBackend()
    storage.initialize(db_path)

    console.print(f"[bold blue]Architectural Brief: {repo_path.name}[/bold blue]\n")

    # 1. Core Models
    models = storage.query("MATCH (c:class) WHERE c.bases CONTAINS 'Model' RETURN c.name AS name LIMIT 10")
    if models:
        console.print("[bold]Core Domain Models:[/bold]")
        console.print("  " + ", ".join([m['name'] for m in models]))

    # 2. Key Entry Points (Routes)
    routes = storage.query("MATCH (r:route) RETURN r.name AS name LIMIT 5")
    if routes:
        console.print("\n[bold]Primary Entry Points:[/bold]")
        for r in routes:
            console.print(f"  - {r['name']}")

    # 3. Message Handlers (Jobs/Events)
    jobs = storage.query("MATCH (j:job) RETURN j.name AS name LIMIT 5")
    if jobs:
        console.print("\n[bold]Asynchronous Jobs:[/bold]")
        for j in jobs:
            console.print(f"  - {j['name']}")

    # 4. Critical Dependencies (Most coupled files)
    coupling = storage.query("MATCH (f1:file)-[r:coupled_with]->(f2:file) RETURN f1.name AS f1, f2.name AS f2 ORDER BY r.weight DESC LIMIT 3")
    if coupling:
        console.print("\n[bold]High-Risk Change Areas (Tightly Coupled):[/bold]")
        for c in coupling:
            console.print(f"  - {c['f1']} <--> {c['f2']}")

    console.print("\n[dim]Use 'axon-pro context <symbol>' for deep-dives.[/dim]")
    storage.close()

@app.command(name="list")
def list_repos() -> None:
    """List all indexed repositories."""
    from axon_pro.mcp.tools import handle_list_repos

    result = handle_list_repos()
    console.print(result)

@app.command()
def clean(
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation prompt."),
) -> None:
    """Delete index for current repository."""
    repo_path = Path.cwd().resolve()
    axon_dir = repo_path / ".axon-pro"

    if not axon_dir.exists():
        console.print(
            f"[red]Error:[/red] No index found at {repo_path}. Nothing to clean."
        )
        raise typer.Exit(code=1)

    if not force:
        confirm = typer.confirm(f"Delete index at {axon_dir}?")
        if not confirm:
            console.print("Aborted.")
            raise typer.Exit()

    shutil.rmtree(axon_dir)
    console.print(f"[green]Deleted[/green] {axon_dir}")

@app.command()
def query(
    q: str = typer.Argument(..., help="Search query for the knowledge graph."),
    limit: int = typer.Option(20, "--limit", "-n", help="Maximum number of results."),
) -> None:
    """Search the knowledge graph."""
    from axon_pro.mcp.tools import handle_query

    storage = _load_storage()
    result = handle_query(storage, q, limit=limit)
    console.print(result)
    storage.close()

@app.command()
def context(
    name: str = typer.Argument(..., help="Symbol name to inspect."),
) -> None:
    """Show 360-degree view of a symbol."""
    from axon_pro.mcp.tools import handle_context

    storage = _load_storage()
    result = handle_context(storage, name)
    console.print(result)
    storage.close()

@app.command()
def impact(
    target: str = typer.Argument(..., help="Symbol to analyze blast radius for."),
    depth: int = typer.Option(3, "--depth", "-d", help="Traversal depth."),
) -> None:
    """Show blast radius of changing a symbol."""
    from axon_pro.mcp.tools import handle_impact

    storage = _load_storage()
    result = handle_impact(storage, target, depth=depth)
    console.print(result)
    storage.close()

@app.command(name="dead-code")
def dead_code() -> None:
    """List all detected dead code."""
    from axon_pro.mcp.tools import handle_dead_code

    storage = _load_storage()
    result = handle_dead_code(storage)
    console.print(result)
    storage.close()

@app.command()
def cypher(
    query: str = typer.Argument(..., help="Raw Cypher query to execute."),
) -> None:
    """Execute raw Cypher against the knowledge graph."""
    from axon_pro.mcp.tools import handle_cypher

    storage = _load_storage()
    result = handle_cypher(storage, query)
    console.print(result)
    storage.close()

@app.command()
def setup(
    claude: bool = typer.Option(False, "--claude", help="Configure MCP for Claude Code."),
    cursor: bool = typer.Option(False, "--cursor", help="Configure MCP for Cursor."),
) -> None:
    """Configure MCP for Claude Code / Cursor."""
    mcp_config = {
        "command": "axon-pro",
        "args": ["serve", "--watch"],
    }

    if claude or (not claude and not cursor):
        console.print("[bold]Add to your Claude Code MCP config:[/bold]")
        console.print(json.dumps({"axon-pro": mcp_config}, indent=2))

    if cursor or (not claude and not cursor):
        console.print("[bold]Add to your Cursor MCP config:[/bold]")
        console.print(json.dumps({"axon-pro": mcp_config}, indent=2))

@app.command()
def watch() -> None:
    """Watch mode — re-index on file changes."""
    import asyncio

    from axon_pro.core.ingestion.pipeline import run_pipeline
    from axon_pro.core.ingestion.watcher import watch_repo
    from axon_pro.core.storage.kuzu_backend import KuzuBackend

    repo_path = Path.cwd().resolve()
    axon_dir = repo_path / ".axon-pro"
    axon_dir.mkdir(parents=True, exist_ok=True)
    db_path = axon_dir / "kuzu"

    storage = KuzuBackend()
    storage.initialize(db_path)

    if not (axon_dir / "meta.json").exists():
        console.print("[bold]Running initial index...[/bold]")
        run_pipeline(repo_path, storage, full=True)

    console.print(f"[bold]Watching[/bold] {repo_path} for changes (Ctrl+C to stop)")

    try:
        asyncio.run(watch_repo(repo_path, storage))
    except KeyboardInterrupt:
        console.print("\n[bold]Watch stopped.[/bold]")
    finally:
        storage.close()

@app.command()
def diff(
    branch_range: str = typer.Argument(..., help="Branch range for comparison (e.g. main..feature)."),
) -> None:
    """Structural branch comparison."""
    from axon_pro.core.diff import diff_branches, format_diff

    repo_path = Path.cwd().resolve()
    try:
        result = diff_branches(repo_path, branch_range)
    except (ValueError, RuntimeError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(code=1) from exc

    console.print(format_diff(result))

@app.command()
def mcp() -> None:
    """Start MCP server (stdio transport)."""
    import asyncio

    from axon_pro.mcp.server import main as mcp_main

    asyncio.run(mcp_main())

@app.command()
def serve(
    watch: bool = typer.Option(False, "--watch", "-w", help="Enable file watching with auto-reindex."),
) -> None:
    """Start MCP server, optionally with live file watching."""
    import asyncio
    import sys

    from axon_pro.mcp.server import main as mcp_main, set_lock, set_storage

    if not watch:
        asyncio.run(mcp_main())
        return

    from axon_pro.core.ingestion.pipeline import run_pipeline
    from axon_pro.core.ingestion.watcher import watch_repo
    from axon_pro.core.storage.kuzu_backend import KuzuBackend

    repo_path = Path.cwd().resolve()
    axon_dir = repo_path / ".axon-pro"
    axon_dir.mkdir(parents=True, exist_ok=True)
    db_path = axon_dir / "kuzu"

    storage = KuzuBackend()
    storage.initialize(db_path)

    if not (axon_dir / "meta.json").exists():
        print("Running initial index...", file=sys.stderr)
        run_pipeline(repo_path, storage, full=True)

    lock = asyncio.Lock()
    set_storage(storage)
    set_lock(lock)

    async def _run() -> None:
        from mcp.server.stdio import stdio_server
        from axon_pro.mcp.server import server as mcp_server

        stop = asyncio.Event()

        async with stdio_server() as (read, write):
            async def _mcp_then_stop():
                await mcp_server.run(read, write, mcp_server.create_initialization_options())
                stop.set()

            await asyncio.gather(
                _mcp_then_stop(),
                watch_repo(repo_path, storage, stop_event=stop, lock=lock),
            )

    try:
        asyncio.run(_run())
    except KeyboardInterrupt:
        pass
    finally:
        storage.close()
