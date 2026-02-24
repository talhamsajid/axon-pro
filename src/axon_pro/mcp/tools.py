"""MCP tool handler implementations for Axon Pro.

Each function accepts a storage backend and the tool-specific arguments,
performs the appropriate query, and returns a human-readable string suitable
for inclusion in an MCP ``TextContent`` response.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from axon_pro.core.search.hybrid import hybrid_search
from axon_pro.core.storage.base import StorageBackend


def _resolve_symbol(storage: StorageBackend, symbol: str) -> list:
    """Resolve a symbol name to search results, preferring exact name matches."""
    if hasattr(storage, "exact_name_search"):
        results = storage.exact_name_search(symbol, limit=1)
        if results:
            return results
    return storage.fts_search(symbol, limit=1)

def handle_list_repos(registry_dir: Path | None = None) -> str:
    """List indexed repositories by scanning for .axon-pro directories.

    Scans the global registry directory (defaults to ``~/.axon-pro/repos``) for
    project metadata files and returns a formatted summary.

    Args:
        registry_dir: Directory containing repo metadata. If ``None``,
            defaults to ``~/.axon-pro/repos``.

    Returns:
        Formatted list of indexed repositories with stats, or a message
        indicating none were found.
    """
    use_cwd_fallback = registry_dir is None
    if registry_dir is None:
        registry_dir = Path.home() / ".axon-pro" / "repos"

    repos: list[dict[str, Any]] = []

    if registry_dir.exists():
        for meta_file in registry_dir.glob("*/meta.json"):
            try:
                data = json.loads(meta_file.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                continue

    if not repos and use_cwd_fallback:
        # Fall back: scan current directory for .axon-pro
        cwd_axon = Path.cwd() / ".axon-pro" / "meta.json"
        if cwd_axon_pro.exists():
            try:
                data = json.loads(cwd_axon_pro.read_text())
                repos.append(data)
            except (json.JSONDecodeError, OSError):
                pass

    if not repos:
        return "No indexed repositories found. Run `axon index` on a project first."

    lines = [f"Indexed repositories ({len(repos)}):"]
    lines.append("")
    for i, repo in enumerate(repos, 1):
        name = repo.get("name", "unknown")
        path = repo.get("path", "")
        stats = repo.get("stats", {})
        files = stats.get("files", "?")
        symbols = stats.get("symbols", "?")
        relationships = stats.get("relationships", "?")
        lines.append(f"  {i}. {name}")
        lines.append(f"     Path: {path}")
        lines.append(f"     Files: {files}  Symbols: {symbols}  Relationships: {relationships}")
        lines.append("")

    return "\n".join(lines)

def handle_query(storage: StorageBackend, query: str, limit: int = 20) -> str:
    """Execute hybrid search and format results.

    Args:
        storage: The storage backend to search against.
        query: Text search query.
        limit: Maximum number of results (default 20).

    Returns:
        Formatted search results with file, name, label, and snippet.
    """
    results = hybrid_search(query, storage, limit=limit)
    if not results:
        return f"No results found for '{query}'."

    lines = []
    for i, r in enumerate(results, 1):
        label = r.label.title() if r.label else "Unknown"
        lines.append(f"{i}. {r.node_name} ({label}) -- {r.file_path}")
        if r.snippet:
            snippet = r.snippet[:200].replace("\n", " ").strip()
            lines.append(f"   {snippet}")
    lines.append("")
    lines.append("PRO-TIP: Use `axon_pro_context(symbol)` on a specific symbol to see its callers, callees, and architectural links.")
    return "\n".join(lines)

def handle_context(storage: StorageBackend, symbol: str) -> str:
    """Provide a 360-degree view of a symbol.

    Looks up the symbol by name via full-text search, then retrieves its
    callers, callees, and type references.

    Args:
        storage: The storage backend.
        symbol: The symbol name to look up.

    Returns:
        Formatted view including callers, callees, type refs, and guidance.
    """
    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    node = storage.get_node(results[0].node_id)
    if not node:
        return f"Symbol '{symbol}' not found."

    label_display = node.label.value.title() if node.label else "Unknown"
    lines = [f"Symbol: {node.name} ({label_display})"]
    lines.append(f"File: {node.file_path}:{node.start_line}-{node.end_line}")

    if node.signature:
        lines.append(f"Signature: {node.signature}")

    if node.is_dead:
        lines.append("Status: DEAD CODE (unreachable)")

    callers = storage.get_callers(node.id)
    if callers:
        lines.append(f"\nCallers ({len(callers)}):")
        for c in callers:
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}")

    callees = storage.get_callees(node.id)
    if callees:
        lines.append(f"\nCallees ({len(callees)}):")
        for c in callees:
            lines.append(f"  -> {c.name}  {c.file_path}:{c.start_line}")

    type_refs = storage.get_type_refs(node.id)
    if type_refs:
        lines.append(f"\nType references ({len(type_refs)}):")
        for t in type_refs:
            lines.append(f"  -> {t.name}  {t.file_path}")

    # --- Laravel Relationships ---
    laravel_rels = []
    
    # 1. Dispatching/Messaging
    dispatches = storage.query("MATCH (n {id: $id})-[:dispatches]->(t) RETURN t.name AS name, labels(t) AS labels", {"id": node.id})
    for d in dispatches:
        laravel_rels.append(f"  [DISPATCHES] -> {d['name']} ({d['labels'][0]})")
        
    # 2. Event Listening
    listens = storage.query("MATCH (n {id: $id})-[:listens_to]->(t) RETURN t.name AS name", {"id": node.id})
    for l in listens:
        laravel_rels.append(f"  [LISTENS TO] -> Event: {l['name']}")
        
    # 3. View Rendering
    renders = storage.query("MATCH (n {id: $id})-[:renders]->(t) RETURN t.name AS name", {"id": node.id})
    for r in renders:
        laravel_rels.append(f"  [RENDERS]    -> View: {r['name']}")
        
    # 4. Security/Validation
    auth = storage.query("MATCH (n {id: $id})-[:authorized_by]->(t) RETURN t.name AS name", {"id": node.id})
    for a in auth:
        laravel_rels.append(f"  [AUTH BY]    -> Policy: {a['name']}")
        
    valid = storage.query("MATCH (n {id: $id})-[:validated_by]->(t) RETURN t.name AS name", {"id": node.id})
    for v in valid:
        laravel_rels.append(f"  [VALID BY]   -> Request: {v['name']}")

    # 5. DI Container Bindings
    binds = storage.query("MATCH (n {id: $id})-[:binds]->(t) RETURN t.name AS name", {"id": node.id})
    for b in binds:
        laravel_rels.append(f"  [BINDS TO]   -> Concrete: {b['name']}")

    if laravel_rels:
        lines.append("\nLaravel Architectural Links:")
        lines.extend(laravel_rels)

    lines.append("")
    lines.append("PRO-TIP: Use `axon_pro_impact(symbol)` to see the blast radius of changing this symbol, or `axon_pro_explain_flow(symbol)` for a natural language behavior summary.")
    return "\n".join(lines)

def handle_impact(storage: StorageBackend, symbol: str, depth: int = 3) -> str:
    """Analyse the blast radius of changing a symbol.

    Uses BFS traversal through CALLS edges to find all affected symbols
    up to the specified depth.

    Args:
        storage: The storage backend.
        symbol: The symbol name to analyse.
        depth: Maximum traversal depth (default 3).

    Returns:
        Formatted impact analysis showing affected symbols at each depth level.
    """
    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        return f"Symbol '{symbol}' not found."

    affected = storage.traverse(start_node.id, depth, direction="callers")
    if not affected:
        return f"No upstream callers found for '{symbol}'."

    lines = [f"Impact analysis for: {start_node.name} ({start_node.label.value.title()})"]
    lines.append(f"Depth: {depth}")
    lines.append(f"Total affected symbols: {len(affected)}")
    lines.append("")

    for i, node in enumerate(affected, 1):
        label = node.label.value.title() if node.label else "Unknown"
        lines.append(f"  {i}. {node.name} ({label}) -- {node.file_path}:{node.start_line}")

    lines.append("")
    lines.append("PRO-TIP: Use `axon_pro_impact_on_tests(symbol)` to identify exactly which tests you must run after modifying this symbol.")
    return "\n".join(lines)

def handle_dead_code(storage: StorageBackend) -> str:
    """List all symbols marked as dead code.

    Delegates to :func:`~axon_pro.mcp.resources.get_dead_code_list` for the
    shared query and formatting.

    Args:
        storage: The storage backend.

    Returns:
        Formatted list of dead code symbols grouped by file.
    """
    from axon_pro.mcp.resources import get_dead_code_list

    return get_dead_code_list(storage)

_DIFF_FILE_PATTERN = re.compile(r"^diff --git a/(.+?) b/(.+?)$", re.MULTILINE)
_DIFF_HUNK_PATTERN = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@", re.MULTILINE)

def handle_detect_changes(storage: StorageBackend, diff: str) -> str:
    """Map git diff output to affected symbols.

    Parses the diff to find changed files and line ranges, then queries
    the storage backend to identify which symbols those lines belong to.

    Args:
        storage: The storage backend.
        diff: Raw git diff output string.

    Returns:
        Formatted list of affected symbols per changed file.
    """
    if not diff.strip():
        return "Empty diff provided."

    changed_files: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None

    for line in diff.split("\n"):
        file_match = _DIFF_FILE_PATTERN.match(line)
        if file_match:
            current_file = file_match.group(2)
            if current_file not in changed_files:
                changed_files[current_file] = []
            continue

        hunk_match = _DIFF_HUNK_PATTERN.match(line)
        if hunk_match and current_file is not None:
            start = int(hunk_match.group(1))
            count = int(hunk_match.group(2) or "1")
            changed_files[current_file].append((start, start + count - 1))

    if not changed_files:
        return "Could not parse any changed files from the diff."

    lines = [f"Changed files: {len(changed_files)}"]
    lines.append("")
    total_affected = 0

    for file_path, ranges in changed_files.items():
        affected_symbols = []
        try:
            rows = storage.execute_raw(
                f"MATCH (n) WHERE n.file_path = '{file_path.replace(chr(39), '')}' "
                f"AND n.start_line > 0 "
                f"RETURN n.id, n.name, n.file_path, n.start_line, n.end_line"
            )
            for row in rows or []:
                node_id = row[0] or ""
                name = row[1] or ""
                start_line = row[3] or 0
                end_line = row[4] or 0
                label_prefix = node_id.split(":", 1)[0] if node_id else ""
                for start, end in ranges:
                    if start_line <= end and end_line >= start:
                        affected_symbols.append(
                            (name, label_prefix.title(), start_line, end_line)
                        )
                        break
        except Exception:
            pass

        lines.append(f"  {file_path}:")
        if affected_symbols:
            for sym_name, label, s_line, e_line in affected_symbols:
                lines.append(f"    - {sym_name} ({label}) lines {s_line}-{e_line}")
                total_affected += 1
        else:
            lines.append("    (no indexed symbols in changed lines)")
        lines.append("")

    lines.append(f"Total affected symbols: {total_affected}")
    lines.append("")
    lines.append("PRO-TIP: Use `axon_pro_impact(symbol)` on these affected symbols to see downstream effects across the entire codebase.")
    return "\n".join(lines)

_WRITE_KEYWORDS = re.compile(
    r"\b(DELETE|DROP|CREATE|SET|REMOVE|MERGE|DETACH|INSTALL|LOAD|COPY|CALL)\b",
    re.IGNORECASE,
)

def handle_cypher(storage: StorageBackend, query: str) -> str:
    """Execute a raw Cypher query and return formatted results.

    Only read-only queries are allowed.  Queries containing write keywords
    (DELETE, DROP, CREATE, SET, etc.) are rejected.

    Args:
        storage: The storage backend.
        query: The Cypher query string.

    Returns:
        Formatted query results, or an error message if execution fails.
    """
    if _WRITE_KEYWORDS.search(query):
        return (
            "Query rejected: only read-only queries (MATCH/RETURN) are allowed. "
            "Write operations (DELETE, DROP, CREATE, SET, MERGE) are not permitted."
        )

    try:
        rows = storage.execute_raw(query)
    except Exception as exc:
        return f"Cypher query failed: {exc}"

    if not rows:
        return "Query returned no results."

    lines = [f"Results ({len(rows)} rows):"]
    lines.append("")
    for i, row in enumerate(rows, 1):
        formatted_values = [str(v) for v in row]
        lines.append(f"  {i}. {' | '.join(formatted_values)}")

    return "\n".join(lines)

def handle_file_context(storage: StorageBackend, file_path: str) -> str:
    """Get a detailed breakdown of a file's context.
    
    Shows all symbols defined in the file, what they call externally, 
    and who calls them from outside the file.
    """
    rows = storage.execute_raw(
        f"MATCH (n) WHERE n.file_path ENDS WITH '{file_path}' "
        f"RETURN n.id, n.name, labels(n)[0] AS label "
        f"ORDER BY n.start_line"
    )
    if not rows:
        return f"No symbols found in file matching '{file_path}'."
        
    lines = [f"File Context for: {file_path}"]
    lines.append("")
    
    for row in rows:
        node_id, name, label = row[0], row[1], row[2]
        lines.append(f"Symbol: {name} ({label})")
        
        # Outgoing calls
        out_rows = storage.execute_raw(
            "MATCH (n {id: $id})-[:calls]->(ext) "
            "WHERE NOT ext.file_path ENDS WITH $file "
            "RETURN ext.name, ext.file_path", 
            {"id": node_id, "file": file_path}
        )
        if out_rows:
            lines.append(f"  -> Calls External ({len(out_rows)}):")
            for out in out_rows:
                lines.append(f"       {out[0]} in {out[1]}")
                
        # Incoming calls
        in_rows = storage.execute_raw(
            "MATCH (ext)-[:calls]->(n {id: $id}) "
            "WHERE NOT ext.file_path ENDS WITH $file "
            "RETURN ext.name, ext.file_path", 
            {"id": node_id, "file": file_path}
        )
        if in_rows:
            lines.append(f"  <- Called By External ({len(in_rows)}):")
            for in_r in in_rows:
                lines.append(f"       {in_r[0]} in {in_r[1]}")
                
        lines.append("")
        
    return "\n".join(lines)

def handle_related_files(storage: StorageBackend, file_path: str) -> str:
    """Find all files that relate to a given file through dependencies."""
    out_rows = storage.execute_raw(
        "MATCH (n)-[:calls]->(ext) "
        "WHERE n.file_path ENDS WITH $file AND NOT ext.file_path ENDS WITH $file "
        "RETURN ext.file_path, count(*) as weight "
        "ORDER BY weight DESC LIMIT 10",
        {"file": file_path}
    )
    
    in_rows = storage.execute_raw(
        "MATCH (ext)-[:calls]->(n) "
        "WHERE n.file_path ENDS WITH $file AND NOT ext.file_path ENDS WITH $file "
        "RETURN ext.file_path, count(*) as weight "
        "ORDER BY weight DESC LIMIT 10",
        {"file": file_path}
    )
    
    if not out_rows and not in_rows:
        return f"No related files found for '{file_path}'."
        
    lines = [f"Related Files for: {file_path}"]
    lines.append("")
    
    if out_rows:
        lines.append("Files this file depends on (Calls Out To):")
        for row in out_rows:
            lines.append(f"  - {row[0]} ({row[1]} references)")
        lines.append("")
        
    if in_rows:
        lines.append("Files that depend on this file (Called By):")
        for row in in_rows:
            lines.append(f"  - {row[0]} ({row[1]} references)")
            
    return "\n".join(lines)

def handle_flow_trace(storage: StorageBackend, symbol: str, depth: int = 5) -> str:
    """Trace the execution flow starting from a specific symbol."""
    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."

    start_node = storage.get_node(results[0].node_id)
    if not start_node:
        return f"Symbol '{symbol}' not found."

    affected = storage.traverse(start_node.id, depth, direction="callees")
    if not affected:
        return f"No downstream calls found for '{symbol}'."

    lines = [f"Execution Flow Trace for: {start_node.name}"]
    lines.append(f"Max Depth: {depth}")
    lines.append(f"Total downstream symbols called: {len(affected)}")
    lines.append("")

    for i, node in enumerate(affected, 1):
        label = node.label.value.title() if node.label else "Unknown"
        lines.append(f"  {i}. {node.name} ({label}) -- {node.file_path}:{node.start_line}")

    lines.append("")
    lines.append("Tip: Use axon_pro_file_context on specific files for more detail.")
    return "\n".join(lines)

