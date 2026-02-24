"""High-utility MCP tool handlers for Axon Pro.

Extends the base MCP tools with advanced architectural analysis, pattern
searching, and deep flow explanation capabilities.
"""

from __future__ import annotations

import logging
from typing import Any
from axon_pro.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

def handle_architecture(storage: StorageBackend) -> str:
    """Get a high-level architectural overview of the project."""
    lines = ["Architectural Overview:"]
    lines.append("-" * 25)
    
    counts = {}
    labels_to_count = ["Class", "Function", "File", "Process", "Folder"]
    for label in labels_to_count:
        try:
            rows = storage.execute_raw(f"MATCH (n:{label}) RETURN count(*)")
            counts[label] = rows[0][0] if rows else 0
        except Exception:
            counts[label] = 0
            
    lines.append(f"Components: {counts.get('Class', 0)} Classes, {counts.get('Function', 0)} Functions")
    lines.append(f"Structure: {counts.get('File', 0)} Files across {counts.get('Folder', 0)} Folders")
    lines.append("")
    
    try:
        models = storage.execute_raw("MATCH (n:Class) WHERE n.name ENDS WITH 'Model' OR n.file_path CONTAINS '/Models/' RETURN n.name LIMIT 10")
        if models:
            lines.append("Key Models:")
            for m in models:
                lines.append(f"  - {m[0]}")
            lines.append("")
            
        controllers = storage.execute_raw("MATCH (n:Class) WHERE n.name ENDS WITH 'Controller' RETURN n.name LIMIT 10")
        if controllers:
            lines.append("Key Controllers:")
            for c in controllers:
                lines.append(f"  - {c[0]}")
            lines.append("")
    except Exception:
        pass
        
    lines.append("PRO-TIP: Use `axon_pro_query(text)` to find specific features or `axon_pro_search_code(pattern)` to find literal implementation details.")
    return "\n".join(lines)

def handle_search_code(storage: StorageBackend, pattern: str, limit: int = 10) -> str:
    """Search for code patterns across the indexed content."""
    try:
        query = (
            f"MATCH (n) WHERE n.content CONTAINS '{pattern.replace(chr(39), '')}' "
            f"RETURN n.name, n.file_path, n.start_line, n.content "
            f"LIMIT {limit}"
        )
        rows = storage.execute_raw(query)
        if not rows:
            return f"No code found matching pattern '{pattern}'."
            
        lines = [f"Code search results for '{pattern}':"]
        lines.append("")
        for i, row in enumerate(rows, 1):
            name, file_path, start_line, content = row[0], row[1], row[2], row[3]
            lines.append(f"{i}. {name} in {file_path}:{start_line}")
            idx = content.lower().find(pattern.lower())
            start = max(0, idx - 40)
            end = min(len(content), idx + len(pattern) + 40)
            snippet = content[start:end].replace("\n", " ").strip()
            lines.append(f"   ...{snippet}...")
            lines.append("")
            
        return "\n".join(lines)
    except Exception as e:
        return f"Code search failed: {e}"

def handle_explain_flow(storage: StorageBackend, start_symbol: str) -> str:
    """Explain the execution flow starting from a symbol in natural language."""
    from axon_pro.mcp.tools import _resolve_symbol
    results = _resolve_symbol(storage, start_symbol)
    if not results:
        return f"Symbol '{start_symbol}' not found."
        
    start_id = results[0].node_id
    node = storage.get_node(start_id)
    if not node:
         return f"Symbol '{start_symbol}' not found."

    lines = [f"Flow Explanation for: {node.name}"]
    lines.append("=" * (len(node.name) + 21))
    lines.append("")
    lines.append(f"1. Entry Point: {node.name} ({node.file_path})")
    
    callees = storage.get_callees(start_id)
    if not callees:
        lines.append("This symbol appears to be a leaf node with no downstream calls.")
        return "\n".join(lines)
        
    lines.append(f"2. This symbol coordinates {len(callees)} primary actions:")
    
    categories = {"Database/Model": [], "External Service": [], "Validation/Security": [], "Helper/Util": [], "Other": []}
    for c in callees:
        if "Model" in c.name or "/Models/" in c.file_path or "db" in c.name.lower():
            categories["Database/Model"].append(c)
        elif "service" in c.file_path.lower() or "client" in c.name.lower():
            categories["External Service"].append(c)
        elif "request" in c.name.lower() or "policy" in c.name.lower() or "auth" in c.name.lower():
            categories["Validation/Security"].append(c)
        elif "util" in c.file_path.lower() or "helper" in c.file_path.lower():
            categories["Helper/Util"].append(c)
        else:
            categories["Other"].append(c)
            
    for cat, items in categories.items():
        if items:
            lines.append(f"   - {cat}:")
            for item in items:
                lines.append(f"     * {item.name}")
                
    lines.append("")
    lines.append("3. Deep Trace Recommendation:")
    complex_callee = None
    max_calls = -1
    for c in callees:
        sub_calls = len(storage.get_callees(c.id))
        if sub_calls > max_calls:
            max_calls = sub_calls
            complex_callee = c
            
    if complex_callee:
        lines.append(f"   The most complex logic appears to reside in '{complex_callee.name}'.")
        lines.append(f"   Consider running flow_trace('{complex_callee.name}') for deeper insight.")
        
    return "\n".join(lines)

def handle_check_nplus1(storage: StorageBackend) -> str:
    """Scan for potential N+1 query patterns in the indexed code."""
    try:
        query = (
            "MATCH (f:Function)-[:calls]->(m:Class) "
            "WHERE m.file_path CONTAINS '/Models/' OR m.name ENDS WITH 'Model' "
            "WITH f, count(m) as call_count "
            "WHERE call_count > 3 "
            "RETURN f.name, f.file_path, call_count "
            "ORDER BY call_count DESC LIMIT 10"
        )
        rows = storage.execute_raw(query)
        if not rows:
            return "No obvious N+1 query patterns detected via high-level graph analysis."
            
        lines = ["Potential N+1 or Heavy Database usage detected:"]
        lines.append("-" * 45)
        for row in rows:
            lines.append(f"- {row[0]} in {row[1]}")
            lines.append(f"  Makes {row[2]} distinct calls to Model classes.")
            
            lines.append("")
            lines.append("PRO-TIP: Use `axon_pro_context(symbol)` to see which relationships are being accessed in these functions.")
            return "\n".join(lines)
        
    except Exception:
        return "N+1 check failed or not supported by current schema."

def handle_impact_on_tests(storage: StorageBackend, symbol: str) -> str:
    """Find tests that might be affected by changing a symbol."""
    from axon_pro.mcp.tools import _resolve_symbol
    results = _resolve_symbol(storage, symbol)
    if not results:
        return f"Symbol '{symbol}' not found."
        
    start_id = results[0].node_id
    affected = storage.traverse(start_id, depth=5, direction="callers")
    tests = [n for n in affected if "/tests/" in n.file_path or "/test_" in n.file_path or n.name.startswith("test_")]
    
    if not tests:
        return f"No direct or indirect tests found that call '{symbol}' (up to 5 hops)."
        
    lines = [f"Test Impact Analysis for: {symbol}"]
    lines.append(f"Found {len(tests)} potentially affected tests:")
    lines.append("")
    
    by_file = {}
    for t in tests:
        by_file.setdefault(t.file_path, []).append(t.name)
        
    for file_path, names in by_file.items():
        lines.append(f"  File: {file_path}")
        for name in names:
            lines.append(f"    - {name}")
        lines.append("")
        
    return "\n".join(lines)
