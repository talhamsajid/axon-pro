"""MCP server for Axon Pro — exposes code intelligence tools over stdio transport.

Registers seven tools and three resources that give AI agents and MCP clients
access to the Axon Pro knowledge graph.  The server lazily initialises a
:class:`KuzuBackend` from the ``.axon-pro/kuzu`` directory in the current
working directory.

Usage::

    # MCP server only
    axon mcp

    # MCP server with live file watching (recommended)
    axon serve --watch
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Resource, TextContent, Tool

from axon_pro.core.storage.kuzu_backend import KuzuBackend
from axon_pro.mcp.resources import get_dead_code_list, get_overview, get_schema, get_agent_guidelines
from axon_pro.mcp.tools import (
    handle_context,
    handle_cypher,
    handle_dead_code,
    handle_detect_changes,
    handle_file_context,
    handle_flow_trace,
    handle_impact,
    handle_list_repos,
    handle_query,
    handle_related_files,
)
from axon_pro.mcp.high_utility_tools import (
    handle_architecture,
    handle_check_nplus1,
    handle_explain_flow,
    handle_impact_on_tests,
    handle_search_code,
)

logger = logging.getLogger(__name__)

server = Server("axon-pro")

_storage: KuzuBackend | None = None
_lock: asyncio.Lock | None = None


def set_storage(storage: KuzuBackend) -> None:
    """Inject a pre-initialised storage backend (e.g. from ``axon serve --watch``)."""
    global _storage  # noqa: PLW0603
    _storage = storage


def set_lock(lock: asyncio.Lock) -> None:
    """Inject a shared lock for coordinating storage access with the file watcher."""
    global _lock  # noqa: PLW0603
    _lock = lock


def _get_storage() -> KuzuBackend:
    """Lazily initialise and return the KuzuDB storage backend.

    Searches upwards from the current directory for a ``.axon-pro/kuzu`` 
    database. If found, the backend is initialised from that path.
    """
    global _storage  # noqa: PLW0603
    if _storage is None:
        _storage = KuzuBackend()
        
        # Search upwards for .axon-pro
        curr = Path.cwd().resolve()
        db_path = None
        for parent in [curr] + list(curr.parents):
            candidate = parent / ".axon-pro" / "kuzu"
            if candidate.exists():
                db_path = candidate
                break
                
        if db_path:
            _storage.initialize(db_path, read_only=True)
            logger.info("Initialised storage (read-only) from %s", db_path)
        else:
            logger.warning("No .axon-pro/kuzu directory found in %s or its parents", Path.cwd())
    return _storage

TOOLS: list[Tool] = [
    Tool(
        name="axon_pro_list_repos",
        description="List all indexed repositories with their stats.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_pro_query",
        description=(
            "Search the knowledge graph using hybrid (keyword + vector) search. "
            "Returns ranked symbols matching the query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 20).",
                    "default": 20,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="axon_pro_context",
        description=(
            "Get a 360-degree view of a symbol: callers, callees, type references, "
            "and community membership."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to look up.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_pro_impact",
        description=(
            "Blast radius analysis: find all symbols affected by changing a given symbol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to analyse.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum traversal depth (default 3).",
                    "default": 3,
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_pro_dead_code",
        description="List all detected dead code in the repository.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_pro_detect_changes",
        description=(
            "Analyse code changes in a diff and find affected symbols. "
            "Helps understand the impact of recent modifications."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "diff": {
                    "type": "string",
                    "description": "The git diff to analyse.",
                },
            },
            "required": ["diff"],
        },
    ),
    Tool(
        name="axon_pro_cypher",
        description="Execute a raw Cypher query against the knowledge graph.",
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Cypher query to execute.",
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="axon_pro_file_context",
        description="Get full context for a specific file, including all its symbols.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the file.",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="axon_pro_related_files",
        description="Find files that are structurally or temporally coupled to a given file.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the reference file.",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="axon_pro_flow_trace",
        description="Trace the execution flow downstream starting from a specific symbol to understand its behavior deeply.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Name of the symbol to trace.",
                },
                "depth": {
                    "type": "integer",
                    "description": "Maximum traversal depth (default 5).",
                    "default": 5,
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_pro_architecture",
        description="Get a high-level architectural overview of the project (Models, Controllers, Jobs, etc).",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_pro_search_code",
        description="Search for code patterns or literal snippets across the indexed content.",
        inputSchema={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Text or pattern to search for in code content.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results (default 10).",
                    "default": 10,
                },
            },
            "required": ["pattern"],
        },
    ),
    Tool(
        name="axon_pro_explain_flow",
        description="Explain the execution flow of a symbol in natural language, categorizing downstream actions.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name to explain.",
                },
            },
            "required": ["symbol"],
        },
    ),
    Tool(
        name="axon_pro_check_nplus1",
        description="Scan for potential N+1 query patterns or heavy database usage in the indexed code.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="axon_pro_impact_on_tests",
        description="Identify which tests might be affected by changing a specific symbol.",
        inputSchema={
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "Symbol name to analyze test impact for.",
                },
            },
            "required": ["symbol"],
        },
    ),
]

@server.list_tools()
async def list_tools() -> list[Tool]:
    """Return the list of available Axon Pro tools."""
    return TOOLS

def _dispatch_tool(name: str, arguments: dict, storage: KuzuBackend) -> str:
    """Synchronous tool dispatch — called directly or via ``asyncio.to_thread``."""
    if name == "axon_pro_list_repos":
        return handle_list_repos()
    elif name == "axon_pro_query":
        return handle_query(storage, arguments.get("query", ""), limit=arguments.get("limit", 20))
    elif name == "axon_pro_context":
        return handle_context(storage, arguments.get("symbol", ""))
    elif name == "axon_pro_impact":
        return handle_impact(storage, arguments.get("symbol", ""), depth=arguments.get("depth", 3))
    elif name == "axon_pro_dead_code":
        return handle_dead_code(storage)
    elif name == "axon_pro_detect_changes":
        return handle_detect_changes(storage, arguments.get("diff", ""))
    elif name == "axon_pro_cypher":
        return handle_cypher(storage, arguments.get("query", ""))
    elif name == "axon_pro_file_context":
        return handle_file_context(storage, arguments.get("file_path", ""))
    elif name == "axon_pro_related_files":
        return handle_related_files(storage, arguments.get("file_path", ""))
    elif name == "axon_pro_flow_trace":
        return handle_flow_trace(storage, arguments.get("symbol", ""), depth=arguments.get("depth", 5))
    elif name == "axon_pro_architecture":
        return handle_architecture(storage)
    elif name == "axon_pro_search_code":
        return handle_search_code(storage, arguments.get("pattern", ""), limit=arguments.get("limit", 10))
    elif name == "axon_pro_explain_flow":
        return handle_explain_flow(storage, arguments.get("symbol", ""))
    elif name == "axon_pro_check_nplus1":
        return handle_check_nplus1(storage)
    elif name == "axon_pro_impact_on_tests":
        return handle_impact_on_tests(storage, arguments.get("symbol", ""))
    else:
        return f"Unknown tool: {name}"


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Dispatch a tool call to the appropriate handler."""
    storage = _get_storage()

    if _lock is not None:
        async with _lock:
            result = await asyncio.to_thread(_dispatch_tool, name, arguments, storage)
    else:
        result = _dispatch_tool(name, arguments, storage)

    return [TextContent(type="text", text=result)]

@server.list_resources()
async def list_resources() -> list[Resource]:
    """Return the list of available Axon Pro resources."""
    return [
        Resource(
            uri="axon-pro://overview",
            name="Codebase Overview",
            description="High-level statistics about the indexed codebase.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon-pro://dead-code",
            name="Dead Code Report",
            description="List of all symbols flagged as unreachable.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon-pro://schema",
            name="Graph Schema",
            description="Description of the Axon Pro knowledge graph schema.",
            mimeType="text/plain",
        ),
        Resource(
            uri="axon-pro://agent-guidelines",
            name="AI Agent Guidelines",
            description="Instructions for AI agents on how to use Axon Pro effectively.",
            mimeType="text/plain",
        ),
    ]

@server.read_resource()
async def read_resource(uri: Any) -> str:
    """Read a specific Axon Pro resource."""
    storage = _get_storage()
    uri_str = str(uri)

    if _lock is not None:
        async with _lock:
            return await asyncio.to_thread(_dispatch_resource, uri_str, storage)
    else:
        return _dispatch_resource(uri_str, storage)

def _dispatch_resource(uri_str: str, storage: KuzuBackend) -> str:
    """Synchronous resource dispatch."""
    if uri_str == "axon-pro://overview":
        return get_overview(storage)
    if uri_str == "axon-pro://dead-code":
        return get_dead_code_list(storage)
    if uri_str == "axon-pro://schema":
        return get_schema()
    if uri_str == "axon-pro://agent-guidelines":
        return get_agent_guidelines()
    return f"Unknown resource: {uri_str}"

async def main() -> None:
    """Run the Axon Pro MCP server over stdio transport."""
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
