# Axon Pro: Enterprise Code Intelligence

Axon Pro is a graph-powered code intelligence engine designed for modern polyglot microservices and large-scale Laravel applications. It transforms your codebase into a structural knowledge graph, enabling AI agents and developers to navigate architecture with 100% precision.

## Core Features

- **Polyglot Analysis:** Native support for PHP, Python, JS/TS, Java, and C#.
- **Laravel Awareness:** Deep structural understanding of Eloquent, Routes, Jobs, Events, Observers, and Blade Templates.
- **Architectural Linting:** Proactively find N+1 queries and violations of architectural boundaries.
- **MCP Integration:** Acts as a high-fidelity "brain" for AI agents (Claude Code, Cursor, Windsurf).
- **Dual Storage:** Uses KuzuDB for local speed and Neo4j for enterprise-wide visualization.

## CLI Commands

### `axon-pro analyze [PATH]`
Indexes the repository.
- `--neo4j`: Export the graph to a Neo4j server.
- `--full`: Force a complete re-index.

### `axon-pro brief`
Generates a high-level architectural summary of the application (Core models, Primary routes, Async jobs).

### `axon-pro check`
Runs an architectural audit.
- Flags direct DB access in Controllers.
- Detects potential N+1 query bottlenecks in loops.

### `axon-pro context [SYMBOL]`
Provides a 360-degree view of a class or method, including callers, callees, and Laravel-specific links (e.g., what view it renders).

### `axon-pro impact [SYMBOL]`
Performs automated blast-radius analysis to see what will break if you change a symbol.

### `axon-pro serve --watch`
Starts the MCP server and keeps the graph in sync with your local edits.

## Cypher Query Examples (Neo4j)

**Find all Controllers that render a specific View:**
```cypher
MATCH (c:Class)-[:RENDERS]->(v:View {name: 'dashboard'})
RETURN c.name
```

**Find all Jobs dispatched by a specific Controller:**
```cypher
MATCH (c:Class {name: 'OrderController'})-[:DISPATCHES]->(j:Job)
RETURN j.name
```

**Audit: Routes without Middleware protection:**
```cypher
MATCH (r:Route)
WHERE NOT (r)-[:PROTECTED_BY]->(:Middleware)
RETURN r.name
```

---
**Axon Pro** is a product of **Technicodes**.
[https://technicodes.com](https://technicodes.com)
