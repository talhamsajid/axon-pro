# Axon Pro

**The Enterprise-grade graph-powered code intelligence engine** — indexes your entire polyglot codebase into a knowledge graph and exposes it via MCP tools for AI agents and a CLI for developers.

```bash
axon analyze .

Phase 1:  Walking files...               1,242 files found
Phase 3:  Parsing code...                1,242/1,242
Phase 5:  Tracing calls...               8,472 calls resolved
Phase 7:  Analyzing types...             2,341 type relationships
Phase 8:  Detecting communities...       42 clusters found
Phase 9:  Detecting execution flows...   156 processes found
Phase 10: Finding dead code...           87 unreachable symbols
Phase 11: Analyzing git history...       148 coupled file pairs

Done in 12.4s — 4,623 symbols, 15,847 edges, 42 clusters, 156 flows
```

Axon Pro transforms your codebase from flat text into a **structural knowledge graph**. By parsing Python, TypeScript, JavaScript, **PHP (Laravel)**, **Java**, and **C# (.NET)**, it builds a deep understanding of how your services interact, how data flows, and where your architectural risks lie.

---

## Why Axon Pro?

**For AI Agents (Claude Code, Cursor, Windsurf):**
- **Deep Polyglot Support:** Understands cross-language dependencies in microservices.
- **Architectural Awareness:** "What breaks if I change this Java interface?" → Instant blast radius across the entire graph.
- **Flow Tracing:** "Show me the full request lifecycle from the PHP frontend to the Python backend."
- **Precision Search:** Hybrid search (BM25 + Vector + RRF) ensures the agent finds the *correct* symbol, not just the most similar text.

**For Engineering Teams:**
- **Zero Cloud Footprint:** Everything runs locally on your infrastructure. No data leakage, no API costs.
- **Refactoring Safety:** Automated impact analysis before you touch legacy code.
- **Onboarding Acceleration:** New devs can query the codebase structure using natural language or Cypher.
- **Technical Debt Management:** Automatically find dead code and tightly coupled files.

---

## Multi-Language Support

Axon Pro is built for the modern enterprise stack:

| Language | Extensions | Framework Support |
|----------|-----------|-------------------|
| **Python** | `.py` | Flask, FastAPI, Django, Click |
| **TypeScript** | `.ts`, `.tsx` | React, Next.js, NestJS |
| **JavaScript** | `.js`, `.jsx`, `.mjs`, `.cjs` | Express, Node.js |
| **PHP** | `.php` | Laravel, Symfony, WordPress |
| **Java** | `.java` | Spring Boot, Jakarta EE |
| **C#** | `.cs` | .NET Core, ASP.NET, Unity |

---

## Core Capabilities

### 11-Phase Analysis Pipeline
Axon Pro executes a deep structural audit of your repository:
1. **File Walking:** Respects `.gitignore` and enterprise-level ignore patterns.
2. **Structure:** Maps the File/Folder hierarchy with `CONTAINS` relationships.
3. **Parsing:** High-performance tree-sitter AST extraction for all supported languages.
4. **Import Resolution:** Resolves namespaces, imports, and use statements across files.
5. **Call Tracing:** Maps the dynamic call graph with multi-language confidence scoring.
6. **Heritage:** Tracks class inheritance (`EXTENDS`) and interface implementation (`IMPLEMENTS`).
7. **Type Analysis:** Extracts type references from signatures and variable annotations.
8. **Community Detection:** Uses the **Leiden algorithm** to cluster related code into functional modules.
9. **Process Detection:** Detects entry points (API routes, CLI commands) and traces execution flows.
10. **Dead Code Detection:** Intelligent multi-pass analysis that understands overrides and protocols.
11. **Change Coupling:** Analyzes Git history to find hidden dependencies between files.

---

## Installation

```bash
# With pip
pip install axon-pro

# With uv (recommended)
uv tool install axon-pro
```

Requires **Python 3.11+**.

---

## Quick Start

### 1. Index Your Codebase
```bash
cd your-polyglot-repo
axon analyze .
```

### 2. Deep Context Query
```bash
# Get a 360-degree view of a Spring Boot controller or Laravel service
axon context UserService
```

### 3. Impact Analysis
```bash
# See what breaks if you change a core C# interface
axon impact IRepository --depth 4
```

### 4. Live Watch Mode
```bash
# Keep the graph in sync with your edits
axon watch
```

---

## MCP Integration

Axon Pro is designed to be the "brain" for your AI coding assistants.

### Setup for AI Agents
```bash
axon setup --claude
axon setup --cursor
```

This adds Axon Pro as an MCP server, giving your agent access to:
- `axon_query`: Semantic and structural search.
- `axon_context`: Deep symbol relationship view.
- `axon_impact`: Automated blast-radius analysis.
- `axon_dead_code`: Technical debt identification.
- `axon_cypher`: Direct graph querying for complex architectural questions.

---

## License

Enterprise-Ready / MIT

---

**Axon Pro** is a product of **Technicodes**.
Empowering AI agents with structural intelligence.
[https://technicodes.com](https://technicodes.com)
