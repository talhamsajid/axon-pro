# Axon Pro 🚀

[![Release](https://img.shields.io/github/v/release/talhamsajid/axon-pro?include_prereleases)](https://github.com/talhamsajid/axon-pro/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Homebrew](https://img.shields.io/badge/Install-Homebrew-orange.svg)](https://github.com/talhamsajid/homebrew-tap)

**The Enterprise-grade graph-powered code intelligence engine.** Axon Pro transforms flat codebases into structural knowledge graphs, providing AI agents and developers with 100% precise architectural awareness across polyglot microservices and modern web frameworks.

```bash
axon-pro analyze .

Phase 1:  Walking files...               1,242 files found
Phase 3:  Parsing code...                1,242/1,242
Phase 5:  Tracing calls...               8,472 calls resolved
Phase 7:  Analyzing types...             2,341 type relationships
Phase 9:  Detecting execution flows...   156 processes found
Phase 12: Linking Laravel Templates...   42 views connected

Done in 12.4s — 4,623 symbols, 15,847 edges, 156 flows
```

---

## 💎 Why Axon Pro?

**For AI Agents (Claude Code, Cursor, Windsurf):**
- **Deep Polyglot Support:** Understands cross-language dependencies (PHP, Python, Java, TS).
- **High-Fidelity Context:** Moves beyond "text similarity" to "structural truth."
- **Architectural Awareness:** AI agents can finally see the "blast radius" of their changes.
- **Flow Tracing:** Traces execution from a Route, through Middleware, into a Controller, and out to a View or Background Job.

**For Engineering Teams:**
- **Zero Cloud Footprint:** Runs locally. No data leakage, no third-party API costs.
- **Refactoring Safety:** Instant impact analysis for complex legacy codebases.
- **Onboarding Speed:** New developers can query the architecture using natural language or Cypher.
- **Technical Debt Audit:** Automatically identifies dead code, N+1 queries, and tight coupling.

---

## 🧠 Elite Laravel IQ

Axon Pro is the only code intelligence engine with native, semantic awareness of the Laravel ecosystem:

- **Eloquent Relationships:** Automatically maps `hasMany`, `belongsTo`, and polymorphic links.
- **Route-to-Code Mapping:** Links HTTP routes directly to their Controller methods.
- **Messaging Lifecycle:** Traces Events → Listeners and Job → Queue dispatches.
- **Security Audit:** Maps Middleware chains and connects actions to their Laravel Policies.
- **Blade Intelligence:** Bridges the gap between PHP logic and `.blade.php` templates.
- **Container Resolution:** Resolves Interface injections to their concrete Service Provider bindings.

---

## 🛠️ Installation

### Homebrew (Recommended)
```bash
brew install talhamsajid/tap/axon-pro
```

### With UV (Fastest)
```bash
uv tool install axon-pro[neo4j]
```

### With PIP
```bash
pip install axon-pro[neo4j]
```

---

## ⚡ Quick Start

### 1. Index Your Project
```bash
cd your-polyglot-repo
axon-pro analyze .
```

### 2. Run an Architectural Audit
Flag N+1 queries and boundary violations instantly.
```bash
axon-pro check
```

### 3. Get a Codebase Briefing
Generate a high-level "map" for your next coding session.
```bash
axon-pro brief
```

### 4. Deep Symbol Context
See callers, views rendered, and security policies for any class.
```bash
axon-pro context OrderController
```

---

## 🖼️ Enterprise Visualization (Neo4j)

Export your codebase graph to **Neo4j** for world-class interactive visualization using Neo4j Bloom or the Browser.

```bash
# Export to a local Neo4j Docker instance
axon-pro analyze . --neo4j --neo4j-url bolt://localhost:7687
```

**Cypher Audit Examples:**
```cypher
// Find all entry points protected by 'auth' middleware
MATCH (r:Route)-[:PROTECTED_BY]->(m:Middleware {name: 'auth'}) RETURN r.name

// View the blast radius of changing a specific Model
MATCH (m:Class {name: 'User'})-[:RELATIONSHIP_TO]->(target) RETURN m, target
```

---

## 🤖 AI Agent Integration (MCP)

Axon Pro exposes its intelligence via the **Model Context Protocol (MCP)**. Give your agent the structural "brain" it needs to stop making guessing errors.

```bash
axon-pro setup --claude
axon-pro setup --cursor
```

---

## 📜 Multi-Language Support

| Language | Extensions | Framework Awareness |
|----------|-----------|-------------------|
| **PHP** | `.php`, `.blade.php` | **Laravel**, Symfony |
| **Python** | `.py` | Flask, FastAPI, Django |
| **TypeScript** | `.ts`, `.tsx` | React, Next.js, NestJS |
| **JavaScript** | `.js`, `.jsx` | Express, Node.js |
| **Java** | `.java` | Spring Boot |
| **C#** | `.cs` | .NET Core, Unity |

---

## License

Enterprise-Ready / MIT

---

### Keywords & Use Cases
Code Intelligence, Knowledge Graph, Dependency Mapping, Static Analysis, N+1 Query Detection, Middleware Tracing, Eloquent Relationship Graph, MCP Server for AI Agents, Neo4j Visualization, Codebase Audit, Laravel Architecture, Polyglot Analysis.

---

**Axon Pro** is a product of **Technicodes**.  
*Empowering AI agents with structural intelligence.*  
[https://technicodes.com](https://technicodes.com)
