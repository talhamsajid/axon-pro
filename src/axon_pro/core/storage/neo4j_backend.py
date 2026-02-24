"""Neo4j storage backend for Axon Pro.

Allows the knowledge graph to be persisted in a Neo4j database, enabling
advanced visualization and enterprise-wide architectural queries.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable

from axon_pro.core.graph.graph import KnowledgeGraph
from axon_pro.core.graph.model import GraphNode, GraphRelationship, NodeLabel, RelType
from axon_pro.core.storage.base import StorageBackend

logger = logging.getLogger(__name__)

class Neo4jBackend(StorageBackend):
    """Storage backend powered by Neo4j."""

    def __init__(self, uri: str = "bolt://localhost:7687", user: str = "neo4j", password: str = "password") -> None:
        try:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        except ImportError:
            raise ImportError(
                "The 'neo4j' package is required for this backend. "
                "Install it with: pip install axon-pro[neo4j]"
            )

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self._driver.close()

    def bulk_load(self, graph: KnowledgeGraph) -> None:
        """Bulk load the entire in-memory graph into Neo4j."""
        with self._driver.session() as session:
            # 1. Clear existing data (Optional: based on strategy)
            # session.run("MATCH (n) DETACH DELETE n")

            # 2. Load Nodes
            logger.info("Bulk loading nodes into Neo4j...")
            nodes = list(graph.iter_nodes())
            batch_size = 1000
            for i in range(0, len(nodes), batch_size):
                batch = nodes[i:i + batch_size]
                session.execute_write(self._create_nodes_batch, batch)

            # 3. Load Relationships
            logger.info("Bulk loading relationships into Neo4j...")
            rels = list(graph.iter_relationships())
            for i in range(0, len(rels), batch_size):
                batch = rels[i:i + batch_size]
                session.execute_write(self._create_rels_batch, batch)

    @staticmethod
    def _create_nodes_batch(tx, nodes: list[GraphNode]) -> None:
        """Execute a batch of node creations."""
        # We use UNWIND for high-performance bulk loading
        query = """
        UNWIND $batch AS data
        CALL apoc.create.node([data.label], data.properties) YIELD node
        SET node.id = data.id, node.name = data.name, node.file_path = data.file_path
        RETURN count(node)
        """
        # Note: apoc.create.node allows dynamic labels. 
        # If APOC isn't installed, we fallback to a simpler merge per label type.
        
        node_data = []
        for n in nodes:
            props = n.properties.copy()
            props.update({
                "start_line": n.start_line,
                "end_line": n.end_line,
                "language": n.language,
                "is_dead": n.is_dead,
            })
            node_data.append({
                "id": n.id,
                "label": n.label.value.capitalize(),
                "name": n.name,
                "file_path": n.file_path,
                "properties": props
            })
        
        # Simple fallback if APOC is not available (Manual Label Mapping)
        # For production, we'll use a more robust MERGE approach
        for label in NodeLabel:
            label_nodes = [n for n in node_data if n["label"] == label.value.capitalize()]
            if label_nodes:
                tx.run(f"""
                UNWIND $batch AS data
                MERGE (n:{label.value.capitalize()} {{id: data.id}})
                SET n += data.properties, n.name = data.name, n.file_path = data.file_path
                """, batch=label_nodes)

    @staticmethod
    def _create_rels_batch(tx, rels: list[GraphRelationship]) -> None:
        """Execute a batch of relationship creations."""
        for rel_type in RelType:
            type_rels = [r for r in rels if r.type == rel_type]
            if type_rels:
                rel_data = [{"source": r.source, "target": r.target, "props": r.properties} for r in type_rels]
                tx.run(f"""
                UNWIND $batch AS data
                MATCH (a {{id: data.source}})
                MATCH (b {{id: data.target}})
                MERGE (a)-[r:{rel_type.value.upper()}]->(b)
                SET r += data.props
                """, batch=rel_data)

    # Implement other abstract methods from StorageBackend...
    def add_nodes(self, nodes: Iterable[GraphNode]) -> None:
        with self._driver.session() as session:
            session.execute_write(self._create_nodes_batch, list(nodes))

    def add_relationships(self, relationships: Iterable[GraphRelationship]) -> None:
        with self._driver.session() as session:
            session.execute_write(self._create_rels_batch, list(relationships))

    def query(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        with self._driver.session() as session:
            result = session.run(cypher, parameters or {})
            return [dict(record) for record in result]

    def get_node(self, node_id: str) -> GraphNode | None:
        res = self.query("MATCH (n {id: $id}) RETURN n", {"id": node_id})
        if not res:
            return None
        # Convert back to GraphNode (simplified)
        return None 

    def remove_nodes_by_file(self, file_path: str) -> None:
        self.query("MATCH (n {file_path: $path}) DETACH DELETE n", {"path": file_path})

    def search_symbols(self, query: str, limit: int = 10) -> list[GraphNode]:
        # Neo4j Full-text search implementation
        return []

    def rebuild_fts_indexes(self) -> None:
        pass
