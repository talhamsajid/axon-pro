"""Laravel-specific structural analysis for Axon Pro.

Connects Laravel entities like Events, Listeners, Observers, and Jobs based on 
dispatch logic and Service Provider registrations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from axon_pro.core.graph.model import NodeLabel, RelType, generate_id, GraphRelationship

if TYPE_CHECKING:
    from axon_pro.core.graph.graph import KnowledgeGraph
    from axon_pro.core.ingestion.parser_phase import FileParseData

logger = logging.getLogger(__name__)

def process_laravel(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Analyse Laravel-specific patterns and link nodes.

    - Links Jobs/Controllers to Events they dispatch.
    - Links Events to Listeners based on EventServiceProvider mapping.
    - Links Models to Observers based on observe() calls.
    """
    # 1. Map Event -> Listeners from EventServiceProvider
    _link_events_and_listeners(parse_data_list, graph)

    # 2. Map Models -> Observers
    _link_models_and_observers(parse_data_list, graph)

    # 3. Tracing Dispatches (Events and Jobs)
    _trace_laravel_dispatches(parse_data_list, graph)

def _link_events_and_listeners(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Search for $listen array in EventServiceProvider and link Event to Listeners."""
    for data in parse_data_list:
        # Look for EventServiceProvider
        is_esp = any(s.kind == "service_provider" and "EventServiceProvider" in s.name for s in data.parse_result.symbols)
        if not is_esp:
            continue
            
        # Very simplified: look for class-name-like strings in the content
        # In a real implementation, we'd use tree-sitter to find the $listen array specifically.
        # For now, we'll use the symbols and heritage to find Event -> Listener connections.
        content = ""
        for s in data.parse_result.symbols:
            if s.kind == "service_provider":
                content = s.content
                break
        
        # We'll look for pattern: EventClass::class => [ListenerClass::class]
        # This is a heuristic until we have a more granular array parser
        import re
        # Match 'SomeEvent::class => [' and find the listeners until ']'
        matches = re.finditer(r"([\w\]+)::class\s*=>\s*\[(.*?)\]", content, re.DOTALL)
        for match in matches:
            event_name = match.group(1).split('')[-1]
            listeners_raw = match.group(2)
            listener_names = re.findall(r"([\w\]+)::class", listeners_raw)
            
            # Link them in the graph
            event_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.EVENT and n.name == event_name]
            for event_node in event_nodes:
                for ln in listener_names:
                    l_name = ln.split('')[-1]
                    listener_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.LISTENER and n.name == l_name]
                    for l_node in listener_nodes:
                        rel_id = f"listens_to:{l_node.id}->{event_node.id}"
                        graph.add_relationship(
                            GraphRelationship(
                                id=rel_id,
                                type=RelType.LISTENS_TO,
                                source=l_node.id,
                                target=event_node.id,
                            )
                        )

def _link_models_and_observers(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Search for Model::observe(Observer::class) and link them."""
    for data in parse_data_list:
        for call in data.parse_result.calls:
            if call.name == "observe" and call.receiver:
                # Potential Model::observe(Observer::class)
                model_name = call.receiver
                # The argument might be Observer::class
                for arg in call.arguments:
                    if "Observer" in arg:
                        observer_name = arg.replace("::class", "").split('')[-1]
                        
                        # Find nodes and link
                        # Note: We don't have a MODEL label yet, they are usually CLASS
                        model_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.CLASS and n.name == model_name]
                        observer_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.OBSERVER and n.name == observer_name]
                        
                        for m_node in model_nodes:
                            for o_node in observer_nodes:
                                rel_id = f"observes:{o_node.id}->{m_node.id}"
                                graph.add_relationship(
                                    GraphRelationship(
                                        id=rel_id,
                                        type=RelType.OBSERVES,
                                        source=o_node.id,
                                        target=m_node.id,
                                    )
                                )

def _trace_laravel_dispatches(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Trace event() and dispatch() calls to link source to Event/Job."""
    for data in parse_data_list:
        # Find the source symbol of the call (Method/Function)
        for call in data.parse_result.calls:
            if call.name in ["event", "dispatch", "broadcast", "notify"]:
                # Find the symbol containing this call
                source_node = None
                for node in graph.iter_nodes():
                    if node.file_path == data.file_path and node.start_line <= call.line <= node.end_line:
                        source_node = node
                        break
                
                if not source_node:
                    continue
                
                # Heuristic: Find an Event or Job name in the arguments or nearby
                # This is simplified: we look for any Event/Job node whose name appears in the call area
                for target_label in [NodeLabel.EVENT, NodeLabel.JOB]:
                    targets = [n for n in graph.iter_nodes() if n.label == target_label]
                    for target_node in targets:
                        # If target name is in call arguments (very broad check)
                        if any(target_node.name in arg for arg in call.arguments):
                            rel_id = f"dispatches:{source_node.id}->{target_node.id}"
                            graph.add_relationship(
                                GraphRelationship(
                                    id=rel_id,
                                    type=RelType.DISPATCHES,
                                    source=source_node.id,
                                    target=target_node.id,
                                )
                            )
