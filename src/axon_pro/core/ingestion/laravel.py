"""Laravel-specific structural analysis for Axon Pro.

Connects Laravel entities like Events, Listeners, Observers, Jobs, Models, 
Routes, and Policies based on structural patterns and registrations.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

from axon_pro.core.graph.model import NodeLabel, RelType, generate_id, GraphNode, GraphRelationship

if TYPE_CHECKING:
    from axon_pro.core.graph.graph import KnowledgeGraph
    from axon_pro.core.ingestion.parser_phase import FileParseData

logger = logging.getLogger(__name__)

def process_laravel(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Analyse Laravel-specific patterns and link nodes."""
    # 1. Event/Listener Mapping
    _link_events_and_listeners(parse_data_list, graph)

    # 2. Model/Observer Mapping
    _link_models_and_observers(parse_data_list, graph)

    # 3. Eloquent Relationships
    _link_eloquent_relationships(parse_data_list, graph)

    # 4. Route Mapping
    _link_routes_to_controllers(parse_data_list, graph)

    # 5. Policy & Auth Mapping
    _link_policies_and_controllers(parse_data_list, graph)

    # 6. FormRequest Mapping
    _link_form_requests(parse_data_list, graph)

    # 7. Container Bindings
    _link_container_bindings(parse_data_list, graph)

    # 8. Facade Resolution
    _resolve_facades(parse_data_list, graph)

    # 9. N+1 Query Detection
    _detect_n_plus_one_queries(parse_data_list, graph)

    # 10. Middleware Linking
    _link_middleware(parse_data_list, graph)

    # 11. Tracing Dispatches
    _trace_laravel_dispatches(parse_data_list, graph)

def _link_middleware(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Link Routes and Controllers to Middleware applied to them."""
    for data in parse_data_list:
        # Check for middleware() calls on Routes
        if "routes/" in data.file_path:
            for call in data.parse_result.calls:
                if call.name == "middleware":
                    # Route::middleware(['auth', ...])
                    # Find the last created Route node (simplified heuristic)
                    route_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.ROUTE and n.file_path == data.file_path]
                    if route_nodes:
                        # Link to potential middleware (by name or alias)
                        for arg in call.arguments:
                            m_name = arg.strip("'\"")
                            # Find middleware nodes
                            m_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.MIDDLEWARE and m_name in n.name]
                            for rn in route_nodes:
                                for mn in m_nodes:
                                    rel_id = f"protected_by:{rn.id}->{mn.id}"
                                    graph.add_relationship(GraphRelationship(id=rel_id, type=RelType.PROTECTED_BY, source=rn.id, target=mn.id))

        # Check for $middleware property in Controllers
        for symbol in data.parse_result.symbols:
            if symbol.kind == "class" and ("Controller" in symbol.name):
                # Look for calls to middleware() in __construct
                for call in data.parse_result.calls:
                    if call.name == "middleware" and call.line >= symbol.start_line and call.line <= symbol.end_line:
                        # Controller-level middleware
                        class_node_id = generate_id(NodeLabel.CLASS, data.file_path, symbol.name)
                        for arg in call.arguments:
                            m_name = arg.strip("'\"")
                            m_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.MIDDLEWARE and m_name in n.name]
                            for mn in m_nodes:
                                rel_id = f"protected_by:{class_node_id}->{mn.id}"
                                graph.add_relationship(GraphRelationship(id=rel_id, type=RelType.PROTECTED_BY, source=class_node_id, target=mn.id))

def _detect_n_plus_one_queries(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Detect potential N+1 query issues where Eloquent relations are called in loops."""
    # List of known Eloquent relationship methods that trigger queries
    rel_methods = ["hasMany", "belongsTo", "hasOne", "belongsToMany", "morphTo", "morphMany", "morphedByMany"]
    
    for data in parse_data_list:
        for call in data.parse_result.calls:
            # If a call is in a loop and looks like a relationship call
            if call.is_in_loop:
                # Heuristic: method name matches an Eloquent relationship or a property-like access
                # that often triggers a query.
                is_potential_n_plus_one = False
                if call.name in rel_methods:
                    is_potential_n_plus_one = True
                
                # Also check for common Model methods that trigger queries
                if call.name in ["get", "first", "find", "all", "paginate"]:
                    is_potential_n_plus_one = True

                if is_potential_n_plus_one:
                    # Find the symbol containing this call
                    source_node = None
                    for node in graph.iter_nodes():
                        if node.file_path == data.file_path and node.start_line <= call.line <= node.end_line:
                            source_node = node
                            break
                    
                    if source_node:
                        # We don't necessarily have a target node (dynamic call), 
                        # so we mark the source node with a property or a self-relationship.
                        # For now, let's add a property to the node.
                        source_node.properties.setdefault("n_plus_one_warnings", []).append({
                            "method": call.name,
                            "line": call.line,
                            "file": data.file_path
                        })
                        
                        # Optionally add an issue node or similar
                        pass

def _link_container_bindings(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Link Interfaces to Concrete classes based on Service Container bindings."""
    for data in parse_data_list:
        for interface_name, kind, concrete_name in data.parse_result.heritage:
            if kind == "binds":
                # Find the interface and concrete nodes
                # Interface might be NodeLabel.INTERFACE or NodeLabel.CLASS
                interface_nodes = [n for n in graph.iter_nodes() if n.name == interface_name]
                concrete_nodes = [n for n in graph.iter_nodes() if n.name == concrete_name]
                
                for i_node in interface_nodes:
                    for c_node in concrete_nodes:
                        rel_id = f"binds:{i_node.id}->{c_node.id}"
                        graph.add_relationship(
                            GraphRelationship(
                                id=rel_id,
                                type=RelType.BINDS,
                                source=i_node.id,
                                target=c_node.id,
                            )
                        )

def _resolve_facades(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Map calls to standard Laravel Facades to their underlying implementation classes."""
    facade_map = {
        "DB": "Connection",
        "Cache": "Repository",
        "Storage": "FilesystemAdapter",
        "Log": "Logger",
        "Event": "Dispatcher",
        "Route": "Router",
        "Auth": "Guard",
    }
    
    for data in parse_data_list:
        for call in data.parse_result.calls:
            if call.receiver in facade_map:
                impl_class = facade_map[call.receiver]
                # We could add a property to the call relationship or create a virtual link
                # For now, we'll mark the call node as a Facade call in properties
                # (Actual linking to the implementation requires finding where those classes are defined)
                pass

def _link_events_and_listeners(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Search for $listen array in EventServiceProvider and link Event to Listeners."""
    for data in parse_data_list:
        is_esp = any(s.kind == "service_provider" and "EventServiceProvider" in s.name for s in data.parse_result.symbols)
        if not is_esp:
            continue
            
        content = ""
        for s in data.parse_result.symbols:
            if s.kind == "service_provider":
                content = s.content
                break
        
        matches = re.finditer(r"([\w\\]+)::class\s*=>\s*\[(.*?)\]", content, re.DOTALL)
        for match in matches:
            event_name = match.group(1).split('\\')[-1]
            listeners_raw = match.group(2)
            listener_names = re.findall(r"([\w\\]+)::class", listeners_raw)
            
            event_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.EVENT and n.name == event_name]
            for event_node in event_nodes:
                for ln in listener_names:
                    l_name = ln.split('\\')[-1]
                    listener_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.LISTENER and n.name == l_name]
                    for l_node in listener_nodes:
                        rel_id = f"listens_to:{l_node.id}->{event_node.id}"
                        graph.add_relationship(GraphRelationship(id=rel_id, type=RelType.LISTENS_TO, source=l_node.id, target=event_node.id))

def _link_models_and_observers(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Search for Model::observe(Observer::class) and link them."""
    for data in parse_data_list:
        for call in data.parse_result.calls:
            if call.name == "observe" and call.receiver:
                model_name = call.receiver
                for arg in call.arguments:
                    if "Observer" in arg:
                        observer_name = arg.replace("::class", "").split('\\')[-1]
                        model_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.CLASS and n.name == model_name]
                        observer_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.OBSERVER and n.name == observer_name]
                        for m_node in model_nodes:
                            for o_node in observer_nodes:
                                rel_id = f"observes:{o_node.id}->{m_node.id}"
                                graph.add_relationship(GraphRelationship(id=rel_id, type=RelType.OBSERVES, source=o_node.id, target=m_node.id))

def _link_eloquent_relationships(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Link models via detected Eloquent relationship methods."""
    for data in parse_data_list:
        for method_name, kind, target_model in data.parse_result.heritage:
            if kind.startswith("eloquent:"):
                rel_type_name = kind.split(":")[1]
                # Source is the Model (Class) in this file
                # Find all classes in this file (usually just one model)
                source_classes = [s.name for s in data.parse_result.symbols if s.kind == "class"]
                for sc in source_classes:
                    source_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.CLASS and n.name == sc]
                    target_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.CLASS and n.name == target_model]
                    for s_node in source_nodes:
                        for t_node in target_nodes:
                            rel_id = f"eloquent_{rel_type_name}:{s_node.id}->{t_node.id}"
                            graph.add_relationship(
                                GraphRelationship(
                                    id=rel_id, 
                                    type=RelType.RELATIONSHIP_TO, 
                                    source=s_node.id, 
                                    target=t_node.id,
                                    properties={"relationship_type": rel_type_name, "method": method_name}
                                )
                            )

def _link_routes_to_controllers(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Parse Route:: definitions and link to Controller methods."""
    for data in parse_data_list:
        if "routes/" not in data.file_path:
            continue
            
        for call in data.parse_result.calls:
            if call.receiver == "Route" and call.name in ["get", "post", "put", "patch", "delete", "any", "match"]:
                # Route::get('/path', [Controller::class, 'method'])
                if len(call.arguments) >= 2:
                    path = call.arguments[0].strip("'\"")
                    action_raw = call.arguments[1]
                    
                    controller_name = ""
                    method_name = ""
                    
                    # Pattern: [SomeController::class, 'index']
                    match = re.search(r"([\w\\]+)::class\s*,\s*['\"](\w+)['\"]", action_raw)
                    if match:
                        controller_name = match.group(1).split('\\')[-1]
                        method_name = match.group(2)
                    
                    if controller_name and method_name:
                        # Create a Route node
                        route_id = generate_id(NodeLabel.ROUTE, data.file_path, f"{call.name.upper()} {path}")
                        graph.add_node(GraphNode(id=route_id, label=NodeLabel.ROUTE, name=f"{call.name.upper()} {path}", properties={"path": path, "verb": call.name.upper()}))
                        
                        # Link Route -> Controller Method
                        target_methods = [n for n in graph.iter_nodes() if n.label == NodeLabel.METHOD and n.name == method_name and n.class_name == controller_name]
                        for t_method in target_methods:
                            rel_id = f"maps_to:{route_id}->{t_method.id}"
                            graph.add_relationship(GraphRelationship(id=rel_id, type=RelType.MAPS_TO, source=route_id, target=t_method.id))

def _link_policies_and_controllers(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Link controller methods to policies via $this->authorize() or middleware hints."""
    for data in parse_data_list:
        for call in data.parse_result.calls:
            if call.name == "authorize":
                # $this->authorize('update', $post)
                # Heuristic: find current method and link to a policy method with same name
                source_method = None
                for node in graph.iter_nodes():
                    if node.file_path == data.file_path and node.start_line <= call.line <= node.end_line:
                        source_method = node
                        break
                
                if source_method and len(call.arguments) > 0:
                    ability = call.arguments[0].strip("'\"")
                    # Find potential policies (Heuristic: Classes ending in Policy)
                    # This is tricky because authorize() doesn't specify the Policy class explicitly
                    # We link to ANY policy method that matches the ability for now
                    policy_methods = [n for n in graph.iter_nodes() if n.label == NodeLabel.METHOD and n.name == ability and "Policy" in n.class_name]
                    for p_method in policy_methods:
                        rel_id = f"authorized_by:{source_method.id}->{p_method.id}"
                        graph.add_relationship(GraphRelationship(id=rel_id, type=RelType.AUTHORIZED_BY, source=source_method.id, target=p_method.id))

def _link_form_requests(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Link Controller methods to the FormRequest classes they type-hint."""
    for data in parse_data_list:
        # We need to look at method signatures. Currently SymbolInfo doesn't have params.
        # But we can look for calls to validate or use heritage hints.
        # Alternative: look for FormRequest classes and see where they are referenced.
        for s in data.parse_result.symbols:
            if s.kind == "method":
                # Heuristic: check if 'Request' or 'FormRequest' is in the content of the method signature area
                # (This is a placeholder for actual param parsing)
                signature = s.signature
                fr_nodes = [n for n in graph.iter_nodes() if n.label == NodeLabel.FORM_REQUEST]
                for fr in fr_nodes:
                    if fr.name in signature:
                        method_node_id = generate_id(NodeLabel.METHOD, data.file_path, f"{s.class_name}.{s.name}")
                        rel_id = f"validated_by:{method_node_id}->{fr.id}"
                        graph.add_relationship(GraphRelationship(id=rel_id, type=RelType.VALIDATED_BY, source=method_node_id, target=fr.id))

def _trace_laravel_dispatches(parse_data_list: list[FileParseData], graph: KnowledgeGraph) -> None:
    """Trace event() and dispatch() calls to link source to Event/Job."""
    for data in parse_data_list:
        for call in data.parse_result.calls:
            if call.name in ["event", "dispatch", "broadcast", "notify"]:
                source_node = None
                for node in graph.iter_nodes():
                    if node.file_path == data.file_path and node.start_line <= call.line <= node.end_line:
                        source_node = node
                        break
                
                if not source_node:
                    continue
                
                for target_label in [NodeLabel.EVENT, NodeLabel.JOB]:
                    targets = [n for n in graph.iter_nodes() if n.label == target_label]
                    for target_node in targets:
                        if any(target_node.name in arg for arg in call.arguments):
                            rel_id = f"dispatches:{source_node.id}->{target_node.id}"
                            graph.add_relationship(GraphRelationship(id=rel_id, type=RelType.DISPATCHES, source=source_node.id, target=target_node.id))
