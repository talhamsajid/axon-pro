"""PHP language parser using tree-sitter.

Extracts functions, classes, methods, namespaces, and inheritance relationships
from PHP source code, with specific support for Laravel structures.
"""

from __future__ import annotations

import tree_sitter_php as tsphp
from tree_sitter import Language, Node, Parser

from axon_pro.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
)

PHP_LANGUAGE = Language(tsphp.language_php())

class PHPParser(LanguageParser):
    """Parses PHP source code using tree-sitter with Laravel awareness."""

    def __init__(self) -> None:
        self._parser = Parser(PHP_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse PHP source and return structured information."""
        tree = self._parser.parse(bytes(content, "utf8"))
        result = ParseResult()
        root = tree.root_node
        self._walk(root, content, result, class_name="")
        return result

    def _walk(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Recursively walk the AST to extract definitions."""
        for child in node.children:
            match child.type:
                case "function_definition":
                    self._extract_function(child, content, result, class_name)
                case "class_declaration":
                    self._extract_class(child, content, result)
                case "interface_declaration":
                    self._extract_interface(child, content, result)
                case "method_declaration":
                    self._extract_method(child, content, result, class_name)
                case "namespace_definition":
                    self._walk(child, content, result, class_name)
                case "namespace_use_declaration":
                    self._extract_import(child, result)
                case "function_call_expression":
                    self._extract_call(child, result)
                case "member_call_expression":
                    self._extract_member_call(child, result)
                case "scoped_call_expression":
                    self._extract_scoped_call(child, result)
                case _:
                    self._walk(child, content, result, class_name)

    def _extract_function(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a function definition."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]
        
        # Extract signature (parameters)
        params_node = node.child_by_field_name("parameters")
        signature = params_node.text.decode("utf8") if params_node else ""

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="function",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                signature=signature,
                class_name=class_name,
            )
        )

    def _extract_method(
        self,
        node: Node,
        content: str,
        result: ParseResult,
        class_name: str,
    ) -> None:
        """Extract a method definition."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]
        
        # Extract signature (parameters)
        params_node = node.child_by_field_name("parameters")
        signature = params_node.text.decode("utf8") if params_node else ""

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                signature=signature,
                class_name=class_name,
            )
        )

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Extract a class definition and its contents with Laravel awareness."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        class_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        # Determine Laravel kind
        kind = "class"
        
        # Check base class/interfaces/traits for Laravel hints
        extends_clause = node.child_by_field_name("extends")
        implements_clause = node.child_by_field_name("implements")
        
        parents = []
        if extends_clause:
            for child in extends_clause.children:
                if child.type == "name":
                    parents.append(child.text.decode("utf8"))
        
        if implements_clause:
            for child in implements_clause.children:
                if child.type == "name":
                    parents.append(child.text.decode("utf8"))

        # Laravel Heuristics
        is_sp = False
        if any(p in ["Migration", "Schema"] for p in parents) or "Migration" in class_name:
            kind = "migration"
        elif any(p in ["Command", "Job", "ShouldQueue"] for p in parents) or class_name.endswith("Job") or class_name.endswith("Command"):
            kind = "job" if "Job" in class_name or "ShouldQueue" in parents else "command"
        elif class_name.endswith("Observer"):
            kind = "observer"
        elif class_name.endswith("Event"):
            kind = "event"
        elif class_name.endswith("Listener"):
            kind = "listener"
        elif class_name.endswith("Policy"):
            kind = "policy"
        elif class_name.endswith("Request") or "FormRequest" in parents:
            kind = "form_request"
        elif class_name.endswith("ServiceProvider"):
            kind = "service_provider"
            is_sp = True

        result.symbols.append(
            SymbolInfo(
                name=class_name,
                kind=kind,
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            )
        )

        # Eloquent Relationship Detection (simplified)
        # We look for methods in classes that might be Models (heuristic)
        is_model = "Model" in parents or any("Eloquent" in p for p in parents)
        
        body = node.child_by_field_name("body")
        if body:
            if is_model:
                self._extract_eloquent_relationships(body, content, result)
            if is_sp:
                self._extract_container_bindings(body, content, result)
            self._walk(body, content, result, class_name=class_name)

    def _extract_container_bindings(self, body: Node, content: str, result: ParseResult) -> None:
        """Extract Service Container bindings like $this->app->bind()."""
        # Look for $this->app->bind(Interface::class, Concrete::class)
        for method in body.children:
            if method.type == "method_declaration":
                method_text = content[method.start_byte:method.end_byte]
                binding_methods = ["bind", "singleton", "scoped", "instance"]
                for bm in binding_methods:
                    if f"->{bm}(" in method_text:
                        import re
                        # Pattern: ->bind(SomeInterface::class, SomeConcrete::class)
                        match = re.search(fr"->{bm}\(([\w\\]+)::class\s*,\s*([\w\\]+)::class", method_text)
                        if match:
                            interface = match.group(1).split('\\')[-1]
                            concrete = match.group(2).split('\\')[-1]
                            result.heritage.append((interface, "binds", concrete))

    def _extract_eloquent_relationships(self, body: Node, content: str, result: ParseResult) -> None:
        """Extract Eloquent relationship methods like hasMany, belongsTo."""
        # We look for return statements in methods that call relationship functions
        for method in body.children:
            if method.type == "method_declaration":
                # Find return $this->hasMany(...)
                method_text = content[method.start_byte:method.end_byte]
                rel_types = ["hasMany", "belongsTo", "hasOne", "belongsToMany", "morphTo", "morphMany", "morphedByMany"]
                for rel_type in rel_types:
                    if f"->{rel_type}(" in method_text:
                        # Extract the target model name from the arguments
                        import re
                        match = re.search(fr"->{rel_type}\(([\w\\]+)::class", method_text)
                        if match:
                            target_model = match.group(1).split('\\')[-1]
                            # We store this in heritage for now with a special kind
                            method_name = method.child_by_field_name("name").text.decode("utf8")
                            result.heritage.append((method_name, f"eloquent:{rel_type}", target_model))

    def _extract_interface(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Extract an interface definition."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        interface_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        result.symbols.append(
            SymbolInfo(
                name=interface_name,
                kind="interface",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            )
        )

        body = node.child_by_field_name("body")
        if body:
            self._walk(body, content, result, class_name=interface_name)

    def _extract_import(self, node: Node, result: ParseResult) -> None:
        """Extract a namespace use declaration."""
        for child in node.children:
            if child.type == "namespace_use_clause":
                name_node = child.child_by_field_name("name")
                if name_node:
                    module = name_node.text.decode("utf8")
                    parts = module.split("\\")
                    result.imports.append(
                        ImportInfo(
                            module=module,
                            names=[parts[-1]],
                        )
                    )

    def _extract_call(self, node: Node, result: ParseResult) -> None:
        """Extract a function call."""
        name_node = node.child_by_field_name("function")
        args_node = node.child_by_field_name("arguments")
        if name_node:
            name = name_node.text.decode("utf8")
            args = []
            if args_node:
                for arg in args_node.children:
                    if arg.type in ["argument", "name", "qualified_name", "class_constant_access"]:
                        args.append(arg.text.decode("utf8"))
            
            result.calls.append(
                CallInfo(
                    name=name,
                    line=node.start_point[0] + 1,
                    arguments=args
                )
            )

    def _extract_member_call(self, node: Node, result: ParseResult) -> None:
        """Extract a member call ($obj->method())."""
        name_node = node.child_by_field_name("name")
        obj_node = node.child_by_field_name("object")
        args_node = node.child_by_field_name("arguments")
        if name_node:
            name = name_node.text.decode("utf8")
            receiver = obj_node.text.decode("utf8") if obj_node else ""
            args = []
            if args_node:
                for arg in args_node.children:
                    if arg.type in ["argument", "name", "qualified_name", "class_constant_access"]:
                        args.append(arg.text.decode("utf8"))
            
            result.calls.append(
                CallInfo(
                    name=name,
                    line=node.start_point[0] + 1,
                    receiver=receiver,
                    arguments=args
                )
            )

    def _extract_scoped_call(self, node: Node, result: ParseResult) -> None:
        """Extract a scoped call (Class::method())."""
        name_node = node.child_by_field_name("name")
        scope_node = node.child_by_field_name("scope")
        args_node = node.child_by_field_name("arguments")
        if name_node:
            name = name_node.text.decode("utf8")
            receiver = scope_node.text.decode("utf8") if scope_node else ""
            args = []
            if args_node:
                for arg in args_node.children:
                    if arg.type in ["argument", "name", "qualified_name", "class_constant_access"]:
                        args.append(arg.text.decode("utf8"))
            
            result.calls.append(
                CallInfo(
                    name=name,
                    line=node.start_point[0] + 1,
                    receiver=receiver,
                    arguments=args
                )
            )
