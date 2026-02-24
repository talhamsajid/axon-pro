"""C# language parser using tree-sitter.

Extracts functions, classes, methods, and inheritance relationships
from C# source code.
"""

from __future__ import annotations

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Node, Parser

from axon_pro.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
)

CSHARP_LANGUAGE = Language(tscsharp.language())

class CSharpParser(LanguageParser):
    """Parses C# source code using tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(CSHARP_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse C# source and return structured information."""
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
                case "class_declaration":
                    self._extract_class(child, content, result)
                case "interface_declaration":
                    self._extract_interface(child, content, result)
                case "method_declaration":
                    self._extract_method(child, content, result, class_name)
                case "using_directive":
                    self._extract_import(child, result)
                case "invocation_expression":
                    self._extract_call(child, result)
                case "namespace_declaration" | "file_scoped_namespace_declaration":
                    self._walk(child, content, result, class_name)
                case _:
                    self._walk(child, content, result, class_name)

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

        result.symbols.append(
            SymbolInfo(
                name=name,
                kind="method",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
                class_name=class_name,
            )
        )

    def _extract_class(
        self,
        node: Node,
        content: str,
        result: ParseResult,
    ) -> None:
        """Extract a class definition and its contents."""
        name_node = node.child_by_field_name("name")
        if name_node is None:
            return

        class_name = name_node.text.decode("utf8")
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        node_content = content[node.start_byte : node.end_byte]

        result.symbols.append(
            SymbolInfo(
                name=class_name,
                kind="class",
                start_line=start_line,
                end_line=end_line,
                content=node_content,
            )
        )

        # Handle inheritance
        base_list = node.child_by_field_name("base_list")
        if base_list:
            for child in base_list.children:
                if child.type == "identifier":
                    result.heritage.append((class_name, "extends", child.text.decode("utf8")))

        body = node.child_by_field_name("body")
        if body:
            self._walk(body, content, result, class_name=class_name)

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
        """Extract a using directive."""
        name_node = node.child_by_field_name("name")
        if name_node:
            module = name_node.text.decode("utf8")
            parts = module.split(".")
            result.imports.append(
                ImportInfo(
                    module=module,
                    names=[parts[-1]],
                )
            )

    def _extract_call(self, node: Node, result: ParseResult) -> None:
        """Extract an invocation expression."""
        # invocation_expression -> (member_access_expression | identifier) (arguments)
        func_node = node.child_by_field_name("function")
        if not func_node:
            # Try first child
            func_node = node.children[0] if node.children else None
            
        if func_node:
            if func_node.type == "member_access_expression":
                name = func_node.child_by_field_name("name")
                obj = func_node.child_by_field_name("expression")
                result.calls.append(
                    CallInfo(
                        name=name.text.decode("utf8") if name else "",
                        line=node.start_point[0] + 1,
                        receiver=obj.text.decode("utf8") if obj else "",
                    )
                )
            else:
                result.calls.append(
                    CallInfo(
                        name=func_node.text.decode("utf8"),
                        line=node.start_point[0] + 1,
                    )
                )
