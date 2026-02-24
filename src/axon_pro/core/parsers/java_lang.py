"""Java language parser using tree-sitter.

Extracts functions, classes, methods, and inheritance relationships
from Java source code.
"""

from __future__ import annotations

import tree_sitter_java as tsjava
from tree_sitter import Language, Node, Parser

from axon_pro.core.parsers.base import (
    CallInfo,
    ImportInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
)

JAVA_LANGUAGE = Language(tsjava.language())

class JavaParser(LanguageParser):
    """Parses Java source code using tree-sitter."""

    def __init__(self) -> None:
        self._parser = Parser(JAVA_LANGUAGE)

    def parse(self, content: str, file_path: str) -> ParseResult:
        """Parse Java source and return structured information."""
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
                case "import_declaration":
                    self._extract_import(child, result)
                case "method_invocation":
                    self._extract_call(child, result)
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
        superclass = node.child_by_field_name("superclass")
        if superclass:
            parent_name = superclass.text.decode("utf8").replace("extends ", "").strip()
            result.heritage.append((class_name, "extends", parent_name))

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
        """Extract an import declaration."""
        # Java tree-sitter: import_declaration has no named children for the path.
        # It's usually 'import' followed by a name or asterisk_import.
        text = node.text.decode("utf8").replace("import ", "").replace(";", "").strip()
        parts = text.split(".")
        result.imports.append(
            ImportInfo(
                module=text,
                names=[parts[-1]],
            )
        )

    def _extract_call(self, node: Node, result: ParseResult) -> None:
        """Extract a method invocation."""
        name_node = node.child_by_field_name("name")
        obj_node = node.child_by_field_name("object")
        if name_node:
            receiver = obj_node.text.decode("utf8") if obj_node else ""
            result.calls.append(
                CallInfo(
                    name=name_node.text.decode("utf8"),
                    line=node.start_point[0] + 1,
                    receiver=receiver,
                )
            )
