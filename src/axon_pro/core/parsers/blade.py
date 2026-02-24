"""Blade template parser for Axon Pro.

Extracts component usage, inclusions, and potential variable dependencies
from Laravel Blade files.
"""

from __future__ import annotations

import re
from axon_pro.core.parsers.base import (
    CallInfo,
    LanguageParser,
    ParseResult,
    SymbolInfo,
)

class BladeParser(LanguageParser):
    """Parses .blade.php files to extract structural relationships."""

    def parse(self, content: str, file_path: str) -> ParseResult:
        result = ParseResult()
        
        # 1. Extract Components (x-component-name)
        # Pattern: <x-([\w\.-]+)
        component_matches = re.finditer(r"<x-([\w\.-]+)", content)
        for match in component_matches:
            name = match.group(1)
            result.calls.append(
                CallInfo(
                    name=f"x-{name}",
                    line=content.count('
', 0, match.start()) + 1,
                    receiver="BladeComponent"
                )
            )

        # 2. Extract Includes (@include('view.name'))
        # Pattern: @include\(['"]([\w\.-]+)['"]
        include_matches = re.finditer(r"@include\(['"]([\w\.-]+)['"]", content)
        for match in include_matches:
            view_name = match.group(1)
            result.calls.append(
                CallInfo(
                    name=view_name,
                    line=content.count('
', 0, match.start()) + 1,
                    receiver="BladeInclude"
                )
            )

        # 3. Extract Component Directives (@component('name'))
        comp_directive_matches = re.finditer(r"@component\(['"]([\w\.-]+)['"]", content)
        for match in comp_directive_matches:
            name = match.group(1)
            result.calls.append(
                CallInfo(
                    name=name,
                    line=content.count('
', 0, match.start()) + 1,
                    receiver="BladeComponent"
                )
            )

        # We treat the whole file as a "view" symbol
        view_name = file_path.replace("resources/views/", "").replace(".blade.php", "").replace("/", ".")
        result.symbols.append(
            SymbolInfo(
                name=view_name,
                kind="view",
                start_line=1,
                end_line=content.count('
') + 1,
                content=content
            )
        )

        return result
