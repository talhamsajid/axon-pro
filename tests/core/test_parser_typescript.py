"""Tests for the TypeScript / TSX / JavaScript parser."""

from __future__ import annotations

import pytest

from axon_pro.core.parsers.typescript import TypeScriptParser

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ts_parser() -> TypeScriptParser:
    return TypeScriptParser(dialect="typescript")


@pytest.fixture
def js_parser() -> TypeScriptParser:
    return TypeScriptParser(dialect="javascript")


# ---------------------------------------------------------------------------
# 1. Parse TypeScript function declaration
# ---------------------------------------------------------------------------


def test_parse_ts_function_declaration(ts_parser: TypeScriptParser) -> None:
    code = """\
function greet(name: string): string {
    return `Hello, ${name}`;
}
"""
    result = ts_parser.parse(code, "greet.ts")

    functions = [s for s in result.symbols if s.kind == "function"]
    assert len(functions) == 1

    fn = functions[0]
    assert fn.name == "greet"
    assert fn.start_line == 1
    assert fn.end_line == 3
    assert "function greet" in fn.content


# ---------------------------------------------------------------------------
# 2. Parse arrow function with type refs
# ---------------------------------------------------------------------------


def test_parse_arrow_function_with_types(ts_parser: TypeScriptParser) -> None:
    code = """\
const validate = (user: User): boolean => {
    return user.isValid();
};
"""
    result = ts_parser.parse(code, "validate.ts")

    functions = [s for s in result.symbols if s.kind == "function"]
    assert len(functions) == 1
    assert functions[0].name == "validate"

    # User should appear as a param type ref; boolean is built-in and skipped.
    type_names = [t.name for t in result.type_refs]
    assert "User" in type_names
    assert "boolean" not in type_names

    # Verify the param_name for the User type ref.
    user_refs = [t for t in result.type_refs if t.name == "User"]
    assert len(user_refs) == 1
    assert user_refs[0].kind == "param"
    assert user_refs[0].param_name == "user"


# ---------------------------------------------------------------------------
# 3. Parse class with methods and heritage
# ---------------------------------------------------------------------------


def test_parse_class_with_heritage(ts_parser: TypeScriptParser) -> None:
    code = """\
class Admin extends User implements Serializable {
    save(): void {
        this.validate();
    }
}
"""
    result = ts_parser.parse(code, "admin.ts")

    classes = [s for s in result.symbols if s.kind == "class"]
    assert len(classes) == 1
    assert classes[0].name == "Admin"

    methods = [s for s in result.symbols if s.kind == "method"]
    assert len(methods) == 1
    assert methods[0].name == "save"
    assert methods[0].class_name == "Admin"

    # Heritage: extends User, implements Serializable
    assert ("Admin", "extends", "User") in result.heritage
    assert ("Admin", "implements", "Serializable") in result.heritage

    # Call: this.validate()
    this_calls = [c for c in result.calls if c.receiver == "this"]
    assert len(this_calls) == 1
    assert this_calls[0].name == "validate"


# ---------------------------------------------------------------------------
# 4. Parse interface
# ---------------------------------------------------------------------------


def test_parse_interface(ts_parser: TypeScriptParser) -> None:
    code = """\
interface AuthConfig {
    secret: string;
    timeout: number;
}
"""
    result = ts_parser.parse(code, "config.ts")

    interfaces = [s for s in result.symbols if s.kind == "interface"]
    assert len(interfaces) == 1
    assert interfaces[0].name == "AuthConfig"
    assert interfaces[0].start_line == 1
    assert interfaces[0].end_line == 4


# ---------------------------------------------------------------------------
# 5. Parse type alias
# ---------------------------------------------------------------------------


def test_parse_type_alias(ts_parser: TypeScriptParser) -> None:
    code = """\
type UserId = string | number;
"""
    result = ts_parser.parse(code, "types.ts")

    type_aliases = [s for s in result.symbols if s.kind == "type_alias"]
    assert len(type_aliases) == 1
    assert type_aliases[0].name == "UserId"


# ---------------------------------------------------------------------------
# 6. Parse imports (named, namespace, default)
# ---------------------------------------------------------------------------


def test_parse_imports(ts_parser: TypeScriptParser) -> None:
    code = """\
import { User, Admin } from './models';
import * as utils from '../utils';
import express from 'express';
"""
    result = ts_parser.parse(code, "app.ts")

    assert len(result.imports) == 3

    # Named imports from relative module.
    named = [i for i in result.imports if i.module == "./models"][0]
    assert set(named.names) == {"User", "Admin"}
    assert named.is_relative is True
    assert named.alias == ""

    # Namespace import from relative module.
    ns = [i for i in result.imports if i.module == "../utils"][0]
    assert ns.names == ["utils"]
    assert ns.alias == "utils"
    assert ns.is_relative is True

    # Default import from package.
    default = [i for i in result.imports if i.module == "express"][0]
    assert default.names == ["express"]
    assert default.is_relative is False


# ---------------------------------------------------------------------------
# 7. Parse JavaScript (no types, require)
# ---------------------------------------------------------------------------


def test_parse_javascript(js_parser: TypeScriptParser) -> None:
    code = """\
function hello(name) {
    console.log(name);
}
const foo = require('./bar');
"""
    result = js_parser.parse(code, "app.js")

    # 1 function
    functions = [s for s in result.symbols if s.kind == "function"]
    assert len(functions) == 1
    assert functions[0].name == "hello"

    # 1 import via require
    assert len(result.imports) == 1
    imp = result.imports[0]
    assert imp.module == "./bar"
    assert imp.names == ["foo"]
    assert imp.is_relative is True

    # Calls include console.log
    log_calls = [c for c in result.calls if c.name == "log"]
    assert len(log_calls) == 1
    assert log_calls[0].receiver == "console"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_invalid_dialect_raises() -> None:
    with pytest.raises(ValueError, match="Unknown dialect"):
        TypeScriptParser(dialect="coffeescript")


def test_empty_source(ts_parser: TypeScriptParser) -> None:
    result = ts_parser.parse("", "empty.ts")
    assert result.symbols == []
    assert result.imports == []
    assert result.calls == []
    assert result.type_refs == []


def test_interface_extends_heritage(ts_parser: TypeScriptParser) -> None:
    code = """\
interface Foo extends Bar {
    x: number;
}
"""
    result = ts_parser.parse(code, "foo.ts")

    interfaces = [s for s in result.symbols if s.kind == "interface"]
    assert len(interfaces) == 1
    assert ("Foo", "extends", "Bar") in result.heritage


def test_function_expression(js_parser: TypeScriptParser) -> None:
    code = """\
const add = function(a, b) { return a + b; };
"""
    result = js_parser.parse(code, "math.js")

    functions = [s for s in result.symbols if s.kind == "function"]
    assert len(functions) == 1
    assert functions[0].name == "add"


def test_variable_type_annotation(ts_parser: TypeScriptParser) -> None:
    code = """\
const config: AppConfig = getConfig();
"""
    result = ts_parser.parse(code, "config.ts")

    var_types = [t for t in result.type_refs if t.kind == "variable"]
    assert len(var_types) == 1
    assert var_types[0].name == "AppConfig"


def test_return_type_ref(ts_parser: TypeScriptParser) -> None:
    code = """\
function getUser(): UserModel {
    return db.find();
}
"""
    result = ts_parser.parse(code, "user.ts")

    return_types = [t for t in result.type_refs if t.kind == "return"]
    assert len(return_types) == 1
    assert return_types[0].name == "UserModel"


# ---------------------------------------------------------------------------
# new expression handling
# ---------------------------------------------------------------------------


def test_new_expression_simple(js_parser: TypeScriptParser) -> None:
    """``new ClassName(args)`` creates a CallInfo targeting the class."""
    code = """\
function init() {
    const mgr = new AchievementManager(this);
}
"""
    result = js_parser.parse(code, "game.js")

    new_calls = [c for c in result.calls if c.name == "AchievementManager"]
    assert len(new_calls) == 1
    assert new_calls[0].line == 2
    assert new_calls[0].receiver == ""


def test_new_expression_with_member(ts_parser: TypeScriptParser) -> None:
    """``new module.ClassName()`` captures receiver and class name."""
    code = """\
const db = new pg.Client();
"""
    result = ts_parser.parse(code, "db.ts")

    new_calls = [c for c in result.calls if c.name == "Client"]
    assert len(new_calls) == 1
    assert new_calls[0].receiver == "pg"


def test_new_expression_callback_args(js_parser: TypeScriptParser) -> None:
    """``new Cls(callbackFn)`` captures bare identifier arguments."""
    code = """\
const watcher = new FileWatcher(onChange);
"""
    result = js_parser.parse(code, "watcher.js")

    new_calls = [c for c in result.calls if c.name == "FileWatcher"]
    assert len(new_calls) == 1
    assert "onChange" in new_calls[0].arguments


def test_new_expression_cookie_clicker_pattern(js_parser: TypeScriptParser) -> None:
    """Real-world pattern: exported class instantiated with ``new``."""
    code = """\
import { AchievementManager } from "./achievements.js";

export class Game {
    constructor() {
        this.achievementManager = new AchievementManager(this);
        this.prestige = new PrestigeManager(this);
    }

    start() {
        this.achievementManager.check();
    }
}
"""
    result = js_parser.parse(code, "game.js")

    # Both new expressions should create calls.
    call_names = [c.name for c in result.calls]
    assert "AchievementManager" in call_names
    assert "PrestigeManager" in call_names

    # Method call on instance should also be captured.
    check_calls = [c for c in result.calls if c.name == "check"]
    assert len(check_calls) == 1
    assert "achievementManager" in check_calls[0].receiver


# ---------------------------------------------------------------------------
# module.exports handling
# ---------------------------------------------------------------------------


def test_module_exports_identifier(js_parser: TypeScriptParser) -> None:
    """``module.exports = ClassName`` marks ClassName as exported."""
    code = """\
class AchievementManager {}
module.exports = AchievementManager;
"""
    result = js_parser.parse(code, "achievements.js")

    assert "AchievementManager" in result.exports


def test_module_exports_object(js_parser: TypeScriptParser) -> None:
    """``module.exports = { A, B }`` marks both A and B as exported."""
    code = """\
class Foo {}
class Bar {}
module.exports = { Foo, Bar };
"""
    result = js_parser.parse(code, "lib.js")

    assert "Foo" in result.exports
    assert "Bar" in result.exports
