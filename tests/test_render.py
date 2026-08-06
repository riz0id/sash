"""Identifier traversal and datum rendering."""

from __future__ import annotations

from sash import IdKind, find_ids, iter_ids, pretty, sh_read, sh_read_program
from sash.render import program_to_datum


def test_iter_ids_reaches_nested_sites() -> None:
    src = "cat <<E\n$a\nE\necho $((b + 1)) $(echo $c)\n"
    syms = {i.sym for i in iter_ids(sh_read(src))}
    assert {"a", "b", "c"} <= syms


def test_find_ids_filters_by_kind_and_sym() -> None:
    program = sh_read("x=1\necho $x | cat")
    assert [i.sym for i in find_ids(program, kind=IdKind.VARIABLE_BINDER)] == ["x"]
    assert [i.kind for i in find_ids(program, sym="|")] == [IdKind.OPERATOR]
    assert find_ids(program, kind=IdKind.OPERATOR, sym="cat") == []


def test_keywords_are_traversed() -> None:
    program = sh_read("if a; then b; fi")
    syms = [i.sym for i in iter_ids(program)]
    assert "if" in syms and "then" in syms and "fi" in syms


def test_pretty_smoke() -> None:
    assert pretty(sh_read("echo hi")) == (
        "(program\n  (simple (word (lit 'echo')) (word (lit 'hi'))))"
    )


def test_datum_ids_without_scopes() -> None:
    datum = program_to_datum(sh_read("a | b"))
    assert datum == (
        "program",
        (
            "pipeline",
            ("simple", ("word", ("lit", "'a'"))),
            ("id", "|", "operator"),
            ("simple", ("word", ("lit", "'b'"))),
        ),
    )


def test_datum_ids_with_scope_names() -> None:
    program, _store = sh_read_program("x=1")
    datum = program_to_datum(program, scopes=True)
    assert datum == (
        "program",
        (
            "simple",
            (
                "assign",
                ("id", "x", "variable-binder", ("core", "top")),
                ("word", ("lit", "'1'")),
            ),
        ),
    )
