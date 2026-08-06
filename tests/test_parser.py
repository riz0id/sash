"""Grammar coverage: AST shapes via datum rendering plus node inspection."""

from __future__ import annotations

import pytest

from sash import (
    Case,
    CmdSub,
    For,
    FunDef,
    Program,
    ShParseError,
    Simple,
    Subshell,
    sh_read,
)
from sash.render import Datum, program_to_datum


def datum(src: str) -> Datum:
    return program_to_datum(sh_read(src))


def simple(*words: str) -> Datum:
    return ("simple", *(("word", ("lit", repr(w))) for w in words))


def test_simple_command() -> None:
    assert datum("echo hi") == ("program", simple("echo", "hi"))


def test_pipeline() -> None:
    assert datum("a | b") == (
        "program",
        ("pipeline", simple("a"), ("id", "|", "operator"), simple("b")),
    )


def test_bang_pipeline() -> None:
    assert datum("! a") == (
        "program",
        ("pipeline", ("id", "!", "operator"), simple("a")),
    )


def test_and_or_left_associative() -> None:
    assert datum("a && b || c") == (
        "program",
        (
            "and-or",
            ("and-or", simple("a"), ("id", "&&", "operator"), simple("b")),
            ("id", "||", "operator"),
            simple("c"),
        ),
    )


def test_async() -> None:
    assert datum("a &") == ("program", ("async", simple("a")))


def test_semicolon_sequence() -> None:
    assert datum("a; b") == ("program", simple("a"), simple("b"))


def test_subshell() -> None:
    assert datum("(a; b)") == ("program", ("subshell", simple("a"), simple("b")))


def test_brace_group() -> None:
    assert datum("{ a; b; }") == ("program", ("brace-group", simple("a"), simple("b")))


def test_if_else() -> None:
    assert datum("if a; then b; else c; fi") == (
        "program",
        (
            "if",
            ("test", simple("a")),
            ("then", simple("b")),
            ("else", simple("c")),
        ),
    )


def test_elif_clauses() -> None:
    assert datum("if a; then b; elif c; then d; fi") == (
        "program",
        (
            "if",
            ("test", simple("a")),
            ("then", simple("b")),
            ("test", simple("c")),
            ("then", simple("d")),
        ),
    )


def test_while() -> None:
    assert datum("while a; do b; done") == (
        "program",
        ("while", ("test", simple("a")), ("body", simple("b"))),
    )


def test_until() -> None:
    assert datum("until a; do b; done") == (
        "program",
        ("until", ("test", simple("a")), ("body", simple("b"))),
    )


def test_for_with_in() -> None:
    assert datum("for x in a b; do c; done") == (
        "program",
        (
            "for",
            ("id", "x", "for-variable"),
            ("in", ("word", ("lit", "'a'")), ("word", ("lit", "'b'"))),
            ("body", simple("c")),
        ),
    )


def test_for_without_in_is_implicit_args() -> None:
    program = sh_read("for x\ndo c\ndone")
    cmd = program.commands[0]
    assert isinstance(cmd, For)
    assert cmd.in_words is None


def test_case_items_and_patterns() -> None:
    program = sh_read("case $x in a|b) echo 1 ;; *) echo 2 ;; esac")
    cmd = program.commands[0]
    assert isinstance(cmd, Case)
    assert len(cmd.items) == 2
    assert len(cmd.items[0].patterns) == 2
    assert len(cmd.items[1].patterns) == 1


def test_case_last_item_without_dsemi() -> None:
    program = sh_read("case x in a) b ;; c) d\nesac")
    cmd = program.commands[0]
    assert isinstance(cmd, Case)
    assert len(cmd.items) == 2


def test_fundef_brace_body() -> None:
    program = sh_read("f() { a; }")
    cmd = program.commands[0]
    assert isinstance(cmd, FunDef)
    assert cmd.name_id.sym == "f"


def test_fundef_subshell_body() -> None:
    program = sh_read("f() (a)")
    cmd = program.commands[0]
    assert isinstance(cmd, FunDef)
    assert isinstance(cmd.body, Subshell)


def test_redirects_with_io_number() -> None:
    program = sh_read("echo hi >out 2>&1")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert len(cmd.redirects) == 2
    assert cmd.redirects[0].n is None
    assert cmd.redirects[0].op_id.sym == ">"
    assert cmd.redirects[1].n == 2
    assert cmd.redirects[1].op_id.sym == ">&"


def test_redirect_on_compound() -> None:
    assert datum("{ a; } >out") == (
        "program",
        (
            "brace-group",
            simple("a"),
            ("redirect", ("id", ">", "operator"), ("word", ("lit", "'out'"))),
        ),
    )


def test_assignment_prefix_only() -> None:
    program = sh_read("x=1 y=2 cmd z=3")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert [a.name_id.sym for a in cmd.assigns] == ["x", "y"]
    assert cmd.cmd_id is not None and cmd.cmd_id.sym == "cmd"
    assert len(cmd.words) == 2  # cmd and the literal word z=3


def test_bare_assignment_has_no_cmd_id() -> None:
    program = sh_read("x=1")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert len(cmd.assigns) == 1
    assert cmd.words == []
    assert cmd.cmd_id is None


def test_quoting_defeats_reserved_word() -> None:
    program = sh_read('"if" a')
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.cmd_id is None


def test_escape_defeats_reserved_word() -> None:
    program = sh_read("\\if a")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)


def test_quoting_defeats_assignment() -> None:
    program = sh_read('"x"=1 cmd')
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.assigns == []


def test_quoted_command_word_has_no_cmd_id() -> None:
    program = sh_read('"echo" hi')
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.cmd_id is None


def test_reserved_word_not_promoted_outside_command_position() -> None:
    program = sh_read("echo }")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert len(cmd.words) == 2


def cmdsub_program(src: str) -> Program:
    program = sh_read(src)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    part = cmd.words[1].parts[0]
    assert isinstance(part, CmdSub)
    assert part.program is not None
    return part.program


def test_cmdsub_recursive_parse() -> None:
    inner = cmdsub_program("echo $(date +%s)")
    assert isinstance(inner.commands[0], Simple)


def test_cmdsub_case_pattern_paren() -> None:
    inner = cmdsub_program("echo $(case a in b) echo c ;; esac)")
    assert isinstance(inner.commands[0], Case)


def test_cmdsub_nested() -> None:
    inner = cmdsub_program("echo $(echo $(echo hi))")
    cmd = inner.commands[0]
    assert isinstance(cmd, Simple)
    part = cmd.words[1].parts[0]
    assert isinstance(part, CmdSub)
    assert part.program is not None


def test_heredoc_body_cmdsub_is_parsed() -> None:
    program = sh_read("cat <<EOF\n$(echo hi)\nEOF\n")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    target = cmd.redirects[0].target
    from sash import HereDoc

    assert isinstance(target, HereDoc)
    assert target.body is not None
    part = target.body[0]
    assert isinstance(part, CmdSub)
    assert part.program is not None


def test_parse_errors() -> None:
    bad = [
        "if a; then b",  # missing fi
        "while a; do b",  # missing done
        "case x in a) b",  # missing esac
        "f() ",  # missing body
        "a |",  # missing pipeline tail
        "a &&",  # missing and-or tail
        "echo $(",  # unterminated command substitution
        "echo ${x",  # unterminated parameter expansion
        ")",  # stray paren
        ";; a",  # stray dsemi
    ]
    for src in bad:
        with pytest.raises(ShParseError):
            sh_read(src)
