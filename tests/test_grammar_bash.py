"""Bash grammar: function, select, C-for, (( )), time, ;&/;;&, coproc."""

from __future__ import annotations

import pytest

from sash import (
    ArithAssign,
    ArithBinary,
    ArithCommand,
    ArithUnary,
    BindingKind,
    BraceGroup,
    Case,
    Command,
    Coproc,
    Dialect,
    ForArith,
    FunDef,
    IdKind,
    Namespace,
    Pipeline,
    Program,
    Select,
    ShParseError,
    Simple,
    Subshell,
    While,
    find_ids,
    iter_ids,
    program_to_datum,
    resolve,
    sh_read,
    sh_read_program,
)
from sash.lexer import Lexer
from sash.nodes import TokEof, TokOp
from sash.render import _datum_str


def bash(src: str) -> Program:
    return sh_read(src, dialect=Dialect.BASH)


def first(src: str) -> Command:
    return bash(src).commands[0]


def op_syms(src: str, dialect: Dialect) -> list[str]:
    lexer = Lexer(src, dialect=dialect)
    syms: list[str] = []
    while True:
        tok = lexer.next_token()
        if isinstance(tok, TokEof):
            return syms
        if isinstance(tok, TokOp):
            syms.append(tok.id.sym)


# -- function keyword ---------------------------------------------------------


def test_function_keyword_brace() -> None:
    cmd = first("function foo { echo hi; }")
    assert isinstance(cmd, FunDef)
    assert cmd.name_id.sym == "foo"
    assert cmd.name_id.kind is IdKind.FUNCTION_NAME
    assert [k.sym for k in cmd.keywords] == ["function"]
    assert isinstance(cmd.body, BraceGroup)


def test_function_keyword_with_parens() -> None:
    cmd = first("function foo () { echo hi; }")
    assert isinstance(cmd, FunDef)
    assert [k.sym for k in cmd.keywords] == ["function", "(", ")"]


def test_function_body_any_compound() -> None:
    cmd = first("function loop while :; do break; done")
    assert isinstance(cmd, FunDef)
    assert isinstance(cmd.body, While)


def test_function_subshell_body() -> None:
    cmd = first("function f (echo hi)")
    assert isinstance(cmd, FunDef)
    assert isinstance(cmd.body, Subshell)


def test_function_simple_body_rejected() -> None:
    with pytest.raises(ShParseError):
        bash("function f echo hi")


def test_posix_function_is_a_word() -> None:
    cmd = sh_read("function x").commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.cmd_id is not None and cmd.cmd_id.sym == "function"


def test_posix_function_brace_form_errors() -> None:
    with pytest.raises(ShParseError):
        sh_read("function foo { echo hi; }")


# -- select -------------------------------------------------------------------


def test_select_with_in() -> None:
    cmd = first("select x in a b; do echo $x; done")
    assert isinstance(cmd, Select)
    assert cmd.var_id.sym == "x"
    assert cmd.var_id.kind is IdKind.FOR_VARIABLE
    assert cmd.in_words is not None and len(cmd.in_words) == 2


def test_select_without_in_is_implicit_args() -> None:
    cmd = first("select x; do :; done")
    assert isinstance(cmd, Select)
    assert cmd.in_words is None


def test_select_variable_binds_like_for() -> None:
    program, _store = sh_read_program(
        "select x in a b; do echo $x; done", dialect=Dialect.BASH
    )
    (ref,) = find_ids(program, kind=IdKind.VARIABLE_REF, sym="x")
    binding = resolve(ref)
    assert binding is not None
    assert binding.kind is BindingKind.FOR


def test_posix_select_is_a_word() -> None:
    cmd = sh_read("select x").commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.cmd_id is not None and cmd.cmd_id.sym == "select"


# -- C-style for --------------------------------------------------------------


def test_c_style_for() -> None:
    cmd = first("for ((i=0; i<5; i++)); do echo $i; done")
    assert isinstance(cmd, ForArith)
    assert isinstance(cmd.init, ArithAssign)
    assert isinstance(cmd.cond, ArithBinary)
    assert isinstance(cmd.step, ArithUnary)
    assert [k.sym for k in cmd.keywords] == ["for", "((", "))", "do", "done"]


def test_c_style_for_empty_sections() -> None:
    cmd = first("for ((;;)); do break; done")
    assert isinstance(cmd, ForArith)
    assert cmd.init is None and cmd.cond is None and cmd.step is None


def test_c_style_for_do_without_semicolon() -> None:
    cmd = first("for ((i=0; i<5; i++)) do :; done")
    assert isinstance(cmd, ForArith)


def test_c_style_for_binders_resolve() -> None:
    program, _store = sh_read_program(
        "for ((i=0; i<5; i++)); do echo $i; done", dialect=Dialect.BASH
    )
    binders = find_ids(program, kind=IdKind.VARIABLE_BINDER, sym="i")
    refs = find_ids(program, kind=IdKind.VARIABLE_REF, sym="i")
    assert len(binders) == 2  # i=0 and i++
    assert len(refs) == 2  # i<5 and $i
    binding = resolve(refs[1])
    assert binding is not None
    assert binding.kind is BindingKind.ASSIGNMENT
    assert binders[0] in [site.id for site in binding.sites]


def test_posix_c_style_for_errors() -> None:
    with pytest.raises(ShParseError):
        sh_read("for ((i=0; i<5; i++)); do :; done")


# -- (( )) arithmetic command -------------------------------------------------


def test_arith_command() -> None:
    cmd = first("(( x=1 ))")
    assert isinstance(cmd, ArithCommand)
    assert isinstance(cmd.expr, ArithAssign)
    assert [k.sym for k in cmd.keywords] == ["((", "))"]


def test_arith_command_assignment_registers() -> None:
    program, _store = sh_read_program("(( x=1 ))\necho $x", dialect=Dialect.BASH)
    (ref,) = find_ids(program, kind=IdKind.VARIABLE_REF, sym="x")
    binding = resolve(ref)
    assert binding is not None
    assert binding.kind is BindingKind.ASSIGNMENT


def test_nested_subshell_not_arith() -> None:
    cmd = first("( (echo a) )")
    assert isinstance(cmd, Subshell)
    assert isinstance(cmd.body[0], Subshell)


def test_adjacent_parens_fall_back_to_subshell() -> None:
    cmd = first("((echo a); echo b)")
    assert isinstance(cmd, Subshell)


def test_arith_command_with_redirect() -> None:
    cmd = first("(( x++ )) >out")
    assert isinstance(cmd, ArithCommand)
    assert len(cmd.redirects) == 1


def test_posix_double_paren_is_nested_subshell() -> None:
    cmd = sh_read("((x=1))").commands[0]
    assert isinstance(cmd, Subshell)
    assert isinstance(cmd.body[0], Subshell)


# -- time pipeline prefix -----------------------------------------------------


def test_time_prefix() -> None:
    cmd = first("time sleep 1")
    assert isinstance(cmd, Pipeline)
    assert cmd.time_id is not None and cmd.time_id.sym == "time"
    assert not cmd.time_p
    assert len(cmd.cmds) == 1


def test_time_p_flag() -> None:
    cmd = first("time -p sleep 1")
    assert isinstance(cmd, Pipeline)
    assert cmd.time_p


def test_time_bang() -> None:
    cmd = first("time ! false")
    assert isinstance(cmd, Pipeline)
    assert cmd.time_id is not None and cmd.bang_id is not None


def test_time_full_pipeline() -> None:
    cmd = first("time a | b")
    assert isinstance(cmd, Pipeline)
    assert len(cmd.cmds) == 2


def test_bare_time() -> None:
    cmd = first("time")
    assert isinstance(cmd, Pipeline)
    assert cmd.time_id is not None
    assert cmd.cmds == []


def test_posix_time_is_a_word() -> None:
    cmd = sh_read("time ls").commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.cmd_id is not None and cmd.cmd_id.sym == "time"


# -- case terminators ;& and ;;& ----------------------------------------------


def test_case_fallthrough_terminators() -> None:
    src = "case $x in a) echo a ;& b) echo b ;;& c) echo c ;; esac"
    cmd = first(src)
    assert isinstance(cmd, Case)
    assert [item.terminator for item in cmd.items] == [";&", ";;&", ";;"]


def test_case_last_item_empty_terminator() -> None:
    cmd = first("case x in a) b ;& c) d\nesac")
    assert isinstance(cmd, Case)
    assert [item.terminator for item in cmd.items] == [";&", ""]


def test_case_terminator_ids_recorded() -> None:
    cmd = first("case x in a) b ;;& esac")
    assert isinstance(cmd, Case)
    assert cmd.items[0].keywords[-1].sym == ";;&"
    assert cmd.items[0].keywords[-1].kind is IdKind.OPERATOR


def test_terminator_operator_binding() -> None:
    program, _store = sh_read_program("case x in a) b ;& esac", dialect=Dialect.BASH)
    op = next(i for i in iter_ids(program) if i.sym == ";&")
    binding = resolve(op, Namespace.OPERATOR)
    assert binding is not None
    assert binding.kind is BindingKind.OPERATOR


def test_bash_case_terminator_lexing() -> None:
    assert op_syms("a ;& b", Dialect.BASH) == [";&"]
    assert op_syms("a ;;& b", Dialect.BASH) == [";;&"]


def test_posix_case_terminator_lexing_pinned() -> None:
    assert op_syms("a ;& b", Dialect.POSIX) == [";", "&"]
    assert op_syms("a ;;& b", Dialect.POSIX) == [";;", "&"]


def test_posix_fallthrough_terminator_errors() -> None:
    with pytest.raises(ShParseError):
        sh_read("case x in a) : ;& esac")


# -- coproc -------------------------------------------------------------------


def test_coproc_named() -> None:
    cmd = first("coproc mycop { echo hi; }")
    assert isinstance(cmd, Coproc)
    assert cmd.name_id is not None and cmd.name_id.sym == "mycop"
    assert cmd.name_id.kind is IdKind.VARIABLE_BINDER
    assert isinstance(cmd.cmd, BraceGroup)
    assert [k.sym for k in cmd.keywords] == ["coproc"]


def test_coproc_unnamed_simple() -> None:
    cmd = first("coproc sort file")
    assert isinstance(cmd, Coproc)
    assert cmd.name_id is None
    assert isinstance(cmd.cmd, Simple)
    assert cmd.cmd.cmd_id is not None and cmd.cmd.cmd_id.sym == "sort"


def test_coproc_unnamed_compound() -> None:
    cmd = first("coproc { echo hi; }")
    assert isinstance(cmd, Coproc)
    assert cmd.name_id is None
    assert isinstance(cmd.cmd, BraceGroup)


def test_posix_coproc_is_a_word() -> None:
    cmd = sh_read("coproc ls").commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.cmd_id is not None and cmd.cmd_id.sym == "coproc"


# -- rendering and painting ---------------------------------------------------

_COVERAGE_SRC = "\n".join(
    [
        "select x in a; do (( x++ )); done",
        "for ((i=0; i<3; i++)); do :; done",
        "coproc c { :; }",
        "time ! :",
        "case y in a) : ;& esac",
    ]
)


def test_datum_coverage() -> None:
    text = _datum_str(program_to_datum(bash(_COVERAGE_SRC)))
    assert "(select (id x for-variable)" in text
    assert "(for-arith (init" in text
    assert "(arith-cmd" in text
    assert "(coproc (id c variable-binder)" in text
    assert "(pipeline (id time operator)" in text
    assert ";&" in text


def test_all_new_ids_painted() -> None:
    program, _store = sh_read_program(_COVERAGE_SRC, dialect=Dialect.BASH)
    assert all(i.scopes for i in iter_ids(program))
