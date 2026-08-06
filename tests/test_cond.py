"""Bash [[ ]] conditional expressions."""

from __future__ import annotations

import pytest

from sash import (
    BindingKind,
    Command,
    CondAnd,
    CondBinary,
    CondCmd,
    CondGroup,
    CondNot,
    CondOr,
    CondUnary,
    Dialect,
    DQuote,
    IdKind,
    Lit,
    Namespace,
    Param,
    Program,
    ShParseError,
    Simple,
    SQuote,
    Word,
    find_ids,
    iter_ids,
    program_to_datum,
    resolve,
    sh_read,
    sh_read_program,
)
from sash.render import _datum_str


def bash(src: str) -> Program:
    return sh_read(src, dialect=Dialect.BASH)


def first(src: str) -> Command:
    return bash(src).commands[0]


def cond(src: str) -> CondCmd:
    cmd = first(src)
    assert isinstance(cmd, CondCmd)
    return cmd


def datum_text(src: str) -> str:
    return _datum_str(program_to_datum(bash(src)))


# -- file and string operators ------------------------------------------------


def test_file_unary() -> None:
    cmd = cond("[[ -f $x ]]")
    assert isinstance(cmd.expr, CondUnary)
    assert cmd.expr.op_id.sym == "-f"
    assert cmd.expr.op_id.kind is IdKind.OPERATOR
    assert isinstance(cmd.expr.operand.parts[0], Param)
    assert [k.sym for k in cmd.keywords] == ["[[", "]]"]


def test_variable_unary() -> None:
    cmd = cond("[[ -v foo ]]")
    assert isinstance(cmd.expr, CondUnary)
    assert cmd.expr.op_id.sym == "-v"


def test_bare_word_tests_nonemptiness() -> None:
    cmd = cond("[[ $x ]]")
    assert isinstance(cmd.expr, Word)
    assert isinstance(cmd.expr.parts[0], Param)


def test_string_eq_unquoted_glob_rhs() -> None:
    cmd = cond("[[ $x == a* ]]")
    assert isinstance(cmd.expr, CondBinary)
    assert cmd.expr.op_id.sym == "=="
    (part,) = cmd.expr.right.parts
    assert isinstance(part, Lit)
    assert part.text == "a*"


def test_string_eq_quoted_rhs_part_structure_differs() -> None:
    dq = cond('[[ $x == "a*" ]]')
    sq = cond("[[ $x == 'a*' ]]")
    assert isinstance(dq.expr, CondBinary) and isinstance(sq.expr, CondBinary)
    (dq_part,) = dq.expr.right.parts
    (sq_part,) = sq.expr.right.parts
    assert isinstance(dq_part, DQuote)
    assert isinstance(sq_part, SQuote)
    assert sq_part.text == "a*"


def test_single_equals_and_ne() -> None:
    assert isinstance(cond("[[ a = b ]]").expr, CondBinary)
    cmd = cond("[[ a != b ]]")
    assert isinstance(cmd.expr, CondBinary)
    assert cmd.expr.op_id.sym == "!="


def test_string_ordering_operators() -> None:
    lt = cond("[[ a < b ]]")
    gt = cond("[[ a > b ]]")
    assert isinstance(lt.expr, CondBinary) and lt.expr.op_id.sym == "<"
    assert isinstance(gt.expr, CondBinary) and gt.expr.op_id.sym == ">"


def test_arithmetic_comparisons() -> None:
    cmd = cond("[[ $n -eq 5 ]]")
    assert isinstance(cmd.expr, CondBinary)
    assert cmd.expr.op_id.sym == "-eq"
    assert "(cond-binary -eq" in datum_text("[[ $n -eq 5 ]]")


# -- regex matching -----------------------------------------------------------


def test_regex_binary() -> None:
    cmd = cond("[[ $x =~ ab ]]")
    assert isinstance(cmd.expr, CondBinary)
    assert cmd.expr.op_id.sym == "=~"
    (part,) = cmd.expr.right.parts
    assert isinstance(part, Lit)
    assert part.text == "ab"


def test_regex_unquoted_parens_stay_in_word() -> None:
    cmd = cond("[[ $x =~ ^(ab|cd)+$ ]]")
    assert isinstance(cmd.expr, CondBinary)
    # the trailing lone '$' is its own literal part, per the usual $-rules
    assert all(isinstance(p, Lit) for p in cmd.expr.right.parts)
    text = "".join(p.text for p in cmd.expr.right.parts if isinstance(p, Lit))
    assert text == "^(ab|cd)+$"


def test_regex_quoted() -> None:
    cmd = cond('[[ $x =~ "a b" ]]')
    assert isinstance(cmd.expr, CondBinary)
    (part,) = cmd.expr.right.parts
    assert isinstance(part, DQuote)


def test_regex_registers_bash_rematch() -> None:
    program, _store = sh_read_program(
        "[[ $x =~ ab ]]; echo ${BASH_REMATCH[0]}", dialect=Dialect.BASH
    )
    (ref,) = find_ids(program, kind=IdKind.VARIABLE_REF, sym="BASH_REMATCH")
    binding = resolve(ref)
    assert binding is not None
    assert binding.kind is BindingKind.SPECIAL_PARAM
    assert binding.sites[0].kind == "implicit"


def test_bash_rematch_registered_once() -> None:
    program, _store = sh_read_program(
        "[[ $x =~ a ]]; [[ $y =~ b ]]; echo $BASH_REMATCH", dialect=Dialect.BASH
    )
    (ref,) = find_ids(program, kind=IdKind.VARIABLE_REF, sym="BASH_REMATCH")
    binding = resolve(ref)
    assert binding is not None
    assert len(binding.sites) == 1


# -- precedence and grouping --------------------------------------------------


def test_and_binds_tighter_than_or() -> None:
    cmd = cond("[[ -f a && -f b || -f c ]]")
    assert isinstance(cmd.expr, CondOr)
    assert isinstance(cmd.expr.left, CondAnd)
    assert isinstance(cmd.expr.right, CondUnary)


def test_not_binds_tighter_than_and() -> None:
    cmd = cond("[[ ! -f a && -f b ]]")
    assert isinstance(cmd.expr, CondAnd)
    assert isinstance(cmd.expr.left, CondNot)
    assert isinstance(cmd.expr.left.operand, CondUnary)


def test_parens_group() -> None:
    cmd = cond("[[ ( -f a || -f b ) && -f c ]]")
    assert isinstance(cmd.expr, CondAnd)
    assert isinstance(cmd.expr.left, CondGroup)
    assert isinstance(cmd.expr.left.inner, CondOr)


def test_precedence_datum_shape() -> None:
    text = datum_text("[[ ! ( -n a || -z b ) && c == d ]]")
    assert "(cond-and (cond-not (cond-group (cond-or" in text
    assert "(cond-binary ==" in text


# -- newlines and redirects ---------------------------------------------------


def test_newline_after_and() -> None:
    cmd = cond("[[ -f a &&\n-f b ]]")
    assert isinstance(cmd.expr, CondAnd)


def test_newline_before_close() -> None:
    cmd = cond("[[ -f a\n]]")
    assert isinstance(cmd.expr, CondUnary)


def test_redirect_after_close() -> None:
    cmd = cond("[[ -f x ]] > out")
    assert len(cmd.redirects) == 1
    assert cmd.redirects[0].op_id.sym == ">"


def test_cond_in_if_and_and_or() -> None:
    assert bash("if [[ -f x ]]; then :; fi")
    assert bash("[[ -f x ]] && echo y")


# -- errors -------------------------------------------------------------------


def test_missing_close_errors() -> None:
    with pytest.raises(ShParseError):
        bash("[[ -f x")


def test_empty_cond_errors() -> None:
    with pytest.raises(ShParseError):
        bash("[[ ]]")


def test_semicolon_inside_cond_errors() -> None:
    with pytest.raises(ShParseError):
        bash("[[ -f x; ]]")


# -- POSIX pin ----------------------------------------------------------------


def test_posix_double_bracket_is_ordinary_command() -> None:
    cmd = sh_read("[[ -f x ]]").commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.cmd_id is not None and cmd.cmd_id.sym == "[["
    assert [w.parts[0].text for w in cmd.words if isinstance(w.parts[0], Lit)] == [
        "[[",
        "-f",
        "x",
        "]]",
    ]


# -- binding and rendering ----------------------------------------------------


def test_operator_ids_resolve() -> None:
    program, _store = sh_read_program("[[ -f $x && $y == z ]]", dialect=Dialect.BASH)
    for sym in ("[[", "]]", "-f", "&&", "=="):
        op = next(i for i in iter_ids(program) if i.sym == sym)
        binding = resolve(op, Namespace.OPERATOR)
        assert binding is not None, sym
        assert binding.kind is BindingKind.OPERATOR


def test_params_inside_cond_resolve() -> None:
    program, _store = sh_read_program("x=1\n[[ $x == 1 ]]", dialect=Dialect.BASH)
    (ref,) = find_ids(program, kind=IdKind.VARIABLE_REF, sym="x")
    binding = resolve(ref)
    assert binding is not None
    assert binding.kind is BindingKind.ASSIGNMENT


def test_all_cond_ids_painted() -> None:
    src = "[[ ! ( -f $a || $b =~ c(d)e ) && $x != 'y' ]] > out"
    program, _store = sh_read_program(src, dialect=Dialect.BASH)
    assert all(i.scopes for i in iter_ids(program))
