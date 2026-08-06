"""Arithmetic expression parsing, rendering, and binding."""

from __future__ import annotations

import pytest

from sash import (
    Arith,
    ArithAssign,
    ArithBinary,
    ArithExpr,
    ArithNum,
    ArithPart,
    ArithTernary,
    ArithUnary,
    ArithVar,
    BindingKind,
    CmdSub,
    IdKind,
    Param,
    Program,
    ShParseError,
    Simple,
    find_ids,
    resolve,
    sh_read,
    sh_read_program,
)
from sash.render import Datum, _arith_datum


def expr_of(src: str) -> ArithExpr:
    program = sh_read(f"echo $(({src}))")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    part = cmd.words[1].parts[0]
    assert isinstance(part, Arith)
    return part.expr


def datum_of(src: str) -> Datum:
    return _arith_datum(expr_of(src), False)


def num(text: str) -> Datum:
    return ("arith-num", text)


def var(sym: str, kind: str = "variable-ref") -> Datum:
    return ("arith-var", ("id", sym, kind))


def read_bound(src: str) -> Program:
    program, _store = sh_read_program(src)
    return program


# -- precedence and associativity -------------------------------------------


def test_multiplicative_over_additive() -> None:
    assert datum_of("1 + 2 * 3") == (
        "arith-binary",
        "+",
        num("1"),
        ("arith-binary", "*", num("2"), num("3")),
    )


def test_additive_left_assoc() -> None:
    assert datum_of("1 - 2 - 3") == (
        "arith-binary",
        "-",
        ("arith-binary", "-", num("1"), num("2")),
        num("3"),
    )


def test_exponent_right_assoc() -> None:
    assert datum_of("2 ** 3 ** 2") == (
        "arith-binary",
        "**",
        num("2"),
        ("arith-binary", "**", num("3"), num("2")),
    )


def test_shift_binds_tighter_than_relational() -> None:
    assert datum_of("1 << 2 < 3") == (
        "arith-binary",
        "<",
        ("arith-binary", "<<", num("1"), num("2")),
        num("3"),
    )


def test_logical_layers() -> None:
    assert datum_of("a || b && c") == (
        "arith-binary",
        "||",
        var("a"),
        ("arith-binary", "&&", var("b"), var("c")),
    )


def test_bitwise_layers() -> None:
    assert datum_of("a | b ^ c & d == e") == (
        "arith-binary",
        "|",
        var("a"),
        (
            "arith-binary",
            "^",
            var("b"),
            (
                "arith-binary",
                "&",
                var("c"),
                ("arith-binary", "==", var("d"), var("e")),
            ),
        ),
    )


def test_unary_before_exponent() -> None:
    # bash: unary minus binds tighter than **, so -2**2 is (-2)**2
    assert datum_of("-2 ** 2") == (
        "arith-binary",
        "**",
        ("arith-unary", "-", num("2")),
        num("2"),
    )


def test_unary_stack() -> None:
    assert datum_of("!~x") == (
        "arith-unary",
        "!",
        ("arith-unary", "~", var("x")),
    )


def test_group_overrides_precedence() -> None:
    assert datum_of("(1 + 2) * 3") == (
        "arith-binary",
        "*",
        ("arith-group", ("arith-binary", "+", num("1"), num("2"))),
        num("3"),
    )


def test_ternary_right_assoc() -> None:
    assert datum_of("a ? 1 : b ? 2 : 3") == (
        "arith-ternary",
        var("a"),
        num("1"),
        ("arith-ternary", var("b"), num("2"), num("3")),
    )


def test_comma_is_lowest() -> None:
    assert datum_of("x = 1, y = 2") == (
        "arith-binary",
        ",",
        ("arith-assign", "=", var("x", "variable-binder"), num("1")),
        ("arith-assign", "=", var("y", "variable-binder"), num("2")),
    )


def test_assignment_right_assoc_and_compound_ops() -> None:
    expr = expr_of("x = y += 2")
    assert isinstance(expr, ArithAssign) and expr.op == "="
    inner = expr.value
    assert isinstance(inner, ArithAssign) and inner.op == "+="
    assert inner.target.id.sym == "y"
    assert inner.target.id.kind is IdKind.VARIABLE_BINDER


def test_assignment_below_ternary() -> None:
    expr = expr_of("x = a ? 1 : 2")
    assert isinstance(expr, ArithAssign)
    assert isinstance(expr.value, ArithTernary)


def test_assignment_target_must_be_variable() -> None:
    with pytest.raises(ShParseError):
        expr_of("1 = 2")


# -- numbers -----------------------------------------------------------------


@pytest.mark.parametrize("text", ["42", "0x1F", "017", "2#101", "64#a_@"])
def test_number_forms(text: str) -> None:
    expr = expr_of(text)
    assert isinstance(expr, ArithNum)
    assert expr.text == text


@pytest.mark.parametrize("text", ["1#0", "65#0", "0x", "2#"])
def test_bad_numbers_raise(text: str) -> None:
    with pytest.raises(ShParseError):
        expr_of(text)


# -- variables and subscripts ------------------------------------------------


def test_subscripted_variable() -> None:
    expr = expr_of("a[i + 1]")
    assert isinstance(expr, ArithVar)
    assert expr.id.sym == "a"
    assert expr.id.kind is IdKind.VARIABLE_REF
    sub = expr.subscript
    assert isinstance(sub, ArithBinary) and sub.op == "+"
    assert isinstance(sub.left, ArithVar)
    assert sub.left.id.sym == "i"


def test_subscripted_assignment_target() -> None:
    expr = expr_of("a[0] = 1")
    assert isinstance(expr, ArithAssign)
    assert expr.target.id.kind is IdKind.VARIABLE_BINDER
    assert expr.target.subscript is not None


# -- increments and binders --------------------------------------------------


def test_prefix_increment_shape() -> None:
    expr = expr_of("++x")
    assert isinstance(expr, ArithUnary)
    assert expr.op == "++" and not expr.postfix
    operand = expr.operand
    assert isinstance(operand, ArithVar)
    assert operand.id.kind is IdKind.VARIABLE_BINDER


def test_postfix_decrement_shape() -> None:
    expr = expr_of("x--")
    assert isinstance(expr, ArithUnary)
    assert expr.op == "--" and expr.postfix
    operand = expr.operand
    assert isinstance(operand, ArithVar)
    assert operand.id.kind is IdKind.VARIABLE_BINDER


def test_prefix_increment_of_non_variable_is_double_plus() -> None:
    assert datum_of("++1") == (
        "arith-unary",
        "+",
        ("arith-unary", "+", num("1")),
    )


def test_arith_assignment_binds_later_ref() -> None:
    program = read_bound("echo $((x = 5))\necho $x")
    binders = find_ids(program, kind=IdKind.VARIABLE_BINDER, sym="x")
    assert len(binders) == 1
    ref = find_ids(program, kind=IdKind.VARIABLE_REF, sym="x")[0]
    binding = resolve(ref)
    assert binding is not None
    assert binding.kind is BindingKind.ASSIGNMENT
    assert binding.sites[0].id is binders[0]


def test_increment_registers_binder() -> None:
    program = read_bound("echo $((++n))\necho $((n++))\necho $n")
    binders = find_ids(program, kind=IdKind.VARIABLE_BINDER, sym="n")
    assert len(binders) == 2
    ref = find_ids(program, kind=IdKind.VARIABLE_REF, sym="n")[0]
    binding = resolve(ref)
    assert binding is not None
    assert binding.kind is BindingKind.ASSIGNMENT
    assert len(binding.sites) == 2


# -- embedded expansions -----------------------------------------------------


def test_embedded_param_and_cmdsub() -> None:
    program = read_bound("y=1\necho $(( ${y} + $(echo $y) ))")
    cmd = program.commands[1]
    assert isinstance(cmd, Simple)
    part = cmd.words[1].parts[0]
    assert isinstance(part, Arith)
    expr = part.expr
    assert isinstance(expr, ArithBinary) and expr.op == "+"
    left, right = expr.left, expr.right
    assert isinstance(left, ArithPart) and isinstance(left.part, Param)
    assert isinstance(right, ArithPart) and isinstance(right.part, CmdSub)
    assert right.part.program is not None
    for ref in find_ids(program, kind=IdKind.VARIABLE_REF, sym="y"):
        binding = resolve(ref)
        assert binding is not None
        assert binding.kind is BindingKind.ASSIGNMENT


def test_nested_arith_becomes_part() -> None:
    expr = expr_of("1 + $((2))")
    assert isinstance(expr, ArithBinary)
    right = expr.right
    assert isinstance(right, ArithPart)
    assert isinstance(right.part, Arith)
    inner = right.part.expr
    assert isinstance(inner, ArithNum) and inner.text == "2"


def test_empty_arith_is_zero() -> None:
    expr = expr_of("")
    assert isinstance(expr, ArithNum)
    assert expr.text == "0"
    assert expr.loc.synthetic


# -- errors ------------------------------------------------------------------


def test_parse_error_has_source_location() -> None:
    with pytest.raises(ShParseError) as excinfo:
        sh_read("echo $((1 +))")
    assert excinfo.value.loc.line == 1
    assert excinfo.value.loc.col == 11


def test_parse_error_on_later_line() -> None:
    with pytest.raises(ShParseError) as excinfo:
        sh_read("true\necho $((2 %% 3))")
    assert excinfo.value.loc.line == 2


def test_stray_dollar_raises() -> None:
    with pytest.raises(ShParseError):
        sh_read("echo $((1 + $))")
