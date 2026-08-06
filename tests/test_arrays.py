"""Bash arrays: assignment forms, parameter subscripts, one-cell binding."""

from __future__ import annotations

import pytest

from sash import (
    ArithBinary,
    ArithNum,
    Assign,
    Binding,
    BindingKind,
    Dialect,
    IdKind,
    Lit,
    Namespace,
    Param,
    Program,
    ShId,
    ShParseError,
    Simple,
    Word,
    find_ids,
    iter_ids,
    resolve,
    sh_read,
    sh_read_program,
)


def first_simple(src: str, dialect: Dialect = Dialect.BASH) -> Simple:
    program = sh_read(src, dialect=dialect)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    return cmd


def first_assign(src: str, dialect: Dialect = Dialect.BASH) -> Assign:
    return first_simple(src, dialect).assigns[0]


def param_of(src: str) -> Param:
    word = first_simple(src).words[1]
    part = word.parts[0]
    assert isinstance(part, Param)
    return part


def lit_text(word: Word) -> str:
    assert len(word.parts) == 1 and isinstance(word.parts[0], Lit)
    return word.parts[0].text


def read_bash(src: str) -> Program:
    program, _store = sh_read_program(src, dialect=Dialect.BASH)
    return program


def must_resolve(sh_id: ShId) -> Binding:
    binding = resolve(sh_id, Namespace.VARIABLE)
    assert binding is not None
    return binding


# -- subscripted assignments ---------------------------------------------------


def test_subscript_assignment() -> None:
    assign = first_assign("a[1]=x")
    assert assign.name_id.sym == "a"
    assert assign.name_id.kind is IdKind.VARIABLE_BINDER
    assert isinstance(assign.subscript, ArithNum) and assign.subscript.text == "1"
    assert assign.subscript_raw == "1"
    assert not assign.append
    assert lit_text(assign.word) == "x"


def test_subscript_arith_expression() -> None:
    assign = first_assign("a[i+1]=x")
    assert isinstance(assign.subscript, ArithBinary) and assign.subscript.op == "+"


def test_subscript_arith_refs_resolve() -> None:
    program = read_bash("i=0\na[i+1]=x")
    ref = find_ids(program, kind=IdKind.VARIABLE_REF, sym="i")[0]
    assert must_resolve(ref).kind is BindingKind.ASSIGNMENT


def test_subscript_arith_side_effect_binds() -> None:
    program = read_bash("a[x++]=v\necho $x")
    ref = find_ids(program, kind=IdKind.VARIABLE_REF, sym="x")[0]
    assert must_resolve(ref).kind is BindingKind.ASSIGNMENT


def test_assoc_subscript_keeps_raw() -> None:
    cmd = first_simple("a[foo bar]=x")
    assert cmd.words == []
    assign = cmd.assigns[0]
    assert assign.name_id.sym == "a"
    assert assign.subscript is None
    assert assign.subscript_raw == "foo bar"
    assert lit_text(assign.word) == "x"


def test_subscript_not_joined_outside_assignment_position() -> None:
    cmd = first_simple("echo a[foo bar]=x")
    assert len(cmd.words) == 3  # echo, a[foo, bar]=x — as in bash


# -- compound assignments ------------------------------------------------------


def test_compound_assignment() -> None:
    assign = first_assign("a=(1 2 3)")
    assert assign.word.parts == []
    assert assign.array_items is not None and len(assign.array_items) == 3
    for item, expected in zip(assign.array_items, ["1", "2", "3"]):
        assert item.subscript is None and item.subscript_raw is None
        assert lit_text(item.word) == expected


def test_compound_assignment_with_subscripts() -> None:
    assign = first_assign("a=([0]=x [1]=y)")
    assert assign.array_items is not None
    first, second = assign.array_items
    assert isinstance(first.subscript, ArithNum) and first.subscript.text == "0"
    assert lit_text(first.word) == "x"
    assert isinstance(second.subscript, ArithNum) and second.subscript.text == "1"
    assert lit_text(second.word) == "y"


def test_compound_assignment_multiline() -> None:
    assign = first_assign("a=(\n  1\n  2\n)")
    assert assign.array_items is not None and len(assign.array_items) == 2


def test_compound_requires_adjacent_paren() -> None:
    with pytest.raises(ShParseError):
        sh_read("a= (1)", dialect=Dialect.BASH)


# -- append --------------------------------------------------------------------


def test_array_append() -> None:
    assign = first_assign("a+=(4)")
    assert assign.append
    assert assign.array_items is not None and len(assign.array_items) == 1


def test_scalar_append() -> None:
    assign = first_assign("x+=v")
    assert assign.append
    assert assign.array_items is None
    assert lit_text(assign.word) == "v"


# -- parameter subscripts ------------------------------------------------------


def test_param_subscript_index() -> None:
    p = param_of("echo ${a[0]}")
    assert p.id.sym == "a" and p.id.kind is IdKind.VARIABLE_REF
    assert isinstance(p.subscript, ArithNum) and p.subscript.text == "0"
    assert p.subscript_raw == "0"


def test_param_subscript_at_and_star() -> None:
    at = param_of("echo ${a[@]}")
    assert isinstance(at.subscript, Lit) and at.subscript.text == "@"
    star = param_of("echo ${a[*]}")
    assert isinstance(star.subscript, Lit) and star.subscript.text == "*"


def test_param_length_with_subscript() -> None:
    p = param_of("echo ${#a[@]}")
    assert p.is_length
    assert isinstance(p.subscript, Lit) and p.subscript.text == "@"


def test_param_array_keys() -> None:
    p = param_of("echo ${!a[@]}")
    assert p.indirect
    assert isinstance(p.subscript, Lit) and p.subscript.text == "@"
    star = param_of("echo ${!a[*]}")
    assert star.indirect
    assert isinstance(star.subscript, Lit) and star.subscript.text == "*"


def test_param_subscript_combines_with_default_op() -> None:
    p = param_of("echo ${a[0]:-d}")
    assert p.op == ":-"
    assert isinstance(p.subscript, ArithNum)
    assert p.word is not None and isinstance(p.word[0], Lit)


def test_param_subscript_combines_with_replace_op() -> None:
    p = param_of("echo ${a[@]/p/r}")
    assert p.op == "/"
    assert isinstance(p.subscript, Lit) and p.subscript.text == "@"
    assert p.pattern is not None


def test_param_assoc_subscript_keeps_raw() -> None:
    p = param_of("echo ${a[foo bar]}")
    assert p.subscript is None
    assert p.subscript_raw == "foo bar"


def test_param_subscript_ref_resolves() -> None:
    program = read_bash("i=1\necho ${a[i]}")
    ref = find_ids(program, kind=IdKind.VARIABLE_REF, sym="i")[0]
    assert must_resolve(ref).kind is BindingKind.ASSIGNMENT


def test_bare_dollar_name_bracket_is_not_array() -> None:
    word = first_simple("echo $a[i]").words[1]
    assert isinstance(word.parts[0], Param)
    assert word.parts[0].subscript is None
    assert isinstance(word.parts[1], Lit) and word.parts[1].text == "[i]"


# -- one-cell binding ----------------------------------------------------------


def test_ref_resolves_to_compound_assignment() -> None:
    program = read_bash("a=(1 2)\necho ${a[0]}")
    ref = find_ids(program, kind=IdKind.VARIABLE_REF, sym="a")[0]
    binding = must_resolve(ref)
    assert binding.kind is BindingKind.ASSIGNMENT
    assert len(binding.sites) == 1


def test_one_cell_across_subscripts() -> None:
    program = read_bash("a[5]=x\necho ${a[2]}")
    ref = find_ids(program, kind=IdKind.VARIABLE_REF, sym="a")[0]
    binding = must_resolve(ref)
    assert binding.kind is BindingKind.ASSIGNMENT
    assert binding.sites[0].kind == "binder"


def test_all_array_ids_painted() -> None:
    src = "a=(1 [n+1]=$x)\nb[i]=3\necho ${a[@]:1:2} ${b[j]:-d}\n"
    program = read_bash(src)
    assert all(i.scopes for i in iter_ids(program))


def test_compound_item_ids_reachable() -> None:
    program = sh_read("a=($x [i]=$y)", dialect=Dialect.BASH)
    syms = {i.sym for i in iter_ids(program)}
    assert {"x", "y", "i"} <= syms


# -- rendering -----------------------------------------------------------------


def test_datum_renders_array_forms() -> None:
    from sash import program_to_datum
    from sash.render import _datum_str

    src = "a=(1 [2]=x)\nb[i]=1\nc+=(y)\necho ${a[@]} ${#a[@]}"
    program = sh_read(src, dialect=Dialect.BASH)
    text = _datum_str(program_to_datum(program))
    for expected in ("array", "item", "subscript", "append"):
        assert expected in text


# -- POSIX pins ----------------------------------------------------------------


def test_posix_subscript_assignment_is_a_word() -> None:
    cmd = first_simple("a[1]=x", dialect=Dialect.POSIX)
    assert cmd.assigns == []
    assert len(cmd.words) == 1
    assert lit_text(cmd.words[0]) == "a[1]=x"
    assert cmd.cmd_id is not None and cmd.cmd_id.sym == "a[1]=x"


def test_posix_append_is_a_word() -> None:
    cmd = first_simple("x+=v", dialect=Dialect.POSIX)
    assert cmd.assigns == []
    assert lit_text(cmd.words[0]) == "x+=v"


def test_posix_compound_assignment_is_an_error() -> None:
    with pytest.raises(ShParseError):
        sh_read("a=(1)")


def test_posix_param_subscript_is_an_error() -> None:
    with pytest.raises(ShParseError):
        sh_read("echo ${a[0]}")
