"""Process substitution: lexing, recursive parse, binding, dialect gating."""

from __future__ import annotations

import pytest

from sash import (
    BindingKind,
    Dialect,
    IdKind,
    Lit,
    ProcSub,
    ScopeKind,
    ShId,
    ShParseError,
    Simple,
    Word,
    find_ids,
    free_identifier_eq,
    iter_ids,
    program_to_datum,
    resolve,
    sh_read,
    sh_read_program,
)
from sash.lexer import Lexer
from sash.nodes import Program, TokEof, TokOp


def words_of(src: str) -> list[Word]:
    program = sh_read(src, dialect=Dialect.BASH)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    return cmd.words


def procsub_of(word: Word) -> ProcSub:
    assert len(word.parts) == 1
    part = word.parts[0]
    assert isinstance(part, ProcSub)
    return part


def refs(program: Program, sym: str) -> list[ShId]:
    return find_ids(program, kind=IdKind.VARIABLE_REF, sym=sym)


def op_syms(src: str, dialect: Dialect) -> list[str]:
    lexer = Lexer(src, dialect=dialect)
    syms: list[str] = []
    while True:
        tok = lexer.next_token()
        if isinstance(tok, TokEof):
            return syms
        if isinstance(tok, TokOp):
            syms.append(tok.id.sym)


# -- parsing -----------------------------------------------------------------


def test_diff_two_procsubs() -> None:
    words = words_of("diff <(sort a) <(sort b)")
    assert len(words) == 3
    for word in words[1:]:
        part = procsub_of(word)
        assert part.direction == "<"
        assert part.program is not None
        sub = part.program.commands[0]
        assert isinstance(sub, Simple)
        assert sub.cmd_id is not None and sub.cmd_id.sym == "sort"


def test_output_direction() -> None:
    part = procsub_of(words_of("tee >(gzip)")[1])
    assert part.direction == ">"
    assert part.raw == "gzip"
    assert part.program is not None


def test_procsub_word_continues_after_paren() -> None:
    word = words_of("echo <(a)x")[1]
    assert len(word.parts) == 2
    assert isinstance(word.parts[0], ProcSub)
    assert isinstance(word.parts[1], Lit) and word.parts[1].text == "x"


def test_procsub_as_redirect_target() -> None:
    program = sh_read("cmd < <(sort a)", dialect=Dialect.BASH)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    target = cmd.redirects[0].target
    assert isinstance(target, Word)
    part = target.parts[0]
    assert isinstance(part, ProcSub) and part.program is not None


def test_nested_procsub() -> None:
    outer = procsub_of(words_of("cat <(cat <(x))")[1])
    assert outer.program is not None
    sub = outer.program.commands[0]
    assert isinstance(sub, Simple)
    inner = sub.words[1].parts[0]
    assert isinstance(inner, ProcSub)
    assert inner.program is not None
    innermost = inner.program.commands[0]
    assert isinstance(innermost, Simple)
    assert innermost.cmd_id is not None and innermost.cmd_id.sym == "x"


def test_procsub_body_loc_offsets() -> None:
    part = procsub_of(words_of("diff <(sort a)")[1])
    assert part.program is not None
    sub = part.program.commands[0]
    assert isinstance(sub, Simple)
    assert sub.cmd_id is not None
    assert sub.cmd_id.loc.col == 7  # 'sort' starts after 'diff <('


def test_unterminated_procsub_raises() -> None:
    with pytest.raises(ShParseError) as excinfo:
        sh_read("echo <(foo", dialect=Dialect.BASH)
    assert "process substitution" in str(excinfo.value)


def test_io_number_before_procsub_raises() -> None:
    with pytest.raises(ShParseError):
        sh_read("echo 2<(x)", dialect=Dialect.BASH)


# -- POSIX dialect unchanged -------------------------------------------------


def test_posix_procsub_lexes_as_redirect_then_paren() -> None:
    assert op_syms("echo <(a)", Dialect.POSIX) == ["<", "(", ")"]


def test_posix_procsub_parse_error_pinned() -> None:
    for src in ("echo <(a)", "echo >(a)"):
        with pytest.raises(ShParseError) as excinfo:
            sh_read(src)
        assert "expected target" in str(excinfo.value)


# -- rendering ---------------------------------------------------------------


def test_procsub_ids_reachable() -> None:
    program = sh_read("diff <(sort $x) <(sort $y)", dialect=Dialect.BASH)
    syms = {i.sym for i in iter_ids(program)}
    assert {"sort", "x", "y"} <= syms


def test_procsub_datum() -> None:
    from sash.render import _datum_str

    program = sh_read("diff <(sort a) >(b)", dialect=Dialect.BASH)
    text = _datum_str(program_to_datum(program))
    assert "(proc-sub < (program" in text
    assert "(proc-sub > (program" in text


# -- binding -----------------------------------------------------------------


def test_procsub_reads_outer_variable() -> None:
    program, _store = sh_read_program("x=1\ncat <(echo $x)", dialect=Dialect.BASH)
    binding = resolve(refs(program, "x")[0])
    assert binding is not None
    assert binding.kind is BindingKind.ASSIGNMENT


def test_procsub_assignment_is_isolated() -> None:
    src = "x=1\ncat <(x=2; echo $x)\necho $x"
    program, _store = sh_read_program(src, dialect=Dialect.BASH)
    inner, outer = refs(program, "x")
    inner_binding = resolve(inner)
    outer_binding = resolve(outer)
    assert inner_binding is not None and outer_binding is not None
    assert inner_binding is not outer_binding
    assert len(outer_binding.sites) == 1
    assert not free_identifier_eq(inner, outer)


def test_procsub_scope_kind() -> None:
    program, _store = sh_read_program("cat <(echo $x)", dialect=Dialect.BASH)
    ref = refs(program, "x")[0]
    assert any(sc.kind is ScopeKind.PROC_SUB for sc in ref.scopes)
