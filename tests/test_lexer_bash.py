"""Bash-dialect word lexing: $'...', $"...", braces, ${} operators, redirects."""

from __future__ import annotations

import pytest

from sash import (
    Arith,
    ArithBinary,
    ArithGroup,
    ArithNum,
    ArithPart,
    ArithUnary,
    CmdSub,
    Dialect,
    DQuote,
    IdKind,
    Lit,
    Param,
    Redirect,
    ShParseError,
    Simple,
    SQuote,
    Word,
    iter_ids,
    sh_read,
    sh_read_program,
)
from sash.lexer import Lexer
from sash.nodes import AnsiCQuote, BraceExpand, TokEof, TokOp


def words_of(src: str, dialect: Dialect = Dialect.BASH) -> list[Word]:
    program = sh_read(src, dialect=dialect)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    return cmd.words


def bash_word(src: str) -> Word:
    return words_of(src)[0]


def posix_word(src: str) -> Word:
    return words_of(src, dialect=Dialect.POSIX)[0]


def ansi_text(src: str) -> str:
    part = bash_word(src).parts[0]
    assert isinstance(part, AnsiCQuote)
    return part.text


def param_of(src: str) -> Param:
    part = bash_word(src).parts[0]
    assert isinstance(part, Param)
    return part


def redirect_of(src: str, dialect: Dialect = Dialect.BASH) -> Redirect:
    program = sh_read(src, dialect=dialect)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    return cmd.redirects[0]


def op_syms(src: str, dialect: Dialect) -> list[str]:
    lexer = Lexer(src, dialect=dialect)
    syms: list[str] = []
    while True:
        tok = lexer.next_token()
        if isinstance(tok, TokEof):
            return syms
        if isinstance(tok, TokOp):
            syms.append(tok.id.sym)


# -- $'...' ------------------------------------------------------------------


def test_ansi_c_simple_escapes() -> None:
    assert ansi_text(r"$'a\nb\tc'") == "a\nb\tc"


def test_ansi_c_escape_char() -> None:
    assert ansi_text(r"$'\e\E'") == "\x1b\x1b"


def test_ansi_c_octal() -> None:
    assert ansi_text(r"$'\101\0'") == "A\0"


def test_ansi_c_hex() -> None:
    assert ansi_text(r"$'\x41'") == "A"
    assert ansi_text(r"$'\x411'") == "A1"  # at most two hex digits


def test_ansi_c_unicode() -> None:
    assert ansi_text("$'" + "\\" + "u0041'") == "A"
    assert ansi_text(r"$'\U00000042'") == "B"
    assert ansi_text("$'" + "\\" + "u00e9'") == "é"


def test_ansi_c_control() -> None:
    assert ansi_text(r"$'\cA'") == "\x01"
    assert ansi_text(r"$'\ca'") == "\x01"


def test_ansi_c_unknown_escape_stays_literal() -> None:
    assert ansi_text(r"$'\q'") == "\\q"


def test_ansi_c_bare_hex_prefix_stays_literal() -> None:
    assert ansi_text(r"$'\xg'") == "\\xg"


def test_ansi_c_escaped_quote() -> None:
    assert ansi_text(r"$'\''") == "'"


def test_ansi_c_raw_preserved() -> None:
    part = bash_word(r"$'a\n'").parts[0]
    assert isinstance(part, AnsiCQuote)
    assert part.raw == r"a\n"


def test_ansi_c_unterminated_raises() -> None:
    with pytest.raises(ShParseError):
        sh_read("echo $'oops", dialect=Dialect.BASH)


def test_posix_ansi_c_is_dollar_then_squote() -> None:
    word = posix_word("$'x'")
    assert len(word.parts) == 2
    assert isinstance(word.parts[0], Lit) and word.parts[0].text == "$"
    assert isinstance(word.parts[1], SQuote) and word.parts[1].text == "x"


def test_ansi_c_not_special_inside_dquote() -> None:
    part = bash_word("\"$'x'\"").parts[0]
    assert isinstance(part, DQuote)
    assert not any(isinstance(p, AnsiCQuote) for p in part.parts)


# -- $"..." ------------------------------------------------------------------


def test_locale_dquote() -> None:
    part = bash_word('$"hi"').parts[0]
    assert isinstance(part, DQuote) and part.locale


def test_plain_dquote_not_locale() -> None:
    part = bash_word('"hi"').parts[0]
    assert isinstance(part, DQuote) and not part.locale


def test_posix_locale_dquote_is_dollar_then_dquote() -> None:
    word = posix_word('$"hi"')
    assert isinstance(word.parts[0], Lit) and word.parts[0].text == "$"
    assert isinstance(word.parts[1], DQuote) and not word.parts[1].locale


# -- brace expansion ---------------------------------------------------------


def test_brace_alternates() -> None:
    part = bash_word("{a,b}").parts[0]
    assert isinstance(part, BraceExpand)
    assert part.range is None and part.alternates is not None
    assert len(part.alternates) == 2
    first = part.alternates[0][0]
    assert isinstance(first, Lit) and first.text == "a"


def test_brace_empty_alternate() -> None:
    part = bash_word("{a,}").parts[0]
    assert isinstance(part, BraceExpand)
    assert part.alternates is not None
    assert part.alternates[1] == []


def test_brace_prefix_suffix() -> None:
    parts = bash_word("x{a,b}y").parts
    assert isinstance(parts[0], Lit) and parts[0].text == "x"
    assert isinstance(parts[1], BraceExpand)
    assert isinstance(parts[2], Lit) and parts[2].text == "y"


def test_brace_nesting() -> None:
    part = bash_word("{a,{b,c}}").parts[0]
    assert isinstance(part, BraceExpand)
    assert part.alternates is not None
    inner = part.alternates[1][0]
    assert isinstance(inner, BraceExpand)
    assert inner.alternates is not None and len(inner.alternates) == 2


def test_brace_alternate_may_contain_expansions() -> None:
    part = bash_word("{$a,'b c'}").parts[0]
    assert isinstance(part, BraceExpand)
    assert part.alternates is not None
    assert isinstance(part.alternates[0][0], Param)
    assert isinstance(part.alternates[1][0], SQuote)


def test_brace_range_int() -> None:
    part = bash_word("{1..5}").parts[0]
    assert isinstance(part, BraceExpand)
    assert part.range == ("1", "5", None)


def test_brace_range_step_and_sign() -> None:
    part = bash_word("{-1..10..2}").parts[0]
    assert isinstance(part, BraceExpand)
    assert part.range == ("-1", "10", "2")


def test_brace_range_letters() -> None:
    part = bash_word("{a..z}").parts[0]
    assert isinstance(part, BraceExpand)
    assert part.range == ("a", "z", None)


def test_brace_no_comma_is_literal() -> None:
    word = bash_word("{a}")
    assert len(word.parts) == 1
    assert isinstance(word.parts[0], Lit) and word.parts[0].text == "{a}"


def test_brace_mixed_range_is_literal() -> None:
    part = bash_word("{1..x}").parts[0]
    assert isinstance(part, Lit) and part.text == "{1..x}"


def test_brace_unterminated_is_literal() -> None:
    word = bash_word("{a,b")
    assert isinstance(word.parts[0], Lit) and word.parts[0].text == "{a,b"


def test_brace_not_inside_dquote() -> None:
    part = bash_word('"{a,b}"').parts[0]
    assert isinstance(part, DQuote)
    assert not any(isinstance(p, BraceExpand) for p in part.parts)


def test_posix_braces_stay_literal() -> None:
    word = posix_word("{a,b}")
    assert len(word.parts) == 1
    assert isinstance(word.parts[0], Lit) and word.parts[0].text == "{a,b}"


def test_brace_alternate_ids_reachable() -> None:
    program = sh_read("echo {$a,$b}", dialect=Dialect.BASH)
    syms = {i.sym for i in iter_ids(program)}
    assert {"a", "b"} <= syms


def test_brace_cmdsub_in_alternate_filled() -> None:
    part = words_of("echo {$(date),b}")[1].parts[0]
    assert isinstance(part, BraceExpand)
    assert part.alternates is not None
    sub = part.alternates[0][0]
    assert isinstance(sub, CmdSub)
    assert sub.program is not None


# -- ${...} extended operators -----------------------------------------------


def test_param_replace() -> None:
    p = param_of("${x/a/b}")
    assert p.op == "/"
    assert p.pattern is not None and isinstance(p.pattern[0], Lit)
    assert p.word is not None and isinstance(p.word[0], Lit)


def test_param_replace_all_and_anchored() -> None:
    assert param_of("${x//a/b}").op == "//"
    assert param_of("${x/#a/b}").op == "/#"
    assert param_of("${x/%a/b}").op == "/%"


def test_param_replace_without_replacement() -> None:
    p = param_of("${x/a}")
    assert p.op == "/"
    assert p.pattern is not None and p.word is None


def test_param_replace_pattern_and_replacement_parts() -> None:
    program = sh_read("echo ${x/$y/$z}", dialect=Dialect.BASH)
    syms = {i.sym for i in iter_ids(program)}
    assert {"x", "y", "z"} <= syms


def test_param_substring() -> None:
    p = param_of("${x:1}")
    assert p.op == ":"
    assert isinstance(p.offset, ArithNum) and p.offset.text == "1"
    assert p.length is None


def test_param_substring_with_length() -> None:
    p = param_of("${x:1:2}")
    assert isinstance(p.offset, ArithNum)
    assert isinstance(p.length, ArithNum) and p.length.text == "2"


def test_param_substring_negative_offset_space() -> None:
    p = param_of("${x: -1}")
    assert isinstance(p.offset, ArithUnary) and p.offset.op == "-"


def test_param_substring_parenthesized_offset() -> None:
    p = param_of("${x:(-1):2}")
    assert isinstance(p.offset, ArithGroup)


def test_param_substring_arith_ids_reachable() -> None:
    program = sh_read("echo ${x:$off:$len}", dialect=Dialect.BASH)
    p = param_of("${x:$off:$len}")
    assert isinstance(p.offset, ArithPart)
    syms = {i.sym for i in iter_ids(program)}
    assert {"off", "len"} <= syms


def test_param_case_modification() -> None:
    assert param_of("${x^}").op == "^"
    assert param_of("${x,}").op == ","
    assert param_of("${x,,}").op == ",,"
    p = param_of("${x^^p}")
    assert p.op == "^^"
    assert p.word is not None and isinstance(p.word[0], Lit)


@pytest.mark.parametrize("transform", list("QEPAKauLU"))
def test_param_transforms(transform: str) -> None:
    p = param_of("${x@" + transform + "}")
    assert p.op == "@" + transform
    assert p.word is None


def test_param_bad_transform_raises() -> None:
    with pytest.raises(ShParseError):
        sh_read("echo ${x@Z}", dialect=Dialect.BASH)


def test_param_indirection() -> None:
    p = param_of("${!x}")
    assert p.indirect and p.op is None
    assert p.id.sym == "x" and p.id.kind is IdKind.VARIABLE_REF


def test_param_indirection_with_op() -> None:
    p = param_of("${!x:-d}")
    assert p.indirect and p.op == ":-"


def test_param_prefix_listing() -> None:
    star = param_of("${!p*}")
    assert star.op == "!*" and not star.indirect
    at = param_of("${!p@}")
    assert at.op == "!@" and not at.indirect
    assert star.id.sym == "p"


def test_param_assign_op_still_binder() -> None:
    p = param_of("${x:=v}")
    assert p.op == ":=" and p.id.kind is IdKind.VARIABLE_REF


def test_posix_extended_param_ops_raise() -> None:
    for src in ("echo ${x/a/b}", "echo ${x:1}", "echo ${x@Q}", "echo ${!x}"):
        with pytest.raises(ShParseError):
            sh_read(src)


# -- redirects ---------------------------------------------------------------


def test_bash_amp_redirect_operators() -> None:
    assert op_syms("a &>f", Dialect.BASH) == ["&>"]
    assert op_syms("a &>>f", Dialect.BASH) == ["&>>"]


def test_posix_amp_redirect_stays_two_operators() -> None:
    assert op_syms("a &>f", Dialect.POSIX) == ["&", ">"]
    assert op_syms("a &>>f", Dialect.POSIX) == ["&", ">>"]


def test_amp_redirect_parses() -> None:
    r = redirect_of("cmd &>log")
    assert r.op_id.sym == "&>" and isinstance(r.target, Word)
    assert redirect_of("cmd &>>log").op_id.sym == "&>>"


def test_here_string() -> None:
    r = redirect_of('cat <<< "hi $x"')
    assert r.here_string
    assert r.op_id.sym == "<<<"
    assert isinstance(r.target, Word)


def test_here_string_ids_reachable() -> None:
    program = sh_read("cat <<< $x", dialect=Dialect.BASH)
    assert "x" in {i.sym for i in iter_ids(program)}


def test_posix_here_string_not_lexed() -> None:
    assert op_syms("cat <<< x", Dialect.POSIX)[0] == "<<"
    with pytest.raises(ShParseError):
        sh_read("cat <<< x")


def test_here_string_inside_cmdsub_raw_capture() -> None:
    program = sh_read("echo $(cat <<< foo\necho hi)", dialect=Dialect.BASH)
    part = words_of("echo $(cat <<< foo\necho hi)")[1].parts[0]
    assert isinstance(part, CmdSub)
    assert part.program is not None
    assert len(part.program.commands) == 2
    assert len(program.commands) == 1


def test_fd_move() -> None:
    r = redirect_of("exec 3>&4-")
    assert r.move and r.n == 3 and r.op_id.sym == ">&"
    assert redirect_of("exec 3<&4-").move


def test_fd_close_is_not_move() -> None:
    assert not redirect_of("exec 3>&-").move


def test_posix_fd_move_not_flagged() -> None:
    assert not redirect_of("exec 3>&4-", dialect=Dialect.POSIX).move


def test_fd_var_redirect() -> None:
    r = redirect_of("cmd {fd}>out")
    assert r.fd_var is not None
    assert r.fd_var.sym == "fd"
    assert r.fd_var.kind is IdKind.VARIABLE_BINDER


def test_fd_var_ids_reachable_and_painted() -> None:
    program, _store = sh_read_program("cmd {fd}>out", dialect=Dialect.BASH)
    ids = [i for i in iter_ids(program) if i.sym == "fd"]
    assert len(ids) == 1
    assert ids[0].kind is IdKind.VARIABLE_BINDER
    assert ids[0].scopes


def test_fd_var_requires_adjacency() -> None:
    program = sh_read("cmd {fd} >out", dialect=Dialect.BASH)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.redirects[0].fd_var is None
    assert len(cmd.words) == 2


def test_fd_var_trailing_redirect() -> None:
    program = sh_read("while :; do :; done {fd}<in", dialect=Dialect.BASH)
    from sash import While

    cmd = program.commands[0]
    assert isinstance(cmd, While)
    assert cmd.redirects[0].fd_var is not None


def test_posix_fd_var_not_recognized() -> None:
    program = sh_read("cmd {fd}>out")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    assert cmd.redirects[0].fd_var is None
    assert len(cmd.words) == 2


# -- $[...] ------------------------------------------------------------------


def test_deprecated_arith() -> None:
    part = bash_word("$[x+1]").parts[0]
    assert isinstance(part, Arith) and part.deprecated
    assert isinstance(part.expr, ArithBinary) and part.expr.op == "+"


def test_dollar_paren_arith_not_deprecated() -> None:
    part = bash_word("$((x))").parts[0]
    assert isinstance(part, Arith) and not part.deprecated


def test_posix_dollar_bracket_is_literal() -> None:
    word = posix_word("$[x]")
    assert isinstance(word.parts[0], Lit) and word.parts[0].text == "$"
    assert isinstance(word.parts[1], Lit) and word.parts[1].text == "[x]"


# -- binding smoke -----------------------------------------------------------


def test_bind_paints_all_new_payload_ids() -> None:
    src = "echo ${x/$y/$z} ${w:$o:2} {$a,b} $'q' {fd}>log"
    program, _store = sh_read_program(src, dialect=Dialect.BASH)
    assert all(i.scopes for i in iter_ids(program))


def test_datum_renders_new_parts() -> None:
    from sash import program_to_datum
    from sash.render import _datum_str

    src = "echo $'q' $\"l\" ${x/$y/b} ${w:1:2} {a,{1..3}} $[n] <<< hi {fd}>&2-"
    program = sh_read(src, dialect=Dialect.BASH)
    text = _datum_str(program_to_datum(program))
    for expected in ("ansi-c", "locale", "pattern", "offset", "brace", "fd-var"):
        assert expected in text
