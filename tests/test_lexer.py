"""Tokenization: operators, quoting, expansions, here-documents."""

from __future__ import annotations

import pytest

from sash import (
    Arith,
    CmdSub,
    CmdSubStyle,
    DQuote,
    Escape,
    HereDoc,
    IdKind,
    Lit,
    Param,
    ShId,
    ShParseError,
    Simple,
    SQuote,
    Word,
    sh_lex,
    sh_read,
)
from sash.nodes import TokEof, TokIoNumber, TokNewline, TokOp, TokWord


def op_syms(src: str) -> list[str]:
    return [t.id.sym for t in sh_lex(src) if isinstance(t, TokOp)]


def first_word(src: str) -> Word:
    program = sh_read(src)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    return cmd.words[0]


def flat_text(word: Word) -> str:
    out = []
    for part in word.parts:
        if isinstance(part, Lit):
            out.append(part.text)
        elif isinstance(part, Escape):
            out.append(part.ch)
        elif isinstance(part, SQuote):
            out.append(part.text)
        elif isinstance(part, DQuote):
            for sub in part.parts:
                if isinstance(sub, Lit):
                    out.append(sub.text)
                elif isinstance(sub, Escape):
                    out.append(sub.ch)
    return "".join(out)


def test_simple_words() -> None:
    toks = sh_lex("echo hi")
    assert [type(t).__name__ for t in toks] == ["TokWord", "TokWord", "TokEof"]


def test_control_operators() -> None:
    assert op_syms("a && b || c; d & e | f") == ["&&", "||", ";", "&", "|"]


def test_redirect_operators_maximal_munch() -> None:
    assert op_syms("a >>b <&3 <>c >|d >&2 <e") == [">>", "<&", "<>", ">|", ">&", "<"]


def test_io_number() -> None:
    toks = sh_lex("2>err")
    assert isinstance(toks[0], TokIoNumber) and toks[0].n == 2
    assert isinstance(toks[1], TokOp) and toks[1].id.sym == ">"


def test_digits_without_redirect_are_a_word() -> None:
    toks = sh_lex("echo 2")
    assert isinstance(toks[1], TokWord)


def test_comment_at_word_start_only() -> None:
    toks = sh_lex("echo hi # comment")
    assert [type(t).__name__ for t in toks] == ["TokWord", "TokWord", "TokEof"]
    assert flat_text(first_word("foo#bar")) == "foo#bar"


def test_newline_token() -> None:
    toks = sh_lex("a\nb")
    assert isinstance(toks[1], TokNewline)
    assert isinstance(toks[-1], TokEof)


def test_line_continuation() -> None:
    assert flat_text(first_word("ab\\\ncd")) == "abcd"


def test_single_quotes_are_raw() -> None:
    word = first_word("'a|$b'")
    assert len(word.parts) == 1
    part = word.parts[0]
    assert isinstance(part, SQuote) and part.text == "a|$b"


def test_double_quote_parts() -> None:
    word = first_word('"a$b"')
    part = word.parts[0]
    assert isinstance(part, DQuote)
    kinds = [type(p).__name__ for p in part.parts]
    assert kinds == ["Lit", "Param"]


def test_double_quote_escapes_special_only() -> None:
    word = first_word('"a\\$b"')
    part = word.parts[0]
    assert isinstance(part, DQuote)
    assert not any(isinstance(p, Param) for p in part.parts)
    assert flat_text(word) == "a$b"


def test_double_quote_backslash_before_ordinary_char_stays() -> None:
    assert flat_text(first_word('"a\\bc"')) == "a\\bc"


def test_dollar_name_kinds() -> None:
    program = sh_read("echo $x $1 $@ $?")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    kinds = []
    for word in cmd.words[1:]:
        part = word.parts[0]
        assert isinstance(part, Param)
        kinds.append(part.id.kind)
    assert kinds == [
        IdKind.VARIABLE_REF,
        IdKind.POSITIONAL,
        IdKind.SPECIAL_PARAM,
        IdKind.SPECIAL_PARAM,
    ]


def test_braced_param_operator() -> None:
    part = first_word("${x:-d}").parts[0]
    assert isinstance(part, Param)
    assert part.op == ":-"
    assert part.word is not None
    lit = part.word[0]
    assert isinstance(lit, Lit) and lit.text == "d"


def test_braced_param_double_percent() -> None:
    part = first_word("${x%%p}").parts[0]
    assert isinstance(part, Param)
    assert part.op == "%%"


def test_param_length() -> None:
    part = first_word("${#x}").parts[0]
    assert isinstance(part, Param)
    assert part.is_length


def test_arith_bare_name_is_ref() -> None:
    part = first_word("$((x + 1))").parts[0]
    assert isinstance(part, Arith)
    refs = [p for p in part.parts if isinstance(p, ShId)]
    assert [r.sym for r in refs] == ["x"]
    assert refs[0].kind is IdKind.VARIABLE_REF


def test_dollar_paren_paren_that_is_not_arith() -> None:
    part = first_word("$((echo hi) | cat)").parts[0]
    assert isinstance(part, CmdSub)
    assert part.program is not None


def test_backtick_cmdsub() -> None:
    part = first_word("`date`").parts[0]
    assert isinstance(part, CmdSub)
    assert part.style is CmdSubStyle.BACKTICK
    assert part.program is not None
    inner = part.program.commands[0]
    assert isinstance(inner, Simple)
    assert flat_text(inner.words[0]) == "date"


def test_backtick_escaped_dollar_is_live_inside() -> None:
    part = first_word("`echo \\$HOME`").parts[0]
    assert isinstance(part, CmdSub)
    assert part.program is not None
    inner = part.program.commands[0]
    assert isinstance(inner, Simple)
    assert any(isinstance(p, Param) for p in inner.words[1].parts)


def heredoc_of(src: str) -> HereDoc:
    program = sh_read(src)
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    target = cmd.redirects[0].target
    assert isinstance(target, HereDoc)
    return target


def test_heredoc_unquoted_body_expands() -> None:
    hd = heredoc_of("cat <<EOF\nhello $x\nEOF\n")
    assert not hd.quoted
    assert hd.body is not None
    assert any(isinstance(p, Param) for p in hd.body)


def test_heredoc_quoted_delim_raw_body() -> None:
    hd = heredoc_of("cat <<'EOF'\nhello $x\nEOF\n")
    assert hd.quoted
    assert hd.body is not None
    assert len(hd.body) == 1
    lit = hd.body[0]
    assert isinstance(lit, Lit) and lit.text == "hello $x\n"


def test_heredoc_strip_tabs() -> None:
    hd = heredoc_of("cat <<-EOF\n\tbody\n\tEOF\n")
    assert hd.strip_tabs
    assert hd.body is not None
    lit = hd.body[0]
    assert isinstance(lit, Lit) and lit.text == "body\n"


def test_two_heredocs_collected_in_order() -> None:
    program = sh_read("cat <<A <<B\none\nA\ntwo\nB\n")
    cmd = program.commands[0]
    assert isinstance(cmd, Simple)
    bodies = []
    for redirect in cmd.redirects:
        target = redirect.target
        assert isinstance(target, HereDoc)
        assert target.body is not None
        lit = target.body[0]
        assert isinstance(lit, Lit)
        bodies.append(lit.text)
    assert bodies == ["one\n", "two\n"]


def test_unterminated_heredoc_raises() -> None:
    with pytest.raises(ShParseError):
        sh_read("cat <<EOF\nbody\n")


def test_unterminated_squote_raises() -> None:
    with pytest.raises(ShParseError):
        sh_read("echo 'oops")


def test_unterminated_dquote_raises() -> None:
    with pytest.raises(ShParseError):
        sh_read('echo "oops')
