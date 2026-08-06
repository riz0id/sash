"""POSIX sh tokenizer (POSIX.1-2017 XCU 2.2, 2.3, 2.6, 2.7).

Pull lexer driven by the parser. Emits operator tokens (whose ShId leaves
are classified as identifiers at read time), word tokens (part sequences
preserving quote structure), IO numbers, and newlines. Reserved words are
NOT recognized here — the parser promotes them positionally.

Command substitutions are captured as raw text (statically locatable staged
re-reads); the parser fills in their parsed programs. Here-document bodies
are collected when the newline token that triggers them is lexed; the
HereDoc placeholder is shared with the parser via `heredoc_for`.
"""

from __future__ import annotations

from typing import Literal, cast

from .nodes import (
    Arith,
    ArithPart,
    CmdSub,
    CmdSubStyle,
    DQuote,
    DQuotePart,
    Escape,
    HereDoc,
    IdKind,
    Lit,
    Loc,
    Param,
    ShId,
    ShParseError,
    SQuote,
    TokEof,
    TokIoNumber,
    TokNewline,
    TokOp,
    TokWord,
    Token,
    Word,
    WordPart,
    is_valid_name,
)

OPERATORS = [
    "<<-",
    "<<",
    "<&",
    "<>",
    ">>",
    ">&",
    ">|",
    "&&",
    "||",
    ";;",
    "<",
    ">",
    "&",
    "|",
    ";",
    "(",
    ")",
]

WORD_DELIMS = " \t\n|&;<>()"
SPECIAL_PARAM_CHARS = "@*#?-$!"
_NAME_START = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz_"
_NAME_CHARS = _NAME_START + "0123456789"

Mode = Literal["word", "dquote", "heredoc"]

_Mark = tuple[int, int, int]  # pos, line, col


class Lexer:
    def __init__(
        self,
        text: str,
        src: str = "<sh>",
        *,
        line: int = 1,
        col: int = 0,
        synthetic: bool = False,
    ) -> None:
        self.text = text
        self.src = src
        self.pos = 0
        self.line = line
        self.col = col
        self.synthetic = synthetic
        self._await_delim: ShId | None = None
        self._pending_heredocs: list[tuple[HereDoc, str]] = []
        self._heredoc_by_word: dict[int, HereDoc] = {}

    # -- position bookkeeping ------------------------------------------------

    def _mark(self) -> _Mark:
        return (self.pos, self.line, self.col)

    def _restore(self, mark: _Mark) -> None:
        self.pos, self.line, self.col = mark

    def _loc(self, mark: _Mark) -> Loc:
        pos, line, col = mark
        return Loc(self.src, line, col, pos, self.pos - pos, self.synthetic)

    def _peek(self, k: int = 0) -> str:
        i = self.pos + k
        return self.text[i] if i < len(self.text) else ""

    def _adv(self, n: int = 1) -> None:
        for _ in range(n):
            if self.pos >= len(self.text):
                return
            if self.text[self.pos] == "\n":
                self.line += 1
                self.col = 0
            else:
                self.col += 1
            self.pos += 1

    def _error(self, message: str, mark: _Mark | None = None) -> ShParseError:
        return ShParseError(
            message, self._loc(mark if mark is not None else self._mark())
        )

    # -- token loop ----------------------------------------------------------

    def next_token(self) -> Token:
        tok = self._scan_token()
        if isinstance(tok, TokWord) and self._await_delim is not None:
            self._register_heredoc(self._await_delim, tok.word)
            self._await_delim = None
        elif isinstance(tok, TokOp) and tok.id.sym in ("<<", "<<-"):
            self._await_delim = tok.id
        else:
            self._await_delim = None
        return tok

    def _skip_blanks(self) -> None:
        while True:
            ch = self._peek()
            if ch == " " or ch == "\t":
                self._adv()
            elif ch == "\\" and self._peek(1) == "\n":
                self._adv(2)
            else:
                return

    def _scan_token(self) -> Token:
        self._skip_blanks()
        mark = self._mark()
        ch = self._peek()
        if ch == "":
            return TokEof(self._loc(mark))
        if ch == "#":
            while self._peek() not in ("", "\n"):
                self._adv()
            return self._scan_token()
        if ch == "\n":
            self._adv()
            loc = self._loc(mark)
            self._collect_heredocs()
            return TokNewline(loc)
        if ch.isdigit():
            j = self.pos
            while j < len(self.text) and self.text[j].isdigit():
                j += 1
            if j < len(self.text) and self.text[j] in "<>":
                n = int(self.text[self.pos : j])
                self._adv(j - self.pos)
                return TokIoNumber(n, self._loc(mark))
        for op in OPERATORS:
            if self.text.startswith(op, self.pos):
                self._adv(len(op))
                loc = self._loc(mark)
                return TokOp(ShId(op, IdKind.OPERATOR, loc), loc)
        word = self._lex_word()
        return TokWord(word, word.loc)

    # -- words ---------------------------------------------------------------

    def _lex_word(self) -> Word:
        mark = self._mark()
        parts = self._lex_parts(stop=WORD_DELIMS, mode="word")
        if not parts:
            raise self._error(f"unexpected character {self._peek()!r}")
        return Word(parts, self._loc(mark))

    def _lex_parts(self, *, stop: str, mode: Mode) -> list[WordPart]:
        parts: list[WordPart] = []
        lit_mark: _Mark | None = None
        lit_chars: list[str] = []

        def flush() -> None:
            nonlocal lit_mark
            if lit_chars:
                assert lit_mark is not None
                parts.append(Lit("".join(lit_chars), self._loc(lit_mark)))
                lit_chars.clear()
            lit_mark = None

        def lit(ch: str, n: int) -> None:
            nonlocal lit_mark
            if lit_mark is None:
                lit_mark = self._mark()
            lit_chars.append(ch)
            self._adv(n)

        while True:
            ch = self._peek()
            if ch == "" or (stop and ch in stop and mode != "heredoc"):
                break
            if ch == "\\":
                nxt = self._peek(1)
                if nxt == "\n":
                    self._adv(2)
                    continue
                if mode == "word":
                    if nxt == "":
                        lit("\\", 1)
                        continue
                    flush()
                    mark = self._mark()
                    self._adv(2)
                    parts.append(Escape(nxt, self._loc(mark)))
                    continue
                # dquote / heredoc: backslash is special only before a few chars
                specials = '$`"\\' if mode == "dquote" else "$`\\"
                if nxt in specials:
                    flush()
                    mark = self._mark()
                    self._adv(2)
                    parts.append(Escape(nxt, self._loc(mark)))
                else:
                    lit("\\", 1)
                continue
            if ch == "'" and mode == "word":
                flush()
                parts.append(self._lex_squote())
                continue
            if ch == '"' and mode == "word":
                flush()
                parts.append(self._lex_dquote())
                continue
            if ch == "$":
                flush()
                parts.append(self._lex_dollar(mode))
                continue
            if ch == "`":
                flush()
                parts.append(self._lex_backtick())
                continue
            lit(ch, 1)
        flush()
        return parts

    def _lex_squote(self) -> SQuote:
        mark = self._mark()
        self._adv()  # '
        start = self.pos
        while self._peek() != "'":
            if self._peek() == "":
                raise self._error("unterminated single quote", mark)
            self._adv()
        text = self.text[start : self.pos]
        self._adv()  # '
        return SQuote(text, self._loc(mark))

    def _lex_dquote(self) -> DQuote:
        mark = self._mark()
        self._adv()  # "
        parts = self._lex_parts(stop='"', mode="dquote")
        if self._peek() != '"':
            raise self._error("unterminated double quote", mark)
        self._adv()  # "
        return DQuote(cast("list[DQuotePart]", parts), self._loc(mark))

    # -- $... ----------------------------------------------------------------

    def _lex_dollar(self, mode: Mode) -> WordPart:
        mark = self._mark()
        self._adv()  # $
        ch = self._peek()
        if ch == "(":
            if self._peek(1) == "(":
                arith = self._try_arith(mark)
                if arith is not None:
                    return arith
            return self._lex_dollar_cmdsub(mark)
        if ch == "{":
            return self._lex_braced_param(mark, mode)
        if ch in _NAME_START:
            start = self.pos
            while self._peek() in _NAME_CHARS and self._peek() != "":
                self._adv()
            name = self.text[start : self.pos]
            loc = self._loc(mark)
            return Param(ShId(name, IdKind.VARIABLE_REF, loc), None, None, False, loc)
        if ch.isdigit():
            self._adv()  # a single digit, per POSIX 2.5.1
            loc = self._loc(mark)
            kind = IdKind.SPECIAL_PARAM if ch == "0" else IdKind.POSITIONAL
            return Param(ShId(ch, kind, loc), None, None, False, loc)
        if ch in SPECIAL_PARAM_CHARS:
            self._adv()
            loc = self._loc(mark)
            return Param(ShId(ch, IdKind.SPECIAL_PARAM, loc), None, None, False, loc)
        # lone $ is literal
        return Lit("$", self._loc(mark))

    def _read_param_name(self) -> tuple[str, IdKind] | None:
        ch = self._peek()
        if ch in _NAME_START:
            start = self.pos
            while self._peek() in _NAME_CHARS and self._peek() != "":
                self._adv()
            return (self.text[start : self.pos], IdKind.VARIABLE_REF)
        if ch.isdigit():
            start = self.pos
            while self._peek().isdigit():
                self._adv()
            name = self.text[start : self.pos]
            kind = IdKind.SPECIAL_PARAM if name == "0" else IdKind.POSITIONAL
            return (name, kind)
        if ch in SPECIAL_PARAM_CHARS:
            self._adv()
            return (ch, IdKind.SPECIAL_PARAM)
        return None

    def _lex_braced_param(self, mark: _Mark, mode: Mode) -> Param:
        self._adv()  # {
        if self._peek() == "#" and self._peek(1) != "}":
            # ${#name} — length form; ${#} itself is the special parameter #
            self._adv()
            name_mark = self._mark()
            named = self._read_param_name()
            if named is None or self._peek() != "}":
                raise self._error("bad substitution", mark)
            self._adv()  # }
            name, kind = named
            loc = self._loc(mark)
            return Param(ShId(name, kind, self._loc(name_mark)), None, None, True, loc)
        name_mark = self._mark()
        named = self._read_param_name()
        if named is None:
            raise self._error("bad substitution", mark)
        name, kind = named
        name_id = ShId(name, kind, self._loc(name_mark))
        if self._peek() == "}":
            self._adv()
            return Param(name_id, None, None, False, self._loc(mark))
        op = self._read_param_op()
        if op is None:
            raise self._error("bad substitution", mark)
        word = self._lex_parts(stop="}", mode=mode if mode == "dquote" else "word")
        if self._peek() != "}":
            raise self._error("unterminated ${...}", mark)
        self._adv()  # }
        return Param(name_id, op, word, False, self._loc(mark))

    def _read_param_op(self) -> str | None:
        ch = self._peek()
        if ch == "":
            return None
        if ch == ":":
            nxt = self._peek(1)
            if nxt != "" and nxt in "-=?+":
                self._adv(2)
                return ":" + nxt
            return None
        if ch in "-=?+":
            self._adv()
            return ch
        if ch in "%#":
            if self._peek(1) == ch:
                self._adv(2)
                return ch * 2
            self._adv()
            return ch
        return None

    # -- $((...)) ------------------------------------------------------------

    def _try_arith(self, mark: _Mark) -> Arith | None:
        """After `$`, at `((`: arithmetic iff the parens close with `))`.

        Otherwise rewind and let `$(` capture a command substitution whose
        program begins with a subshell (the pragmatic dash/bash reading of
        POSIX's "shall determine whether it can be parsed as arithmetic").
        """
        entry = self._mark()
        self._adv(2)  # ((
        content_mark = self._mark()
        depth = 0
        while True:
            ch = self._peek()
            if ch == "":
                self._restore(entry)
                return None
            if ch == "\\":
                self._adv(2)
                continue
            if ch == "'":
                self._skip_squote_raw()
                continue
            if ch == '"':
                self._skip_dquote_raw()
                continue
            if ch == "`":
                self._skip_backtick_raw()
                continue
            if ch == "(":
                depth += 1
                self._adv()
                continue
            if ch == ")":
                if depth > 0:
                    depth -= 1
                    self._adv()
                    continue
                if self._peek(1) == ")":
                    content = self.text[content_mark[0] : self.pos]
                    parts = _lex_arith_text(content, self._loc(content_mark))
                    self._adv(2)  # ))
                    return Arith(parts, self._loc(mark))
                self._restore(entry)
                return None
            self._adv()

    # -- $(...) and `...` ----------------------------------------------------

    def _lex_dollar_cmdsub(self, mark: _Mark) -> CmdSub:
        self._adv()  # (
        content_mark = self._mark()
        self._scan_dollar_paren(mark)
        raw = self.text[content_mark[0] : self.pos - 1]
        pos, line, col = content_mark
        raw_loc = Loc(self.src, line, col, pos, len(raw), self.synthetic)
        return CmdSub(raw, CmdSubStyle.DOLLAR, self._loc(mark), raw_loc)

    def _lex_backtick(self) -> CmdSub:
        mark = self._mark()
        self._adv()  # `
        chars: list[str] = []
        while True:
            ch = self._peek()
            if ch == "":
                raise self._error("unterminated backquoted command substitution", mark)
            if ch == "`":
                self._adv()
                break
            if ch == "\\":
                nxt = self._peek(1)
                if nxt in ("$", "`", "\\"):
                    chars.append(nxt)
                    self._adv(2)
                    continue
                chars.append("\\")
                self._adv()
                continue
            chars.append(ch)
            self._adv()
        raw = "".join(chars)
        pos, line, col = mark
        raw_loc = Loc(self.src, line, col + 1, pos + 1, len(raw), True)
        return CmdSub(raw, CmdSubStyle.BACKTICK, self._loc(mark), raw_loc)

    # -- raw-capture scanner for $(...) --------------------------------------
    #
    # Consumes source through the `)` matching an already-consumed `(`. Must
    # not be fooled by `)` inside quotes, comments, here-doc bodies, nested
    # substitutions, or the unbalanced `)` closing a `case` pattern.

    def _scan_dollar_paren(self, outer_mark: _Mark) -> None:
        expecting_in = 0  # `case` words seen, awaiting their `in`
        case_depth = 0  # open case bodies at this nesting level
        cmd_pos = True
        word = ""
        pending: list[tuple[str, bool]] = []  # (delimiter, strip_tabs)

        def flush() -> None:
            nonlocal word, cmd_pos, expecting_in, case_depth
            if not word:
                return
            if cmd_pos and word == "case":
                expecting_in += 1
            elif word == "in" and expecting_in > 0:
                expecting_in -= 1
                case_depth += 1
            elif word == "esac" and case_depth > 0:
                case_depth -= 1
            cmd_pos = word in (
                "if",
                "then",
                "elif",
                "else",
                "while",
                "until",
                "do",
                "{",
                "!",
            )
            word = ""

        def consume_heredoc_bodies() -> None:
            while pending:
                delim, strip = pending.pop(0)
                while True:
                    if self._peek() == "":
                        raise self._error(
                            "unterminated here-document in command substitution",
                            outer_mark,
                        )
                    start = self.pos
                    while self._peek() not in ("", "\n"):
                        self._adv()
                    line_text = self.text[start : self.pos]
                    if self._peek() == "\n":
                        self._adv()
                    if (line_text.lstrip("\t") if strip else line_text) == delim:
                        break

        while True:
            ch = self._peek()
            if ch == "":
                raise self._error("unterminated command substitution", outer_mark)
            if ch == "\\":
                self._adv(2 if self._peek(1) != "" else 1)
                word += "\\"
                continue
            if ch == "'":
                flush()
                self._skip_squote_raw()
                cmd_pos = False
                continue
            if ch == '"':
                flush()
                self._skip_dquote_raw()
                cmd_pos = False
                continue
            if ch == "`":
                flush()
                self._skip_backtick_raw()
                cmd_pos = False
                continue
            if ch == "$":
                self._adv()
                if self._peek() == "(":
                    self._adv()
                    self._scan_dollar_paren(outer_mark)
                    cmd_pos = False
                else:
                    word += "$"
                continue
            if ch == "#" and word == "":
                while self._peek() not in ("", "\n"):
                    self._adv()
                continue
            if ch == "(":
                flush()
                self._adv()
                self._scan_dollar_paren(outer_mark)
                cmd_pos = False
                continue
            if ch == ")":
                flush()
                if case_depth > 0:
                    self._adv()
                    cmd_pos = True
                    continue
                self._adv()
                return
            if ch == "<" and self._peek(1) == "<" and self._peek(2) != "&":
                flush()
                self._adv(2)
                strip = self._peek() == "-"
                if strip:
                    self._adv()
                while self._peek() in (" ", "\t"):
                    self._adv()
                delim_chars: list[str] = []
                while (
                    self._peek() not in ("", "\n") and self._peek() not in WORD_DELIMS
                ):
                    c = self._peek()
                    if c == "\\":
                        self._adv()
                        delim_chars.append(self._peek())
                        self._adv()
                    elif c in "'\"":
                        quote = c
                        self._adv()
                        while self._peek() not in ("", quote):
                            delim_chars.append(self._peek())
                            self._adv()
                        self._adv()
                    else:
                        delim_chars.append(c)
                        self._adv()
                pending.append(("".join(delim_chars), strip))
                cmd_pos = False
                continue
            if ch == "\n":
                flush()
                self._adv()
                consume_heredoc_bodies()
                cmd_pos = True
                continue
            if ch in " \t":
                flush()
                self._adv()
                continue
            if ch in ";&|":
                flush()
                self._adv()
                cmd_pos = True
                continue
            word += ch
            self._adv()

    def _skip_squote_raw(self) -> None:
        mark = self._mark()
        self._adv()
        while self._peek() != "'":
            if self._peek() == "":
                raise self._error("unterminated single quote", mark)
            self._adv()
        self._adv()

    def _skip_dquote_raw(self) -> None:
        mark = self._mark()
        self._adv()
        while True:
            ch = self._peek()
            if ch == "":
                raise self._error("unterminated double quote", mark)
            if ch == '"':
                self._adv()
                return
            if ch == "\\":
                self._adv(2 if self._peek(1) != "" else 1)
                continue
            if ch == "`":
                self._skip_backtick_raw()
                continue
            if ch == "$" and self._peek(1) == "(":
                self._adv(2)
                self._scan_dollar_paren(mark)
                continue
            self._adv()

    def _skip_backtick_raw(self) -> None:
        mark = self._mark()
        self._adv()
        while True:
            ch = self._peek()
            if ch == "":
                raise self._error("unterminated backquoted command substitution", mark)
            if ch == "`":
                self._adv()
                return
            if ch == "\\":
                self._adv(2 if self._peek(1) != "" else 1)
                continue
            self._adv()

    # -- here-documents -------------------------------------------------------

    def heredoc_for(self, word: Word) -> HereDoc | None:
        return self._heredoc_by_word.pop(id(word), None)

    def _register_heredoc(self, op_id: ShId, word: Word) -> None:
        quoted = any(not isinstance(p, Lit) for p in word.parts)
        delim = _delimiter_text(word)
        hd = HereDoc(op_id, word, quoted, op_id.sym == "<<-", word.loc)
        self._pending_heredocs.append((hd, delim))
        self._heredoc_by_word[id(word)] = hd

    def _collect_heredocs(self) -> None:
        while self._pending_heredocs:
            hd, delim = self._pending_heredocs.pop(0)
            body_mark = self._mark()
            lines: list[str] = []
            terminated = False
            while self.pos < len(self.text):
                start = self.pos
                while self._peek() not in ("", "\n"):
                    self._adv()
                line_text = self.text[start : self.pos]
                if self._peek() == "\n":
                    self._adv()
                cmp = line_text.lstrip("\t") if hd.strip_tabs else line_text
                if cmp == delim:
                    terminated = True
                    break
                lines.append(cmp if hd.strip_tabs else line_text)
            if not terminated:
                raise self._error(
                    f"here-document delimited by end of file (wanted {delim!r})",
                    body_mark,
                )
            body_text = "".join(line_text + "\n" for line_text in lines)
            _, line, col = body_mark
            if hd.quoted:
                loc = Loc(self.src, line, col, body_mark[0], len(body_text), True)
                hd.body = [Lit(body_text, loc)] if body_text else []
            else:
                sub = Lexer(body_text, self.src, line=line, synthetic=True)
                hd.body = sub._lex_parts(stop="", mode="heredoc")


def _delimiter_text(word: Word) -> str:
    chars: list[str] = []
    for part in word.parts:
        if isinstance(part, Lit) or isinstance(part, SQuote):
            chars.append(part.text)
        elif isinstance(part, Escape):
            chars.append(part.ch)
        elif isinstance(part, DQuote):
            for inner in part.parts:
                if isinstance(inner, Lit):
                    chars.append(inner.text)
                elif isinstance(inner, Escape):
                    chars.append(inner.ch)
                else:
                    raise ShParseError(
                        "here-document delimiter must be a literal word", word.loc
                    )
        else:
            raise ShParseError(
                "here-document delimiter must be a literal word", word.loc
            )
    return "".join(chars)


def _lex_arith_text(text: str, base: Loc) -> list[ArithPart]:
    """Extract variable references from $((...)) content.

    `$x`, `${x}`, `$(...)` and backticks are lexed for real; bare NAMEs
    surface as VARIABLE_REF identifiers (POSIX arithmetic treats them as
    variables); everything else remains literal text. No expression AST.
    """
    sub = Lexer(text, base.src, line=base.line, col=base.col, synthetic=base.synthetic)
    parts: list[ArithPart] = []
    lit_mark: _Mark | None = None
    lit_chars: list[str] = []

    def flush() -> None:
        nonlocal lit_mark
        if lit_chars:
            assert lit_mark is not None
            parts.append(Lit("".join(lit_chars), sub._loc(lit_mark)))
            lit_chars.clear()
        lit_mark = None

    while True:
        ch = sub._peek()
        if ch == "":
            break
        if ch == "$":
            flush()
            dollar = sub._lex_dollar("word")
            if isinstance(dollar, (Lit, Param, CmdSub, Arith)):
                if isinstance(dollar, Arith):
                    parts.extend(dollar.parts)
                else:
                    parts.append(dollar)
            continue
        if ch == "`":
            flush()
            parts.append(sub._lex_backtick())
            continue
        if ch in _NAME_START:
            flush()
            mark = sub._mark()
            start = sub.pos
            while sub._peek() in _NAME_CHARS and sub._peek() != "":
                sub._adv()
            name = sub.text[start : sub.pos]
            parts.append(ShId(name, IdKind.VARIABLE_REF, sub._loc(mark)))
            continue
        if lit_mark is None:
            lit_mark = sub._mark()
        lit_chars.append(ch)
        sub._adv()
    flush()
    return parts
