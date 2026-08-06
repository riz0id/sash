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

import re
from typing import Literal, cast

from .nodes import (
    AnsiCQuote,
    Arith,
    ArithExpr,
    BraceExpand,
    CmdSub,
    CmdSubStyle,
    Dialect,
    DQuote,
    DQuotePart,
    Escape,
    HereDoc,
    IdKind,
    Lit,
    Loc,
    Param,
    ProcSub,
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

BASH_OPERATORS = [
    "<<<",
    "<<-",
    "&>>",
    "<<",
    "<&",
    "<>",
    ">>",
    ">&",
    ">|",
    "&>",
    "&&",
    "||",
    ";;&",
    ";;",
    ";&",
    "<",
    ">",
    "&",
    "|",
    ";",
    "(",
    ")",
]

WORD_DELIMS = " \t\n|&;<>()"
# Inside [[ ]], a regex word keeps unquoted parens (and their contents);
# only unparenthesized metacharacters and blanks terminate it.
COND_REGEX_STOP = " \t\n|&;<>"
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
        dialect: Dialect = Dialect.POSIX,
    ) -> None:
        self.text = text
        self.src = src
        self.pos = 0
        self.line = line
        self.col = col
        self.synthetic = synthetic
        self.dialect = dialect
        self._await_delim: ShId | None = None
        self._pending_heredocs: list[tuple[HereDoc, str]] = []
        self._heredoc_by_word: dict[int, HereDoc] = {}
        # bash lexes `a[foo bar]=x` as one word only where an assignment is
        # grammatically acceptable: at command position or after one
        self._assign_ok = True

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
        if isinstance(tok, (TokOp, TokNewline)):
            self._assign_ok = True
        elif isinstance(tok, TokWord):
            self._assign_ok = _is_assignment_shaped(tok.word)
        else:
            self._assign_ok = False
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
        if self.dialect is Dialect.BASH and ch in "<>" and self._peek(1) == "(":
            word = self._lex_procsub_word()
            return TokWord(word, word.loc)
        ops = BASH_OPERATORS if self.dialect is Dialect.BASH else OPERATORS
        for op in ops:
            if self.text.startswith(op, self.pos):
                self._adv(len(op))
                loc = self._loc(mark)
                return TokOp(ShId(op, IdKind.OPERATOR, loc), loc)
        if self.dialect is Dialect.BASH and self._assign_ok and ch in _NAME_START:
            assign_word = self._try_lex_assignment_word()
            if assign_word is not None:
                return TokWord(assign_word, assign_word.loc)
        word = self._lex_word()
        return TokWord(word, word.loc)

    # -- words ---------------------------------------------------------------

    def _lex_word(self) -> Word:
        mark = self._mark()
        parts = self._lex_parts(stop=WORD_DELIMS, mode="word", brace=True)
        if not parts:
            raise self._error(f"unexpected character {self._peek()!r}")
        return Word(parts, self._loc(mark))

    # -- bash [[ ]] conditional tokens ---------------------------------------
    #
    # Inside [[ ]] words lex with the ordinary quoting/expansion rules, but
    # '<' '>' '&&' '||' '(' ')' are operators (no redirects here), newlines
    # are blanks, and ']]' terminates (recognized as a word by the parser).

    def next_cond_token(self) -> Token:
        return self._scan_cond_token(regex=False)

    def next_cond_regex_token(self) -> Token:
        return self._scan_cond_token(regex=True)

    def _scan_cond_token(self, *, regex: bool) -> Token:
        while True:
            self._skip_blanks()
            ch = self._peek()
            if ch == "\n":
                self._adv()
                self._collect_heredocs()
                continue
            if ch == "#":
                while self._peek() not in ("", "\n"):
                    self._adv()
                continue
            break
        mark = self._mark()
        ch = self._peek()
        if ch == "":
            return TokEof(self._loc(mark))
        if not regex:
            for op in ("&&", "||", "(", ")", "<", ">"):
                if self.text.startswith(op, self.pos):
                    self._adv(len(op))
                    loc = self._loc(mark)
                    return TokOp(ShId(op, IdKind.OPERATOR, loc), loc)
            parts = self._lex_parts(stop=WORD_DELIMS, mode="word")
        else:
            parts = self._lex_parts(stop=COND_REGEX_STOP, mode="word", cond_regex=True)
        if not parts:
            raise self._error(
                f"unexpected character {ch!r} in conditional expression", mark
            )
        word = Word(parts, self._loc(mark))
        return TokWord(word, word.loc)

    def _try_lex_assignment_word(self) -> Word | None:
        """A word with a `name[...]=` / `name[...]+=` prefix (bash arrays).

        The subscript runs to the matching `]` and may contain almost
        anything, including blanks, so ordinary word splitting must not
        apply inside it.
        """
        j = self.pos
        while j < len(self.text) and self.text[j] in _NAME_CHARS:
            j += 1
        if not self.text.startswith("[", j):
            return None
        depth = 0
        k = j
        while k < len(self.text):
            c = self.text[k]
            if c == "\n":
                return None
            if c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        else:
            return None
        end = k + 1
        if self.text.startswith("+=", end):
            end += 2
        elif self.text.startswith("=", end):
            end += 1
        else:
            return None
        mark = self._mark()
        self._adv(end - self.pos)
        parts: list[WordPart] = [Lit(self.text[mark[0] : self.pos], self._loc(mark))]
        parts.extend(self._lex_parts(stop=WORD_DELIMS, mode="word", brace=True))
        return Word(parts, self._loc(mark))

    def _lex_parts(
        self, *, stop: str, mode: Mode, brace: bool = False, cond_regex: bool = False
    ) -> list[WordPart]:
        parts: list[WordPart] = []
        lit_mark: _Mark | None = None
        lit_chars: list[str] = []
        # bash keeps unquoted parens (and everything inside them) as part of
        # a =~ regex word, so stop characters are inert while depth > 0
        depth = 0

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
            if ch == "":
                break
            if cond_regex and ch == "(":
                depth += 1
                lit(ch, 1)
                continue
            if cond_regex and ch == ")":
                if depth == 0:
                    break
                depth -= 1
                lit(ch, 1)
                continue
            if stop and ch in stop and mode != "heredoc" and depth == 0:
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
            if ch == "{" and brace and self.dialect is Dialect.BASH:
                expand = self._try_brace_expand()
                if expand is not None:
                    flush()
                    parts.append(expand)
                    continue
                lit("{", 1)
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

    # -- bash quoting and brace expansion ------------------------------------

    def _lex_ansi_c(self, mark: _Mark) -> AnsiCQuote:
        self._adv()  # '
        start = self.pos
        while True:
            ch = self._peek()
            if ch == "":
                raise self._error("unterminated $'...'", mark)
            if ch == "\\" and self._peek(1) != "":
                self._adv(2)
                continue
            if ch == "'":
                break
            self._adv()
        raw = self.text[start : self.pos]
        self._adv()  # '
        return AnsiCQuote(_decode_ansi_c(raw), raw, self._loc(mark))

    def _try_brace_expand(self) -> BraceExpand | None:
        entry = self._mark()
        self._adv()  # {
        m = _BRACE_RANGE_RE.match(self.text, self.pos)
        if m is not None and _range_endpoints_compatible(m.group(1), m.group(2)):
            self._adv(m.end() - self.pos)
            return BraceExpand(
                None, (m.group(1), m.group(2), m.group(3)), self._loc(entry)
            )
        alternates: list[list[WordPart]] = []
        saw_comma = False
        while True:
            parts = self._lex_parts(stop=",}" + WORD_DELIMS, mode="word", brace=True)
            ch = self._peek()
            if ch == ",":
                alternates.append(parts)
                saw_comma = True
                self._adv()
                continue
            if ch == "}" and saw_comma:
                alternates.append(parts)
                self._adv()
                return BraceExpand(alternates, None, self._loc(entry))
            self._restore(entry)
            return None

    # -- $... ----------------------------------------------------------------

    def _lex_dollar(self, mode: Mode) -> WordPart:
        mark = self._mark()
        self._adv()  # $
        ch = self._peek()
        if self.dialect is Dialect.BASH and mode == "word":
            if ch == "'":
                return self._lex_ansi_c(mark)
            if ch == '"':
                dq = self._lex_dquote()
                dq.locale = True
                dq.loc = self._loc(mark)
                return dq
            if ch == "[":
                return self._lex_deprecated_arith(mark)
        if ch == "(":
            if self._peek(1) == "(":
                arith = self._try_arith(mark)
                if arith is not None:
                    return arith
            return self._lex_dollar_cmdsub(mark)
        if ch == "{":
            return self._lex_braced_param(mark, mode)
        if ch != "" and ch in _NAME_START:
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
        if ch != "" and ch in SPECIAL_PARAM_CHARS:
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
            if named is None:
                raise self._error("bad substitution", mark)
            name, kind = named
            subscript, subscript_raw = self._maybe_param_subscript(kind, mark)
            if self._peek() != "}":
                raise self._error("bad substitution", mark)
            self._adv()  # }
            loc = self._loc(mark)
            return Param(
                ShId(name, kind, self._loc(name_mark)),
                None,
                None,
                True,
                loc,
                subscript=subscript,
                subscript_raw=subscript_raw,
            )
        indirect = False
        if (
            self.dialect is Dialect.BASH
            and self._peek() == "!"
            and self._peek(1) != ""
            and self._peek(1) in _NAME_START
        ):
            self._adv()  # !
            indirect = True
        name_mark = self._mark()
        named = self._read_param_name()
        if named is None:
            raise self._error("bad substitution", mark)
        name, kind = named
        name_id = ShId(name, kind, self._loc(name_mark))
        if indirect and self._peek() in ("*", "@") and self._peek(1) == "}":
            listing = self._peek()
            self._adv(2)  # * or @, then }
            return Param(name_id, "!" + listing, None, False, self._loc(mark))
        subscript, subscript_raw = self._maybe_param_subscript(kind, mark)
        if self._peek() == "}":
            self._adv()
            return Param(
                name_id,
                None,
                None,
                False,
                self._loc(mark),
                indirect=indirect,
                subscript=subscript,
                subscript_raw=subscript_raw,
            )
        if self.dialect is Dialect.BASH:
            ext = self._lex_bash_param_ext(
                mark, mode, name_id, indirect, subscript, subscript_raw
            )
            if ext is not None:
                return ext
        op = self._read_param_op()
        if op is None:
            raise self._error("bad substitution", mark)
        word = self._lex_parts(stop="}", mode=mode if mode == "dquote" else "word")
        if self._peek() != "}":
            raise self._error("unterminated ${...}", mark)
        self._adv()  # }
        return Param(
            name_id,
            op,
            word,
            False,
            self._loc(mark),
            indirect=indirect,
            subscript=subscript,
            subscript_raw=subscript_raw,
        )

    def _maybe_param_subscript(
        self, kind: IdKind, mark: _Mark
    ) -> tuple[ArithExpr | Lit | None, str | None]:
        if (
            self.dialect is not Dialect.BASH
            or kind is not IdKind.VARIABLE_REF
            or self._peek() != "["
        ):
            return None, None
        self._adv()  # [
        sub_mark = self._mark()
        if self._peek() in ("@", "*") and self._peek(1) == "]":
            ch = self._peek()
            self._adv()
            lit = Lit(ch, self._loc(sub_mark))
            self._adv()  # ]
            return lit, ch
        depth = 0
        while True:
            ch = self._peek()
            if ch == "":
                raise self._error("unterminated ${...}", mark)
            if ch == "[":
                depth += 1
            elif ch == "]":
                if depth == 0:
                    break
                depth -= 1
            self._adv()
        raw = self.text[sub_mark[0] : self.pos]
        loc = self._loc(sub_mark)
        self._adv()  # ]
        from .arith import parse_arith

        try:
            return parse_arith(raw, loc, dialect=self.dialect), raw
        except ShParseError:
            # not arithmetic; keep the raw text (associative-array key)
            return None, raw

    def _lex_bash_param_ext(
        self,
        mark: _Mark,
        mode: Mode,
        name_id: ShId,
        indirect: bool,
        subscript: ArithExpr | Lit | None,
        subscript_raw: str | None,
    ) -> Param | None:
        word_mode: Mode = mode if mode == "dquote" else "word"
        ch = self._peek()
        if ch == "/":
            if self._peek(1) == "/":
                op = "//"
                self._adv(2)
            elif self._peek(1) in ("#", "%"):
                op = "/" + self._peek(1)
                self._adv(2)
            else:
                op = "/"
                self._adv()
            pattern = self._lex_parts(stop="/}", mode=word_mode)
            word: list[WordPart] | None = None
            if self._peek() == "/":
                self._adv()
                word = self._lex_parts(stop="}", mode=word_mode)
            if self._peek() != "}":
                raise self._error("unterminated ${...}", mark)
            self._adv()  # }
            return Param(
                name_id,
                op,
                word,
                False,
                self._loc(mark),
                pattern=pattern,
                indirect=indirect,
                subscript=subscript,
                subscript_raw=subscript_raw,
            )
        if ch == ":" and not (self._peek(1) != "" and self._peek(1) in "-=?+"):
            self._adv()
            offset = self._scan_arith_until(":}", mark)
            length: ArithExpr | None = None
            if self._peek() == ":":
                self._adv()
                length = self._scan_arith_until("}", mark)
            if self._peek() != "}":
                raise self._error("unterminated ${...}", mark)
            self._adv()  # }
            return Param(
                name_id,
                ":",
                None,
                False,
                self._loc(mark),
                offset=offset,
                length=length,
                indirect=indirect,
                subscript=subscript,
                subscript_raw=subscript_raw,
            )
        if ch in ("^", ","):
            op = ch * 2 if self._peek(1) == ch else ch
            self._adv(len(op))
            word = self._lex_parts(stop="}", mode=word_mode)
            if self._peek() != "}":
                raise self._error("unterminated ${...}", mark)
            self._adv()  # }
            return Param(
                name_id,
                op,
                word,
                False,
                self._loc(mark),
                indirect=indirect,
                subscript=subscript,
                subscript_raw=subscript_raw,
            )
        if ch == "@":
            transform = self._peek(1)
            if transform == "" or transform not in "QEPAKauLU":
                raise self._error("bad substitution", mark)
            self._adv(2)
            if self._peek() != "}":
                raise self._error("bad substitution", mark)
            self._adv()  # }
            return Param(
                name_id,
                "@" + transform,
                None,
                False,
                self._loc(mark),
                indirect=indirect,
                subscript=subscript,
                subscript_raw=subscript_raw,
            )
        return None

    def _scan_arith_until(self, stops: str, mark: _Mark) -> ArithExpr:
        from .arith import parse_arith

        start = self._mark()
        depth = 0
        while True:
            ch = self._peek()
            if ch == "":
                raise self._error("unterminated ${...}", mark)
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            elif depth == 0 and ch in stops:
                break
            self._adv()
        text = self.text[start[0] : self.pos]
        return parse_arith(text, self._loc(start), dialect=self.dialect)

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
                    from .arith import parse_arith

                    content = self.text[content_mark[0] : self.pos]
                    expr = parse_arith(
                        content, self._loc(content_mark), dialect=self.dialect
                    )
                    self._adv(2)  # ))
                    return Arith(expr, self._loc(mark))
                self._restore(entry)
                return None
            self._adv()

    def _lex_deprecated_arith(self, mark: _Mark) -> Arith:
        from .arith import parse_arith

        self._adv()  # [
        content_mark = self._mark()
        depth = 0
        while True:
            ch = self._peek()
            if ch == "":
                raise self._error("unterminated $[...]", mark)
            if ch == "[":
                depth += 1
            elif ch == "]":
                if depth == 0:
                    break
                depth -= 1
            self._adv()
        content = self.text[content_mark[0] : self.pos]
        expr = parse_arith(content, self._loc(content_mark), dialect=self.dialect)
        self._adv()  # ]
        return Arith(expr, self._loc(mark), deprecated=True)

    # -- $(...) and `...` ----------------------------------------------------

    def _lex_dollar_cmdsub(self, mark: _Mark) -> CmdSub:
        self._adv()  # (
        content_mark = self._mark()
        self._scan_dollar_paren(mark)
        raw = self.text[content_mark[0] : self.pos - 1]
        pos, line, col = content_mark
        raw_loc = Loc(self.src, line, col, pos, len(raw), self.synthetic)
        return CmdSub(raw, CmdSubStyle.DOLLAR, self._loc(mark), raw_loc)

    def _lex_procsub_word(self) -> Word:
        """A word beginning with bash process substitution `<(...)` / `>(...)`."""
        mark = self._mark()
        direction = self._peek()
        self._adv(2)  # < or >, then (
        content_mark = self._mark()
        self._scan_dollar_paren(mark, what="process substitution")
        raw = self.text[content_mark[0] : self.pos - 1]
        parts: list[WordPart] = [ProcSub(direction, raw, self._loc(mark))]
        parts.extend(self._lex_parts(stop=WORD_DELIMS, mode="word", brace=True))
        return Word(parts, self._loc(mark))

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

    def _scan_dollar_paren(
        self, outer_mark: _Mark, what: str = "command substitution"
    ) -> None:
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
                            f"unterminated here-document in {what}",
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
                raise self._error(f"unterminated {what}", outer_mark)
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
            if (
                self.dialect is Dialect.BASH
                and ch == "<"
                and self._peek(1) == "<"
                and self._peek(2) == "<"
            ):
                # here-string, not a here-document: no body to skip
                flush()
                self._adv(3)
                cmd_pos = False
                continue
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
                sub = Lexer(
                    body_text,
                    self.src,
                    line=line,
                    synthetic=True,
                    dialect=self.dialect,
                )
                hd.body = sub._lex_parts(stop="", mode="heredoc")


_BRACE_RANGE_RE = re.compile(
    r"([+-]?\d+|[A-Za-z])\.\.([+-]?\d+|[A-Za-z])(?:\.\.([+-]?\d+))?\}"
)

_ASSIGN_SHAPE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\[.*\])?\+?=")


def _is_assignment_shaped(word: Word) -> bool:
    if not word.parts or not isinstance(word.parts[0], Lit):
        return False
    return _ASSIGN_SHAPE_RE.match(word.parts[0].text) is not None


_HEX_DIGITS = "0123456789abcdefABCDEF"

_ANSI_SIMPLE = {
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "E": "\x1b",
    "f": "\f",
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "v": "\v",
    "\\": "\\",
    "'": "'",
    '"': '"',
    "?": "?",
}


def _range_endpoints_compatible(start: str, end: str) -> bool:
    return start.lstrip("+-").isdigit() == end.lstrip("+-").isdigit()


def _decode_ansi_c(raw: str) -> str:
    out: list[str] = []
    i = 0
    n = len(raw)
    while i < n:
        ch = raw[i]
        if ch != "\\" or i + 1 == n:
            out.append(ch)
            i += 1
            continue
        c = raw[i + 1]
        if c in _ANSI_SIMPLE:
            out.append(_ANSI_SIMPLE[c])
            i += 2
            continue
        if c in "01234567":
            j = i + 1
            while j < n and j < i + 4 and raw[j] in "01234567":
                j += 1
            out.append(chr(int(raw[i + 1 : j], 8) & 0xFF))
            i = j
            continue
        if c == "x":
            j = i + 2
            while j < n and j < i + 4 and raw[j] in _HEX_DIGITS:
                j += 1
            if j == i + 2:
                out.append("\\x")
            else:
                out.append(chr(int(raw[i + 2 : j], 16)))
            i = max(j, i + 2)
            continue
        if c in "uU":
            width = 4 if c == "u" else 8
            j = i + 2
            while j < n and j < i + 2 + width and raw[j] in _HEX_DIGITS:
                j += 1
            if j == i + 2:
                out.append("\\" + c)
            else:
                out.append(chr(int(raw[i + 2 : j], 16)))
            i = max(j, i + 2)
            continue
        if c == "c":
            if i + 2 >= n:
                out.append("\\c")
                i += 2
                continue
            target = raw[i + 2]
            consumed = 3
            if target == "\\" and i + 3 < n and raw[i + 3] == "\\":
                consumed = 4  # backslash after \c must itself be escaped
            out.append(chr(ord(target.upper()) & 0x1F))
            i += consumed
            continue
        out.append("\\" + c)
        i += 2
    return "".join(out)


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
