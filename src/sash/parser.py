"""Recursive-descent parser for the POSIX shell grammar (XCU 2.10).

One-token lookahead plus word inspection (two tokens only for the
`name ( )` function-definition form). Reserved words are promoted
positionally: a word counts as a reserved word only when it is a single
unquoted literal appearing where the grammar recognizes it, so `"if"` and
`\\if` are ordinary words.

Command substitutions captured by the lexer as raw text are recursively
parsed here (a staged re-read run eagerly, since the site and its contents
are statically known); here-document bodies, collected by the lexer at
newlines, get the same treatment after the main parse.
"""

from __future__ import annotations

import re

from .arith import iter_arith, parse_arith
from .lexer import Lexer
from .nodes import (
    ArithCommand,
    ArithExpr,
    ArithPart,
    AndOr,
    Arith,
    ArrayItem,
    Assign,
    BASH_RESERVED_WORDS,
    BraceExpand,
    Async,
    BraceGroup,
    Case,
    CaseItem,
    CmdSub,
    Command,
    COND_BINARY_OPS,
    COND_UNARY_OPS,
    CondAnd,
    CondBinary,
    CondCmd,
    CondExpr,
    CondGroup,
    CondNot,
    CondOr,
    CondUnary,
    Coproc,
    Dialect,
    DQuote,
    For,
    ForArith,
    FunDef,
    HereDoc,
    IdKind,
    If,
    Lit,
    Loc,
    Param,
    Pipeline,
    ProcSub,
    Program,
    Redirect,
    Select,
    ShId,
    ShParseError,
    Simple,
    Subshell,
    TokEof,
    TokIoNumber,
    TokNewline,
    TokOp,
    TokWord,
    Token,
    Until,
    While,
    Word,
    WordPart,
    is_valid_name,
    matching_bracket,
    word_single_unquoted_lit,
    word_to_reserved,
)

REDIR_OPS = frozenset(["<", ">", ">>", "<<", "<<-", "<&", ">&", "<>", ">|"])

_NAME_PREFIX_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def _subscript_arith(raw: str, loc: Loc, dialect: Dialect) -> ArithExpr | None:
    """Arithmetic view of a subscript, or None for assoc-style raw keys."""
    if raw.strip(" \t\n") == "":
        return None
    try:
        return parse_arith(raw, loc, dialect=dialect)
    except ShParseError:
        return None


BASH_REDIR_OPS = REDIR_OPS | frozenset(["&>", "&>>", "<<<"])

_BAD_COMMAND_START = frozenset(
    ["then", "elif", "else", "fi", "do", "done", "esac", "in", "}"]
)


def parse(
    text: str,
    src: str = "<sh>",
    *,
    line: int = 1,
    col: int = 0,
    synthetic: bool = False,
    dialect: Dialect = Dialect.POSIX,
) -> Program:
    return _Parser(
        text, src, line=line, col=col, synthetic=synthetic, dialect=dialect
    ).parse_program()


class _Parser:
    def __init__(
        self,
        text: str,
        src: str,
        *,
        line: int = 1,
        col: int = 0,
        synthetic: bool = False,
        dialect: Dialect = Dialect.POSIX,
    ) -> None:
        self.src = src
        self.dialect = dialect
        self._redir_ops = BASH_REDIR_OPS if dialect is Dialect.BASH else REDIR_OPS
        self.lexer = Lexer(
            text, src, line=line, col=col, synthetic=synthetic, dialect=dialect
        )
        self.program_loc = Loc(src, line, col, 0, len(text), synthetic)
        self._peeked: Token | None = None
        self.tok: Token = self.lexer.next_token()
        self._heredocs: list[HereDoc] = []
        # current token of the [[ ]] mini-grammar; live only during
        # _parse_cond_cmd, which drives the lexer's cond regime directly
        self._cond_tok: Token | None = None

    # -- token plumbing ------------------------------------------------------

    def _advance(self) -> None:
        if self._peeked is not None:
            self.tok = self._peeked
            self._peeked = None
        else:
            self.tok = self.lexer.next_token()

    def _peek2(self) -> Token:
        if self._peeked is None:
            self._peeked = self.lexer.next_token()
        return self._peeked

    def _tok_loc(self) -> Loc:
        return self.tok.loc

    def _error(self, message: str) -> ShParseError:
        return ShParseError(message, self._tok_loc())

    def _at_op(self, *syms: str) -> bool:
        return isinstance(self.tok, TokOp) and self.tok.id.sym in syms

    def _reserved(self) -> str | None:
        if isinstance(self.tok, TokWord):
            name = word_to_reserved(self.tok.word)
            if name is not None:
                return name
            if self.dialect is Dialect.BASH:
                text = word_single_unquoted_lit(self.tok.word)
                if text is not None and text in BASH_RESERVED_WORDS:
                    return text
        return None

    def _skip_newlines(self) -> None:
        while isinstance(self.tok, TokNewline):
            self._advance()

    def _take_word(self, what: str) -> Word:
        if not isinstance(self.tok, TokWord):
            raise self._error(f"expected {what}")
        word = self.tok.word
        self._advance()
        self._fill_parts(word.parts)
        return word

    def _expect_op(self, sym: str) -> ShId:
        if not self._at_op(sym):
            raise self._error(f"expected {sym!r}")
        assert isinstance(self.tok, TokOp)
        op_id = self.tok.id
        self._advance()
        return op_id

    def _expect_reserved(self, name: str) -> ShId:
        if self._reserved() != name:
            raise self._error(f"expected {name!r}")
        assert isinstance(self.tok, TokWord)
        kw = ShId(name, IdKind.OPERATOR, self.tok.word.loc)
        self._advance()
        return kw

    # -- nested command substitutions ---------------------------------------

    def _fill_parts(self, parts: list[WordPart] | None) -> None:
        if parts is None:
            return
        for part in parts:
            if isinstance(part, CmdSub):
                if part.program is None:
                    base = part.raw_loc if part.raw_loc is not None else part.loc
                    part.program = parse(
                        part.raw,
                        self.src,
                        line=base.line,
                        col=base.col,
                        synthetic=base.synthetic,
                        dialect=self.dialect,
                    )
            elif isinstance(part, ProcSub):
                if part.program is None:
                    base = part.loc
                    part.program = parse(
                        part.raw,
                        self.src,
                        line=base.line,
                        col=base.col + 2,  # raw begins after '<(' / '>('
                        synthetic=base.synthetic,
                        dialect=self.dialect,
                    )
            elif isinstance(part, DQuote):
                self._fill_parts(list(part.parts))
            elif isinstance(part, Param):
                self._fill_parts(part.pattern)
                self._fill_parts(part.word)
                if part.offset is not None:
                    self._fill_arith(part.offset)
                if part.length is not None:
                    self._fill_arith(part.length)
                if part.subscript is not None and not isinstance(part.subscript, Lit):
                    self._fill_arith(part.subscript)
            elif isinstance(part, Arith):
                self._fill_arith(part.expr)
            elif isinstance(part, BraceExpand):
                if part.alternates is not None:
                    for alt in part.alternates:
                        self._fill_parts(alt)

    def _fill_arith(self, expr: ArithExpr) -> None:
        for node in iter_arith(expr):
            if isinstance(node, ArithPart):
                self._fill_parts([node.part])

    # -- program structure ---------------------------------------------------

    def parse_program(self) -> Program:
        commands: list[Command] = []
        self._skip_newlines()
        while not isinstance(self.tok, TokEof):
            commands.extend(self._parse_complete_command())
            self._skip_newlines()
        for hd in self._heredocs:
            self._fill_parts(hd.body)
        return Program(commands, self.program_loc)

    def _parse_complete_command(self) -> list[Command]:
        result: list[Command] = []
        while True:
            cmd = self._parse_and_or()
            if self._at_op("&"):
                assert isinstance(self.tok, TokOp)
                cmd = Async(cmd, self.tok.id, cmd.loc)
                self._advance()
            elif self._at_op(";"):
                self._advance()
            else:
                result.append(cmd)
                break
            result.append(cmd)
            if isinstance(self.tok, (TokNewline, TokEof)):
                break
        if isinstance(self.tok, TokNewline):
            self._advance()
        elif not isinstance(self.tok, TokEof):
            raise self._error(f"unexpected token after command")
        return result

    def _parse_compound_list(self, stop: frozenset[str]) -> list[Command]:
        commands: list[Command] = []
        self._skip_newlines()
        while True:
            if self._at_stop(stop):
                return commands
            if isinstance(self.tok, TokEof):
                raise self._error(
                    f"unexpected end of input (expected one of {sorted(stop)})"
                )
            cmd = self._parse_and_or()
            if self._at_op("&"):
                assert isinstance(self.tok, TokOp)
                cmd = Async(cmd, self.tok.id, cmd.loc)
                self._advance()
            elif self._at_op(";"):
                self._advance()
            commands.append(cmd)
            self._skip_newlines()

    def _at_stop(self, stop: frozenset[str]) -> bool:
        reserved = self._reserved()
        if reserved is not None and reserved in stop:
            return True
        return isinstance(self.tok, TokOp) and self.tok.id.sym in stop

    # -- pipelines and lists -------------------------------------------------

    def _parse_and_or(self) -> Command:
        left = self._parse_pipeline()
        while self._at_op("&&", "||"):
            assert isinstance(self.tok, TokOp)
            op_id = self.tok.id
            self._advance()
            self._skip_newlines()
            right = self._parse_pipeline()
            left = AndOr(left, op_id, right, left.loc)
        return left

    def _parse_pipeline(self) -> Command:
        time_id: ShId | None = None
        time_p = False
        if self._reserved() == "time":
            assert isinstance(self.tok, TokWord)
            time_id = ShId("time", IdKind.OPERATOR, self.tok.word.loc)
            self._advance()
            if (
                isinstance(self.tok, TokWord)
                and word_single_unquoted_lit(self.tok.word) == "-p"
            ):
                time_p = True
                self._advance()
        bang_id: ShId | None = None
        if self._reserved() == "!":
            assert isinstance(self.tok, TokWord)
            bang_id = ShId("!", IdKind.OPERATOR, self.tok.word.loc)
            self._advance()
        if time_id is not None and bang_id is None and self._time_ends_pipeline():
            # a bare `time` is valid bash: it times nothing
            return Pipeline(None, [], [], time_id.loc, time_id=time_id, time_p=time_p)
        first = self._parse_command()
        cmds = [first]
        pipe_ids: list[ShId] = []
        while self._at_op("|"):
            assert isinstance(self.tok, TokOp)
            pipe_ids.append(self.tok.id)
            self._advance()
            self._skip_newlines()
            cmds.append(self._parse_command())
        if bang_id is None and time_id is None and not pipe_ids:
            return first
        return Pipeline(
            bang_id, cmds, pipe_ids, first.loc, time_id=time_id, time_p=time_p
        )

    def _time_ends_pipeline(self) -> bool:
        if isinstance(self.tok, (TokNewline, TokEof)):
            return True
        if self._at_op(";", "&", "&&", "||", ")", ";;"):
            return True
        reserved = self._reserved()
        return reserved is not None and reserved in _BAD_COMMAND_START

    # -- commands ------------------------------------------------------------

    def _parse_command(self) -> Command:
        if self._at_op("("):
            if self.dialect is Dialect.BASH and self._at_double_paren():
                arith_cmd = self._try_parse_arith_command()
                if arith_cmd is not None:
                    return arith_cmd
            return self._parse_subshell()
        reserved = self._reserved()
        if reserved is not None:
            if reserved == "if":
                return self._parse_if()
            if reserved == "while":
                return self._parse_while_until(is_while=True)
            if reserved == "until":
                return self._parse_while_until(is_while=False)
            if reserved == "for":
                return self._parse_for()
            if reserved == "function":
                return self._parse_function()
            if reserved == "select":
                return self._parse_select()
            if reserved == "coproc":
                return self._parse_coproc()
            if reserved == "[[":
                return self._parse_cond_cmd()
            if reserved == "case":
                return self._parse_case()
            if reserved == "{":
                return self._parse_brace_group()
            if reserved in _BAD_COMMAND_START:
                raise self._error(f"unexpected reserved word {reserved!r}")
            # 'in', '!' handled elsewhere; fall through is impossible
        if isinstance(self.tok, TokWord):
            name = word_single_unquoted_lit(self.tok.word)
            if (
                name is not None
                and is_valid_name(name)
                and isinstance(self._peek2(), TokOp)
                and self._peek2().id.sym == "("  # type: ignore[union-attr]
            ):
                return self._parse_fundef(name)
        return self._parse_simple()

    def _parse_fundef(self, name: str) -> FunDef:
        assert isinstance(self.tok, TokWord)
        name_id = ShId(name, IdKind.FUNCTION_NAME, self.tok.word.loc)
        loc = self.tok.word.loc
        self._advance()
        lparen = self._expect_op("(")
        rparen = self._expect_op(")")
        self._skip_newlines()
        body = self._parse_command()
        if isinstance(body, Simple):
            raise ShParseError("function body must be a compound command", body.loc)
        return FunDef(name_id, body, [], [lparen, rparen], loc)

    def _parse_function(self) -> FunDef:
        kw = self._expect_reserved("function")
        keywords = [kw]
        if not isinstance(self.tok, TokWord):
            raise self._error("expected a function name after 'function'")
        name = word_single_unquoted_lit(self.tok.word)
        if name is None or not is_valid_name(name):
            raise self._error("expected a function name after 'function'")
        name_id = ShId(name, IdKind.FUNCTION_NAME, self.tok.word.loc)
        self._advance()
        if self._at_op("("):
            nxt = self._peek2()
            if isinstance(nxt, TokOp) and nxt.id.sym == ")":
                keywords.append(self._expect_op("("))
                keywords.append(self._expect_op(")"))
        self._skip_newlines()
        body = self._parse_command()
        if isinstance(body, Simple):
            raise ShParseError("function body must be a compound command", body.loc)
        return FunDef(name_id, body, [], keywords, kw.loc)

    def _parse_coproc(self) -> Coproc:
        kw = self._expect_reserved("coproc")
        name_id: ShId | None = None
        if isinstance(self.tok, TokWord):
            name = word_single_unquoted_lit(self.tok.word)
            if (
                name is not None
                and is_valid_name(name)
                and self._starts_compound(self._peek2())
            ):
                name_id = ShId(name, IdKind.VARIABLE_BINDER, self.tok.word.loc)
                self._advance()
        cmd = self._parse_command()
        return Coproc(name_id, cmd, [kw], kw.loc)

    # -- bash [[ ]] conditional expressions -----------------------------------

    def _parse_cond_cmd(self) -> CondCmd:
        assert isinstance(self.tok, TokWord)
        open_id = ShId("[[", IdKind.OPERATOR, self.tok.word.loc)
        # the lexer sits just past '[[': switch it to the cond regime
        assert self._peeked is None
        self._cond_advance()
        expr = self._parse_cond_or()
        close_id = self._expect_cond_close()
        self._cond_tok = None
        self.tok = self.lexer.next_token()
        redirects = self._parse_trailing_redirects()
        return CondCmd(expr, redirects, [open_id, close_id], open_id.loc)

    def _cond_advance(self) -> None:
        self._cond_tok = self.lexer.next_cond_token()

    def _cond_error(self, message: str) -> ShParseError:
        assert self._cond_tok is not None
        return ShParseError(message, self._cond_tok.loc)

    def _cond_text(self) -> str | None:
        if isinstance(self._cond_tok, TokWord):
            return word_single_unquoted_lit(self._cond_tok.word)
        return None

    def _cond_at_op(self, *syms: str) -> bool:
        return isinstance(self._cond_tok, TokOp) and self._cond_tok.id.sym in syms

    def _cond_take_word(self, what: str) -> Word:
        tok = self._cond_tok
        if not isinstance(tok, TokWord) or self._cond_text() == "]]":
            raise self._cond_error(f"expected {what}")
        self._fill_parts(tok.word.parts)
        self._cond_advance()
        return tok.word

    def _expect_cond_close(self) -> ShId:
        if self._cond_text() != "]]":
            raise self._cond_error("expected ']]'")
        assert isinstance(self._cond_tok, TokWord)
        return ShId("]]", IdKind.OPERATOR, self._cond_tok.word.loc)

    def _parse_cond_or(self) -> CondExpr:
        left = self._parse_cond_and()
        while self._cond_at_op("||"):
            assert isinstance(self._cond_tok, TokOp)
            op_id = self._cond_tok.id
            self._cond_advance()
            right = self._parse_cond_and()
            left = CondOr(left, right, op_id, left.loc)
        return left

    def _parse_cond_and(self) -> CondExpr:
        left = self._parse_cond_not()
        while self._cond_at_op("&&"):
            assert isinstance(self._cond_tok, TokOp)
            op_id = self._cond_tok.id
            self._cond_advance()
            right = self._parse_cond_not()
            left = CondAnd(left, right, op_id, left.loc)
        return left

    def _parse_cond_not(self) -> CondExpr:
        if self._cond_text() == "!":
            assert isinstance(self._cond_tok, TokWord)
            bang_id = ShId("!", IdKind.OPERATOR, self._cond_tok.word.loc)
            self._cond_advance()
            return CondNot(bang_id, self._parse_cond_not(), bang_id.loc)
        return self._parse_cond_primary()

    def _parse_cond_primary(self) -> CondExpr:
        tok = self._cond_tok
        if isinstance(tok, TokOp) and tok.id.sym == "(":
            lparen = tok.id
            self._cond_advance()
            inner = self._parse_cond_or()
            if not self._cond_at_op(")"):
                raise self._cond_error("expected ')' in conditional expression")
            assert isinstance(self._cond_tok, TokOp)
            rparen = self._cond_tok.id
            self._cond_advance()
            return CondGroup(inner, lparen, rparen, lparen.loc)
        if isinstance(tok, TokEof):
            raise self._cond_error("unexpected end of input (expected ']]')")
        text = self._cond_text()
        if text == "]]":
            raise self._cond_error("expected a conditional expression")
        if text is not None and text in COND_UNARY_OPS:
            assert isinstance(tok, TokWord)
            op_id = ShId(text, IdKind.OPERATOR, tok.word.loc)
            self._cond_advance()
            if isinstance(self._cond_tok, TokWord) and self._cond_text() != "]]":
                operand = self._cond_take_word(f"operand of {text!r}")
                return CondUnary(op_id, operand, op_id.loc)
            # trailing unary op with no operand: bash's bare-word test
            return tok.word
        left = self._cond_take_word("a conditional operand")
        tok = self._cond_tok
        if isinstance(tok, TokOp) and tok.id.sym in ("<", ">"):
            op_id = tok.id
            self._cond_advance()
            right = self._cond_take_word(f"operand of {op_id.sym!r}")
            return CondBinary(op_id, left, right, left.loc)
        btext = self._cond_text()
        if btext is not None and btext in COND_BINARY_OPS:
            assert isinstance(tok, TokWord)
            op_id = ShId(btext, IdKind.OPERATOR, tok.word.loc)
            if btext == "=~":
                # regex regime: unquoted parens stay part of the word
                self._cond_tok = self.lexer.next_cond_regex_token()
                right = self._cond_take_word("a regular expression")
            else:
                self._cond_advance()
                right = self._cond_take_word(f"operand of {btext!r}")
            return CondBinary(op_id, left, right, left.loc)
        return left

    def _starts_compound(self, tok: Token) -> bool:
        if isinstance(tok, TokOp):
            return tok.id.sym == "("
        if isinstance(tok, TokWord):
            text = word_single_unquoted_lit(tok.word)
            return text in ("{", "if", "while", "until", "for", "case", "select")
        return False

    # -- bash '((' forms ------------------------------------------------------

    def _at_double_paren(self) -> bool:
        if not self._at_op("("):
            return False
        nxt = self._peek2()
        return (
            isinstance(nxt, TokOp)
            and nxt.id.sym == "("
            and nxt.loc.pos == self.tok.loc.pos + 1
        )

    def _capture_double_paren(self) -> tuple[str, Loc, Loc] | None:
        """Raw text through the `))` matching an already-lexed adjacent `((`.

        Called with `tok`/`_peeked` holding the two `(` operators, so the
        lexer sits just past `((`. Mirrors `_try_arith`: on failure the lexer
        is restored and the tokens are left for the subshell parse. On
        success returns (raw, raw_loc, close_loc) and advances past `))`.
        """
        lx = self.lexer
        entry = lx._mark()
        depth = 0
        while True:
            ch = lx._peek()
            if ch == "":
                lx._restore(entry)
                return None
            if ch == "\\":
                lx._adv(2)
                continue
            if ch == "'":
                lx._skip_squote_raw()
                continue
            if ch == '"':
                lx._skip_dquote_raw()
                continue
            if ch == "`":
                lx._skip_backtick_raw()
                continue
            if ch == "(":
                depth += 1
                lx._adv()
                continue
            if ch == ")":
                if depth > 0:
                    depth -= 1
                    lx._adv()
                    continue
                if lx._peek(1) == ")":
                    raw = lx.text[entry[0] : lx.pos]
                    close_mark = lx._mark()
                    lx._adv(2)
                    close_loc = lx._loc(close_mark)
                    raw_loc = Loc(
                        lx.src, entry[1], entry[2], entry[0], len(raw), lx.synthetic
                    )
                    self._peeked = None
                    self.tok = lx.next_token()
                    return raw, raw_loc, close_loc
                lx._restore(entry)
                return None
            lx._adv()

    def _open_double_paren_id(self) -> ShId:
        loc = self.tok.loc
        wide = Loc(loc.src, loc.line, loc.col, loc.pos, 2, loc.synthetic)
        return ShId("((", IdKind.OPERATOR, wide)

    def _try_parse_arith_command(self) -> ArithCommand | None:
        open_id = self._open_double_paren_id()
        captured = self._capture_double_paren()
        if captured is None:
            return None
        raw, raw_loc, close_loc = captured
        expr = parse_arith(raw, raw_loc, dialect=self.dialect)
        keywords = [open_id, ShId("))", IdKind.OPERATOR, close_loc)]
        redirects = self._parse_trailing_redirects()
        return ArithCommand(expr, redirects, keywords, open_id.loc)

    def _parse_for_arith(self, kw_for: ShId) -> ForArith:
        open_id = self._open_double_paren_id()
        captured = self._capture_double_paren()
        if captured is None:
            raise self._error("expected '))' closing arithmetic 'for' header")
        raw, raw_loc, close_loc = captured
        keywords = [kw_for, open_id, ShId("))", IdKind.OPERATOR, close_loc)]
        sections = self._split_arith_sections(raw, raw_loc)
        if len(sections) != 3:
            raise ShParseError("expected two ';' in arithmetic 'for' header", raw_loc)
        exprs: list[ArithExpr | None] = []
        for text, loc in sections:
            if text.strip(" \t\n") == "":
                exprs.append(None)
            else:
                exprs.append(parse_arith(text, loc, dialect=self.dialect))
        if self._at_op(";"):
            self._advance()
        self._skip_newlines()
        keywords.append(self._expect_reserved("do"))
        body = self._parse_compound_list(frozenset({"done"}))
        keywords.append(self._expect_reserved("done"))
        redirects = self._parse_trailing_redirects()
        return ForArith(
            exprs[0], exprs[1], exprs[2], body, redirects, keywords, kw_for.loc
        )

    def _split_arith_sections(self, raw: str, base: Loc) -> list[tuple[str, Loc]]:
        sections: list[tuple[str, Loc]] = []
        depth = 0
        start = 0
        line, col = base.line, base.col
        sline, scol = line, col
        for i, ch in enumerate(raw):
            if ch == "(":
                depth += 1
            elif ch == ")" and depth > 0:
                depth -= 1
            split = ch == ";" and depth == 0
            if split:
                sections.append(
                    (
                        raw[start:i],
                        Loc(
                            base.src,
                            sline,
                            scol,
                            base.pos + start,
                            i - start,
                            base.synthetic,
                        ),
                    )
                )
                start = i + 1
            if ch == "\n":
                line += 1
                col = 0
            else:
                col += 1
            if split:
                sline, scol = line, col
        sections.append(
            (
                raw[start:],
                Loc(
                    base.src,
                    sline,
                    scol,
                    base.pos + start,
                    len(raw) - start,
                    base.synthetic,
                ),
            )
        )
        return sections

    def _parse_subshell(self) -> Subshell:
        lparen = self._expect_op("(")
        loc = lparen.loc
        body = self._parse_compound_list(frozenset({")"}))
        rparen = self._expect_op(")")
        redirects = self._parse_trailing_redirects()
        return Subshell(body, redirects, [lparen, rparen], loc)

    def _parse_brace_group(self) -> BraceGroup:
        lbrace = self._expect_reserved("{")
        body = self._parse_compound_list(frozenset({"}"}))
        rbrace = self._expect_reserved("}")
        redirects = self._parse_trailing_redirects()
        return BraceGroup(body, redirects, [lbrace, rbrace], lbrace.loc)

    def _parse_if(self) -> If:
        kw_if = self._expect_reserved("if")
        keywords = [kw_if]
        clauses: list[tuple[list[Command], list[Command]]] = []
        test = self._parse_compound_list(frozenset({"then"}))
        keywords.append(self._expect_reserved("then"))
        then_body = self._parse_compound_list(frozenset({"elif", "else", "fi"}))
        clauses.append((test, then_body))
        else_body: list[Command] | None = None
        while True:
            reserved = self._reserved()
            if reserved == "elif":
                keywords.append(self._expect_reserved("elif"))
                test = self._parse_compound_list(frozenset({"then"}))
                keywords.append(self._expect_reserved("then"))
                body = self._parse_compound_list(frozenset({"elif", "else", "fi"}))
                clauses.append((test, body))
                continue
            if reserved == "else":
                keywords.append(self._expect_reserved("else"))
                else_body = self._parse_compound_list(frozenset({"fi"}))
            keywords.append(self._expect_reserved("fi"))
            break
        redirects = self._parse_trailing_redirects()
        return If(clauses, else_body, redirects, keywords, kw_if.loc)

    def _parse_while_until(self, *, is_while: bool) -> Command:
        kw = self._expect_reserved("while" if is_while else "until")
        test = self._parse_compound_list(frozenset({"do"}))
        kw_do = self._expect_reserved("do")
        body = self._parse_compound_list(frozenset({"done"}))
        kw_done = self._expect_reserved("done")
        redirects = self._parse_trailing_redirects()
        keywords = [kw, kw_do, kw_done]
        if is_while:
            return While(test, body, redirects, keywords, kw.loc)
        return Until(test, body, redirects, keywords, kw.loc)

    def _parse_for(self) -> For | ForArith:
        kw_for = self._expect_reserved("for")
        if self.dialect is Dialect.BASH and self._at_double_paren():
            return self._parse_for_arith(kw_for)
        keywords = [kw_for]
        if not isinstance(self.tok, TokWord):
            raise self._error("expected a variable name after 'for'")
        name = word_single_unquoted_lit(self.tok.word)
        if name is None or not is_valid_name(name):
            raise self._error("expected a variable name after 'for'")
        var_id = ShId(name, IdKind.FOR_VARIABLE, self.tok.word.loc)
        self._advance()
        self._skip_newlines()
        in_words: list[Word] | None = None
        if self._reserved() == "in":
            keywords.append(self._expect_reserved("in"))
            in_words = []
            while isinstance(self.tok, TokWord):
                in_words.append(self._take_word("word"))
            if self._at_op(";"):
                self._advance()
            elif isinstance(self.tok, TokNewline):
                pass
            else:
                raise self._error("expected ';' or newline after 'for' word list")
        elif self._at_op(";"):
            self._advance()
        self._skip_newlines()
        keywords.append(self._expect_reserved("do"))
        body = self._parse_compound_list(frozenset({"done"}))
        keywords.append(self._expect_reserved("done"))
        redirects = self._parse_trailing_redirects()
        return For(var_id, in_words, body, redirects, keywords, kw_for.loc)

    def _parse_select(self) -> Select:
        kw_select = self._expect_reserved("select")
        keywords = [kw_select]
        if not isinstance(self.tok, TokWord):
            raise self._error("expected a variable name after 'select'")
        name = word_single_unquoted_lit(self.tok.word)
        if name is None or not is_valid_name(name):
            raise self._error("expected a variable name after 'select'")
        var_id = ShId(name, IdKind.FOR_VARIABLE, self.tok.word.loc)
        self._advance()
        self._skip_newlines()
        in_words: list[Word] | None = None
        if self._reserved() == "in":
            keywords.append(self._expect_reserved("in"))
            in_words = []
            while isinstance(self.tok, TokWord):
                in_words.append(self._take_word("word"))
            if self._at_op(";"):
                self._advance()
            elif isinstance(self.tok, TokNewline):
                pass
            else:
                raise self._error("expected ';' or newline after 'select' word list")
        elif self._at_op(";"):
            self._advance()
        self._skip_newlines()
        keywords.append(self._expect_reserved("do"))
        body = self._parse_compound_list(frozenset({"done"}))
        keywords.append(self._expect_reserved("done"))
        redirects = self._parse_trailing_redirects()
        return Select(var_id, in_words, body, redirects, keywords, kw_select.loc)

    def _parse_case(self) -> Case:
        kw_case = self._expect_reserved("case")
        keywords = [kw_case]
        subject = self._take_word("word after 'case'")
        self._skip_newlines()
        keywords.append(self._expect_reserved("in"))
        self._skip_newlines()
        items: list[CaseItem] = []
        while self._reserved() != "esac":
            if isinstance(self.tok, TokEof):
                raise self._error("unexpected end of input (expected 'esac')")
            items.append(self._parse_case_item())
            self._skip_newlines()
        keywords.append(self._expect_reserved("esac"))
        redirects = self._parse_trailing_redirects()
        return Case(subject, items, redirects, keywords, kw_case.loc)

    def _parse_case_item(self) -> CaseItem:
        keywords: list[ShId] = []
        loc = self._tok_loc()
        if self._at_op("("):
            keywords.append(self._expect_op("("))
        patterns = [self._take_word("case pattern")]
        while self._at_op("|"):
            self._advance()
            patterns.append(self._take_word("case pattern"))
        keywords.append(self._expect_op(")"))
        stop = frozenset({";;", "esac"})
        if self.dialect is Dialect.BASH:
            stop = frozenset({";;", ";&", ";;&", "esac"})
        body = self._parse_compound_list(stop)
        terminator = ""
        if self._at_op(";;", ";&", ";;&"):
            assert isinstance(self.tok, TokOp)
            terminator = self.tok.id.sym
            keywords.append(self.tok.id)
            self._advance()
            self._skip_newlines()
        return CaseItem(patterns, body, keywords, loc, terminator)

    # -- simple commands and redirections ------------------------------------

    def _parse_simple(self) -> Simple:
        loc = self._tok_loc()
        assigns: list[Assign] = []
        words: list[Word] = []
        redirects: list[Redirect] = []
        cmd_id: ShId | None = None
        prefix = True
        while True:
            tok = self.tok
            if isinstance(tok, TokIoNumber):
                n = tok.n
                self._advance()
                redirects.append(self._parse_redirect(n))
                continue
            if isinstance(tok, TokOp) and tok.id.sym in self._redir_ops:
                redirects.append(self._parse_redirect(None))
                continue
            fd_var = self._try_fd_var()
            if fd_var is not None:
                redirects.append(self._parse_redirect(None, fd_var=fd_var))
                continue
            if isinstance(tok, TokWord):
                if prefix:
                    assign = self._try_assignment(tok.word)
                    if assign is not None:
                        word_end = tok.word.loc.pos + tok.word.loc.span
                        self._advance()
                        if (
                            self.dialect is Dialect.BASH
                            and not assign.word.parts
                            and assign.subscript_raw is None
                            and self._at_op("(")
                            and self.tok.loc.pos == word_end
                        ):
                            assign.array_items = self._parse_array_items()
                        assigns.append(assign)
                        continue
                    prefix = False
                    text = word_single_unquoted_lit(tok.word)
                    if text is not None:
                        cmd_id = ShId(text, IdKind.COMMAND, tok.word.loc)
                words.append(self._take_word("word"))
                continue
            break
        if not assigns and not words and not redirects:
            raise self._error("expected a command")
        return Simple(assigns, words, cmd_id, redirects, loc)

    def _try_assignment(self, word: Word) -> Assign | None:
        if not word.parts or not isinstance(word.parts[0], Lit):
            return None
        text = word.parts[0].text
        m = _NAME_PREFIX_RE.match(text)
        if m is None:
            return None
        name = m.group(0)
        i = m.end()
        first_loc = word.parts[0].loc
        subscript: ArithExpr | None = None
        subscript_raw: str | None = None
        if self.dialect is Dialect.BASH and text.startswith("[", i):
            end = matching_bracket(text, i)
            if end is None:
                return None
            subscript_raw = text[i + 1 : end]
            sub_loc = Loc(
                first_loc.src,
                first_loc.line,
                first_loc.col + i + 1,
                first_loc.pos + i + 1,
                len(subscript_raw),
                first_loc.synthetic,
            )
            subscript = _subscript_arith(subscript_raw, sub_loc, self.dialect)
            if subscript is not None:
                self._fill_arith(subscript)
            i = end + 1
        append = False
        if self.dialect is Dialect.BASH and text.startswith("+=", i):
            append = True
            i += 2
        elif text.startswith("=", i):
            i += 1
        else:
            return None
        name_loc = Loc(
            first_loc.src,
            first_loc.line,
            first_loc.col,
            first_loc.pos,
            len(name),
            first_loc.synthetic,
        )
        value_parts: list[WordPart] = list(word.parts[1:])
        rest = text[i:]
        if rest:
            rest_loc = Loc(
                first_loc.src,
                first_loc.line,
                first_loc.col + i,
                first_loc.pos + i,
                len(rest),
                first_loc.synthetic,
            )
            value_parts.insert(0, Lit(rest, rest_loc))
        self._fill_parts(value_parts)
        name_id = ShId(name, IdKind.VARIABLE_BINDER, name_loc)
        return Assign(
            name_id,
            Word(value_parts, word.loc),
            word.loc,
            subscript=subscript,
            subscript_raw=subscript_raw,
            append=append,
        )

    def _parse_array_items(self) -> list[ArrayItem]:
        self._expect_op("(")
        items: list[ArrayItem] = []
        while True:
            self._skip_newlines()
            if self._at_op(")"):
                self._advance()
                return items
            if isinstance(self.tok, TokEof):
                raise self._error("unterminated array assignment")
            items.append(self._array_item(self._take_word("array element")))

    def _array_item(self, word: Word) -> ArrayItem:
        if not word.parts or not isinstance(word.parts[0], Lit):
            return ArrayItem(None, None, word, word.loc)
        text = word.parts[0].text
        if not text.startswith("["):
            return ArrayItem(None, None, word, word.loc)
        end = matching_bracket(text, 0)
        if end is None or not text.startswith("=", end + 1):
            return ArrayItem(None, None, word, word.loc)
        first_loc = word.parts[0].loc
        subscript_raw = text[1:end]
        sub_loc = Loc(
            first_loc.src,
            first_loc.line,
            first_loc.col + 1,
            first_loc.pos + 1,
            len(subscript_raw),
            first_loc.synthetic,
        )
        subscript = _subscript_arith(subscript_raw, sub_loc, self.dialect)
        if subscript is not None:
            self._fill_arith(subscript)
        value_parts: list[WordPart] = list(word.parts[1:])
        rest = text[end + 2 :]
        if rest:
            rest_loc = Loc(
                first_loc.src,
                first_loc.line,
                first_loc.col + end + 2,
                first_loc.pos + end + 2,
                len(rest),
                first_loc.synthetic,
            )
            value_parts.insert(0, Lit(rest, rest_loc))
        return ArrayItem(
            subscript, subscript_raw, Word(value_parts, word.loc), word.loc
        )

    def _try_fd_var(self) -> ShId | None:
        """A `{NAME}` word immediately before a redirect operator (bash)."""
        if self.dialect is not Dialect.BASH or not isinstance(self.tok, TokWord):
            return None
        word = self.tok.word
        text = word_single_unquoted_lit(word)
        if text is None or len(text) < 3:
            return None
        if not (text.startswith("{") and text.endswith("}")):
            return None
        name = text[1:-1]
        if not is_valid_name(name):
            return None
        nxt = self._peek2()
        if not (isinstance(nxt, TokOp) and nxt.id.sym in self._redir_ops):
            return None
        if nxt.loc.pos != word.loc.pos + word.loc.span:
            return None
        loc = word.loc
        name_loc = Loc(
            loc.src, loc.line, loc.col + 1, loc.pos + 1, len(name), loc.synthetic
        )
        self._advance()
        return ShId(name, IdKind.VARIABLE_BINDER, name_loc)

    def _parse_redirect(self, n: int | None, fd_var: ShId | None = None) -> Redirect:
        # Reachable with a non-operator token via `N<(...)` in bash, where the
        # io-number is followed by a process-substitution word.
        if not isinstance(self.tok, TokOp):
            raise self._error("expected a redirect operator")
        op_id = self.tok.id
        self._advance()
        word = self._take_word(f"target of {op_id.sym!r}")
        target: Word | HereDoc = word
        move = False
        if op_id.sym in ("<<", "<<-"):
            hd = self.lexer.heredoc_for(word)
            if hd is None:
                raise ShParseError("missing here-document", op_id.loc)
            self._heredocs.append(hd)
            target = hd
        elif self.dialect is Dialect.BASH and op_id.sym in ("<&", ">&"):
            text = word_single_unquoted_lit(word)
            if (
                text is not None
                and len(text) > 1
                and text.endswith("-")
                and text[:-1].isdigit()
            ):
                move = True
        return Redirect(
            n,
            op_id,
            target,
            op_id.loc,
            here_string=op_id.sym == "<<<",
            move=move,
            fd_var=fd_var,
        )

    def _parse_trailing_redirects(self) -> list[Redirect]:
        redirects: list[Redirect] = []
        while True:
            if isinstance(self.tok, TokIoNumber):
                n = self.tok.n
                self._advance()
                redirects.append(self._parse_redirect(n))
                continue
            if isinstance(self.tok, TokOp) and self.tok.id.sym in self._redir_ops:
                redirects.append(self._parse_redirect(None))
                continue
            fd_var = self._try_fd_var()
            if fd_var is not None:
                redirects.append(self._parse_redirect(None, fd_var=fd_var))
                continue
            return redirects
