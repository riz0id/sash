"""Precedence-climbing parser for shell arithmetic ($((...)) content).

One grammar serves both dialects (POSIX arithmetic is a subset of bash's).
Embedded `$var`/`${...}`/`$(...)`/backtick forms are lexed with the ordinary
lexer machinery, so command-substitution sites inside arithmetic stay
statically locatable staged re-reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from .lexer import _NAME_CHARS, _NAME_START, Lexer
from .nodes import (
    Arith,
    ArithAssign,
    ArithBinary,
    ArithExpr,
    ArithGroup,
    ArithNum,
    ArithPart,
    ArithTernary,
    ArithUnary,
    ArithVar,
    CmdSub,
    Dialect,
    IdKind,
    Lit,
    Loc,
    Param,
    ShId,
    ShParseError,
)

_OPS = [
    "<<=",
    ">>=",
    "**",
    "++",
    "--",
    "<<",
    ">>",
    "<=",
    ">=",
    "==",
    "!=",
    "&&",
    "||",
    "*=",
    "/=",
    "%=",
    "+=",
    "-=",
    "&=",
    "^=",
    "|=",
    "?",
    ":",
    ",",
    "=",
    "!",
    "~",
    "+",
    "-",
    "*",
    "/",
    "%",
    "<",
    ">",
    "&",
    "^",
    "|",
    "(",
    ")",
    "[",
    "]",
]

_ASSIGN_OPS = frozenset(
    ["=", "*=", "/=", "%=", "+=", "-=", "<<=", ">>=", "&=", "^=", "|="]
)

_BINARY_LEVELS: list[tuple[str, ...]] = [
    ("||",),
    ("&&",),
    ("|",),
    ("^",),
    ("&",),
    ("==", "!="),
    ("<=", ">=", "<", ">"),
    ("<<", ">>"),
    ("+", "-"),
    ("*", "/", "%"),
]

_DIGITS = "0123456789"
_HEX_DIGITS = "0123456789abcdefABCDEF"


@dataclass(eq=False)
class _Tok:
    kind: str  # "num" | "name" | "op" | "part" | "eof"
    text: str
    loc: Loc
    part: Param | CmdSub | Arith | None = None


def parse_arith(text: str, base: Loc, *, dialect: Dialect = Dialect.POSIX) -> ArithExpr:
    return _ArithParser(text, base, dialect).parse()


def iter_arith(expr: ArithExpr) -> Iterator[ArithExpr]:
    yield expr
    if isinstance(expr, ArithVar):
        if expr.subscript is not None:
            yield from iter_arith(expr.subscript)
    elif isinstance(expr, ArithUnary):
        yield from iter_arith(expr.operand)
    elif isinstance(expr, ArithBinary):
        yield from iter_arith(expr.left)
        yield from iter_arith(expr.right)
    elif isinstance(expr, ArithAssign):
        yield from iter_arith(expr.target)
        yield from iter_arith(expr.value)
    elif isinstance(expr, ArithTernary):
        yield from iter_arith(expr.cond)
        yield from iter_arith(expr.then)
        yield from iter_arith(expr.otherwise)
    elif isinstance(expr, ArithGroup):
        yield from iter_arith(expr.inner)


class _ArithParser:
    def __init__(self, text: str, base: Loc, dialect: Dialect) -> None:
        self.lx = Lexer(
            text,
            base.src,
            line=base.line,
            col=base.col,
            synthetic=base.synthetic,
            dialect=dialect,
        )
        self.tok = self._next_tok()

    # -- tokens --------------------------------------------------------------

    def _next_tok(self) -> _Tok:
        lx = self.lx
        while True:
            ch = lx._peek()
            if ch in " \t\n" and ch != "":
                lx._adv()
                continue
            if ch == "\\" and lx._peek(1) == "\n":
                lx._adv(2)
                continue
            break
        mark = lx._mark()
        ch = lx._peek()
        if ch == "":
            return _Tok("eof", "", lx._loc(mark))
        if ch == "$":
            part = lx._lex_dollar("word")
            if not isinstance(part, (Param, CmdSub, Arith)):
                raise ShParseError(
                    "unexpected '$' in arithmetic expression", lx._loc(mark)
                )
            return _Tok("part", "$", lx._loc(mark), part)
        if ch == "`":
            return _Tok("part", "`", lx._loc(mark), lx._lex_backtick())
        if ch in _DIGITS:
            return self._lex_number(mark)
        if ch in _NAME_START:
            while lx._peek() in _NAME_CHARS and lx._peek() != "":
                lx._adv()
            return _Tok("name", lx.text[mark[0] : lx.pos], lx._loc(mark))
        for op in _OPS:
            if lx.text.startswith(op, lx.pos):
                lx._adv(len(op))
                return _Tok("op", op, lx._loc(mark))
        raise ShParseError(
            f"unexpected character {ch!r} in arithmetic expression", lx._loc(mark)
        )

    def _lex_number(self, mark: tuple[int, int, int]) -> _Tok:
        lx = self.lx
        if lx._peek() == "0" and lx._peek(1) != "" and lx._peek(1) in "xX":
            lx._adv(2)
            start = lx.pos
            while lx._peek() in _HEX_DIGITS and lx._peek() != "":
                lx._adv()
            if lx.pos == start:
                raise ShParseError("malformed hexadecimal constant", lx._loc(mark))
        else:
            while lx._peek() in _DIGITS and lx._peek() != "":
                lx._adv()
            if lx._peek() == "#":
                base = int(lx.text[mark[0] : lx.pos])
                if not 2 <= base <= 64:
                    raise ShParseError(f"invalid arithmetic base {base}", lx._loc(mark))
                lx._adv()
                start = lx.pos
                while (c := lx._peek()) != "" and (c.isalnum() or c in "@_"):
                    lx._adv()
                if lx.pos == start:
                    raise ShParseError(
                        "missing digits after arithmetic base", lx._loc(mark)
                    )
        return _Tok("num", lx.text[mark[0] : lx.pos], lx._loc(mark))

    def _advance(self) -> None:
        self.tok = self._next_tok()

    def _at_op(self, *syms: str) -> bool:
        return self.tok.kind == "op" and self.tok.text in syms

    def _expect(self, sym: str) -> None:
        if not self._at_op(sym):
            raise self._error(f"expected {sym!r}")
        self._advance()

    def _error(self, message: str) -> ShParseError:
        return ShParseError(f"{message} in arithmetic expression", self.tok.loc)

    # -- grammar -------------------------------------------------------------

    def parse(self) -> ArithExpr:
        if self.tok.kind == "eof":
            loc = self.tok.loc
            return ArithNum("0", Loc(loc.src, loc.line, loc.col, loc.pos, 0, True))
        expr = self._comma()
        if self.tok.kind != "eof":
            raise self._error(f"unexpected {self.tok.text!r}")
        return expr

    def _comma(self) -> ArithExpr:
        left = self._assignment()
        while self._at_op(","):
            self._advance()
            left = ArithBinary(",", left, self._assignment(), left.loc)
        return left

    def _assignment(self) -> ArithExpr:
        left = self._ternary()
        if self.tok.kind == "op" and self.tok.text in _ASSIGN_OPS:
            op = self.tok
            if not isinstance(left, ArithVar):
                raise ShParseError(
                    "attempted assignment to non-variable in arithmetic expression",
                    op.loc,
                )
            self._advance()
            left.id.kind = IdKind.VARIABLE_BINDER
            return ArithAssign(left, op.text, self._assignment(), left.loc)
        return left

    def _ternary(self) -> ArithExpr:
        cond = self._binary(0)
        if self._at_op("?"):
            self._advance()
            then = self._comma()
            self._expect(":")
            return ArithTernary(cond, then, self._ternary(), cond.loc)
        return cond

    def _binary(self, level: int) -> ArithExpr:
        if level == len(_BINARY_LEVELS):
            return self._exponent()
        left = self._binary(level + 1)
        while self._at_op(*_BINARY_LEVELS[level]):
            op = self.tok
            self._advance()
            left = ArithBinary(op.text, left, self._binary(level + 1), left.loc)
        return left

    def _exponent(self) -> ArithExpr:
        left = self._unary()
        if self._at_op("**"):
            self._advance()
            return ArithBinary("**", left, self._exponent(), left.loc)
        return left

    def _unary(self) -> ArithExpr:
        if self._at_op("!", "~", "+", "-"):
            op = self.tok
            self._advance()
            return ArithUnary(op.text, self._unary(), op.loc)
        if self._at_op("++", "--"):
            op = self.tok
            self._advance()
            operand = self._unary()
            if isinstance(operand, ArithVar):
                operand.id.kind = IdKind.VARIABLE_BINDER
                return ArithUnary(op.text, operand, op.loc)
            # bash re-reads `++`/`--` before a non-variable as two unary signs
            ch = op.text[0]
            return ArithUnary(ch, ArithUnary(ch, operand, op.loc), op.loc)
        return self._postfix()

    def _postfix(self) -> ArithExpr:
        expr = self._primary()
        if isinstance(expr, ArithVar) and self._at_op("++", "--"):
            op = self.tok
            self._advance()
            expr.id.kind = IdKind.VARIABLE_BINDER
            return ArithUnary(op.text, expr, expr.loc, postfix=True)
        return expr

    def _primary(self) -> ArithExpr:
        tok = self.tok
        if tok.kind == "num":
            self._advance()
            return ArithNum(tok.text, tok.loc)
        if tok.kind == "name":
            self._advance()
            subscript: ArithExpr | None = None
            if self._at_op("[") and self.tok.loc.pos == tok.loc.pos + tok.loc.span:
                self._advance()
                subscript = self._comma()
                self._expect("]")
            return ArithVar(
                ShId(tok.text, IdKind.VARIABLE_REF, tok.loc), subscript, tok.loc
            )
        if tok.kind == "part":
            self._advance()
            assert tok.part is not None
            return ArithPart(tok.part, tok.loc)
        if self._at_op("("):
            self._advance()
            inner = self._comma()
            self._expect(")")
            return ArithGroup(inner, tok.loc)
        if tok.kind == "eof":
            raise self._error("unexpected end of input")
        raise self._error(f"unexpected {tok.text!r}")
