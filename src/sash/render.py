"""S-expression-style rendering and identifier traversal.

`program_to_datum` produces nested tuples of strings for inspection;
`iter_ids`/`find_ids` yield every ShId leaf in a program, including those
inside nested command-substitution programs, here-document bodies, and
arithmetic expansions — so questions like "which `|` leaves resolve to the
pipe operator?" are one filter away.
"""

from __future__ import annotations

from typing import Iterator, Union

from .arith import iter_arith
from .nodes import (
    AndOr,
    AnsiCQuote,
    Arith,
    ArithAssign,
    ArithBinary,
    ArithCommand,
    ArithExpr,
    ArithGroup,
    ArithNum,
    ArithPart,
    ArithTernary,
    ArithUnary,
    ArithVar,
    ArrayItem,
    Assign,
    Async,
    BraceExpand,
    BraceGroup,
    Case,
    CmdSub,
    CmdSubStyle,
    Command,
    CondAnd,
    CondBinary,
    CondCmd,
    CondExpr,
    CondGroup,
    CondNot,
    CondOr,
    CondUnary,
    Coproc,
    DQuote,
    Escape,
    For,
    ForArith,
    FunDef,
    HereDoc,
    IdKind,
    If,
    Lit,
    Param,
    Pipeline,
    ProcSub,
    Program,
    Redirect,
    Select,
    ShId,
    Simple,
    SQuote,
    Subshell,
    Until,
    While,
    Word,
    WordPart,
)

Datum = Union[str, tuple["Datum", ...]]


# --------------------------------------------------------------------------
# identifier traversal


def iter_ids(program: Program) -> Iterator[ShId]:
    for cmd in program.commands:
        yield from _cmd_ids(cmd)


def find_ids(
    program: Program, kind: IdKind | None = None, sym: str | None = None
) -> list[ShId]:
    return [
        i
        for i in iter_ids(program)
        if (kind is None or i.kind is kind) and (sym is None or i.sym == sym)
    ]


def _cmd_ids(cmd: Command) -> Iterator[ShId]:
    if isinstance(cmd, Simple):
        if cmd.cmd_id is not None:
            yield cmd.cmd_id
        for assign in cmd.assigns:
            yield from _assign_ids(assign)
        for word in cmd.words:
            yield from _parts_ids(word.parts)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, Pipeline):
        if cmd.time_id is not None:
            yield cmd.time_id
        if cmd.bang_id is not None:
            yield cmd.bang_id
        for sub, pid in zip(cmd.cmds, [None, *cmd.pipe_ids]):
            if pid is not None:
                yield pid
            yield from _cmd_ids(sub)
    elif isinstance(cmd, AndOr):
        yield from _cmd_ids(cmd.left)
        yield cmd.op_id
        yield from _cmd_ids(cmd.right)
    elif isinstance(cmd, Async):
        yield from _cmd_ids(cmd.cmd)
        yield cmd.amp_id
    elif isinstance(cmd, (Subshell, BraceGroup)):
        yield from cmd.keywords
        for sub_cmd in cmd.body:
            yield from _cmd_ids(sub_cmd)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, If):
        yield from cmd.keywords
        for test, body in cmd.clauses:
            for sub_cmd in [*test, *body]:
                yield from _cmd_ids(sub_cmd)
        if cmd.else_body is not None:
            for sub_cmd in cmd.else_body:
                yield from _cmd_ids(sub_cmd)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, (While, Until)):
        yield from cmd.keywords
        for sub_cmd in [*cmd.test, *cmd.body]:
            yield from _cmd_ids(sub_cmd)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, (For, Select)):
        yield from cmd.keywords
        yield cmd.var_id
        for word in cmd.in_words or []:
            yield from _parts_ids(word.parts)
        for sub_cmd in cmd.body:
            yield from _cmd_ids(sub_cmd)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, ForArith):
        yield from cmd.keywords
        for expr in (cmd.init, cmd.cond, cmd.step):
            if expr is not None:
                yield from _arith_ids(expr)
        for sub_cmd in cmd.body:
            yield from _cmd_ids(sub_cmd)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, ArithCommand):
        yield from cmd.keywords
        yield from _arith_ids(cmd.expr)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, Coproc):
        yield from cmd.keywords
        if cmd.name_id is not None:
            yield cmd.name_id
        yield from _cmd_ids(cmd.cmd)
    elif isinstance(cmd, CondCmd):
        yield from cmd.keywords
        yield from _cond_ids(cmd.expr)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, Case):
        yield from cmd.keywords
        yield from _parts_ids(cmd.subject.parts)
        for item in cmd.items:
            yield from item.keywords
            for pattern in item.patterns:
                yield from _parts_ids(pattern.parts)
            for sub_cmd in item.body:
                yield from _cmd_ids(sub_cmd)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, FunDef):
        yield cmd.name_id
        yield from cmd.keywords
        yield from _cmd_ids(cmd.body)
        yield from _redirect_ids(cmd.redirects)


def _cond_ids(expr: CondExpr) -> Iterator[ShId]:
    if isinstance(expr, Word):
        yield from _parts_ids(expr.parts)
    elif isinstance(expr, CondUnary):
        yield expr.op_id
        yield from _parts_ids(expr.operand.parts)
    elif isinstance(expr, CondBinary):
        yield from _parts_ids(expr.left.parts)
        yield expr.op_id
        yield from _parts_ids(expr.right.parts)
    elif isinstance(expr, CondNot):
        yield expr.bang_id
        yield from _cond_ids(expr.operand)
    elif isinstance(expr, (CondAnd, CondOr)):
        yield from _cond_ids(expr.left)
        yield expr.op_id
        yield from _cond_ids(expr.right)
    else:
        yield expr.lparen_id
        yield from _cond_ids(expr.inner)
        yield expr.rparen_id


def _assign_ids(assign: Assign) -> Iterator[ShId]:
    yield assign.name_id
    if assign.subscript is not None:
        yield from _arith_ids(assign.subscript)
    if assign.array_items is not None:
        for item in assign.array_items:
            if item.subscript is not None:
                yield from _arith_ids(item.subscript)
            yield from _parts_ids(item.word.parts)
    yield from _parts_ids(assign.word.parts)


def _redirect_ids(redirects: list[Redirect]) -> Iterator[ShId]:
    for redirect in redirects:
        if redirect.fd_var is not None:
            yield redirect.fd_var
        yield redirect.op_id
        if isinstance(redirect.target, HereDoc):
            if redirect.target.body is not None:
                yield from _parts_ids(redirect.target.body)
        else:
            yield from _parts_ids(redirect.target.parts)


def _parts_ids(parts: list[WordPart]) -> Iterator[ShId]:
    for part in parts:
        if isinstance(part, Param):
            yield part.id
            if part.subscript is not None and not isinstance(part.subscript, Lit):
                yield from _arith_ids(part.subscript)
            if part.pattern is not None:
                yield from _parts_ids(part.pattern)
            if part.word is not None:
                yield from _parts_ids(part.word)
            if part.offset is not None:
                yield from _arith_ids(part.offset)
            if part.length is not None:
                yield from _arith_ids(part.length)
        elif isinstance(part, DQuote):
            yield from _parts_ids(list(part.parts))
        elif isinstance(part, CmdSub):
            if part.program is not None:
                yield from iter_ids(part.program)
        elif isinstance(part, ProcSub):
            if part.program is not None:
                yield from iter_ids(part.program)
        elif isinstance(part, Arith):
            yield from _arith_ids(part.expr)
        elif isinstance(part, BraceExpand):
            if part.alternates is not None:
                for alt in part.alternates:
                    yield from _parts_ids(alt)


def _arith_ids(expr: ArithExpr) -> Iterator[ShId]:
    for node in iter_arith(expr):
        if isinstance(node, ArithVar):
            yield node.id
        elif isinstance(node, ArithPart):
            yield from _parts_ids([node.part])


# --------------------------------------------------------------------------
# datum rendering


def program_to_datum(program: Program, *, scopes: bool = False) -> Datum:
    return ("program", *(_cmd_datum(c, scopes) for c in program.commands))


def pretty(program: Program, *, scopes: bool = False) -> str:
    datum = program_to_datum(program, scopes=scopes)
    assert isinstance(datum, tuple)
    lines = [str(datum[0])]
    lines.extend("  " + _datum_str(c) for c in datum[1:])
    return "(" + "\n".join(lines) + ")"


def _datum_str(datum: Datum) -> str:
    if isinstance(datum, str):
        return datum
    return "(" + " ".join(_datum_str(d) for d in datum) + ")"


def _id_datum(sh_id: ShId, scopes: bool) -> Datum:
    base: tuple[Datum, ...] = ("id", sh_id.sym, sh_id.kind.value)
    if scopes:
        names = tuple(sorted(sc.name for sc in sh_id.scopes))
        return (*base, names)
    return base


def _word_datum(word: Word, scopes: bool) -> Datum:
    return ("word", *(_part_datum(p, scopes) for p in word.parts))


def _subscript_datum(
    subscript: Union[ArithExpr, Lit, None], raw: str | None, scopes: bool
) -> Datum | None:
    if isinstance(subscript, Lit):
        return ("subscript", subscript.text)
    if subscript is not None:
        return ("subscript", _arith_datum(subscript, scopes))
    if raw is not None:
        return ("subscript", repr(raw))
    return None


def _part_datum(part: WordPart, scopes: bool) -> Datum:
    if isinstance(part, Lit):
        return ("lit", repr(part.text))
    if isinstance(part, Escape):
        return ("esc", repr(part.ch))
    if isinstance(part, SQuote):
        return ("squote", repr(part.text))
    if isinstance(part, AnsiCQuote):
        return ("ansi-c", repr(part.text))
    if isinstance(part, DQuote):
        head: tuple[Datum, ...] = ("dquote", "locale") if part.locale else ("dquote",)
        return (*head, *(_part_datum(p, scopes) for p in part.parts))
    if isinstance(part, Param):
        items: list[Datum] = ["param", _id_datum(part.id, scopes)]
        if part.is_length:
            items.append("length")
        if part.indirect:
            items.append("indirect")
        sub = _subscript_datum(part.subscript, part.subscript_raw, scopes)
        if sub is not None:
            items.append(sub)
        if part.op is not None:
            items.append(part.op)
        if part.pattern is not None:
            items.append(("pattern", *(_part_datum(p, scopes) for p in part.pattern)))
        if part.word is not None:
            items.append(("word", *(_part_datum(p, scopes) for p in part.word)))
        if part.offset is not None:
            items.append(("offset", _arith_datum(part.offset, scopes)))
        if part.length is not None:
            items.append(("length", _arith_datum(part.length, scopes)))
        return tuple(items)
    if isinstance(part, CmdSub):
        tag = "cmdsub" if part.style is CmdSubStyle.DOLLAR else "backtick"
        if part.program is None:
            return (tag, repr(part.raw))
        return (tag, program_to_datum(part.program, scopes=scopes))
    if isinstance(part, ProcSub):
        if part.program is None:
            return ("proc-sub", part.direction, repr(part.raw))
        return (
            "proc-sub",
            part.direction,
            program_to_datum(part.program, scopes=scopes),
        )
    if isinstance(part, BraceExpand):
        if part.range is not None:
            start, end, step = part.range
            if step is None:
                return ("brace-range", start, end)
            return ("brace-range", start, end, step)
        assert part.alternates is not None
        return (
            "brace",
            *(
                ("alt", *(_part_datum(p, scopes) for p in alt))
                for alt in part.alternates
            ),
        )
    return ("arith", _arith_datum(part.expr, scopes))


def _arith_datum(expr: ArithExpr, scopes: bool) -> Datum:
    if isinstance(expr, ArithNum):
        return ("arith-num", expr.text)
    if isinstance(expr, ArithVar):
        items: list[Datum] = ["arith-var", _id_datum(expr.id, scopes)]
        if expr.subscript is not None:
            items.append(_arith_datum(expr.subscript, scopes))
        return tuple(items)
    if isinstance(expr, ArithUnary):
        tag = "arith-postfix" if expr.postfix else "arith-unary"
        return (tag, expr.op, _arith_datum(expr.operand, scopes))
    if isinstance(expr, ArithBinary):
        return (
            "arith-binary",
            expr.op,
            _arith_datum(expr.left, scopes),
            _arith_datum(expr.right, scopes),
        )
    if isinstance(expr, ArithAssign):
        return (
            "arith-assign",
            expr.op,
            _arith_datum(expr.target, scopes),
            _arith_datum(expr.value, scopes),
        )
    if isinstance(expr, ArithTernary):
        return (
            "arith-ternary",
            _arith_datum(expr.cond, scopes),
            _arith_datum(expr.then, scopes),
            _arith_datum(expr.otherwise, scopes),
        )
    if isinstance(expr, ArithGroup):
        return ("arith-group", _arith_datum(expr.inner, scopes))
    return ("arith-part", _part_datum(expr.part, scopes))


def _cond_datum(expr: CondExpr, scopes: bool) -> Datum:
    if isinstance(expr, Word):
        return _word_datum(expr, scopes)
    if isinstance(expr, CondUnary):
        return (
            "cond-unary",
            expr.op_id.sym,
            _word_datum(expr.operand, scopes),
        )
    if isinstance(expr, CondBinary):
        return (
            "cond-binary",
            expr.op_id.sym,
            _word_datum(expr.left, scopes),
            _word_datum(expr.right, scopes),
        )
    if isinstance(expr, CondNot):
        return ("cond-not", _cond_datum(expr.operand, scopes))
    if isinstance(expr, CondAnd):
        return (
            "cond-and",
            _cond_datum(expr.left, scopes),
            _cond_datum(expr.right, scopes),
        )
    if isinstance(expr, CondOr):
        return (
            "cond-or",
            _cond_datum(expr.left, scopes),
            _cond_datum(expr.right, scopes),
        )
    return ("cond-group", _cond_datum(expr.inner, scopes))


def _redirect_datum(redirect: Redirect, scopes: bool) -> Datum:
    items: list[Datum] = ["redirect"]
    if redirect.n is not None:
        items.append(str(redirect.n))
    if redirect.fd_var is not None:
        items.append(("fd-var", _id_datum(redirect.fd_var, scopes)))
    items.append(_id_datum(redirect.op_id, scopes))
    if redirect.move:
        items.append("move")
    if isinstance(redirect.target, HereDoc):
        hd = redirect.target
        body: tuple[Datum, ...] = ()
        if hd.body is not None:
            body = tuple(_part_datum(p, scopes) for p in hd.body)
        items.append(("heredoc", _word_datum(hd.delim, scopes), *body))
    else:
        items.append(_word_datum(redirect.target, scopes))
    return tuple(items)


def _body_datum(tag: str, commands: list[Command], scopes: bool) -> Datum:
    return (tag, *(_cmd_datum(c, scopes) for c in commands))


def _assign_datum(assign: Assign, scopes: bool) -> Datum:
    items: list[Datum] = ["assign", _id_datum(assign.name_id, scopes)]
    if assign.append:
        items.append("append")
    sub = _subscript_datum(assign.subscript, assign.subscript_raw, scopes)
    if sub is not None:
        items.append(sub)
    if assign.array_items is not None:
        items.append(
            ("array", *(_array_item_datum(item, scopes) for item in assign.array_items))
        )
    items.append(_word_datum(assign.word, scopes))
    return tuple(items)


def _array_item_datum(item: ArrayItem, scopes: bool) -> Datum:
    items: list[Datum] = ["item"]
    sub = _subscript_datum(item.subscript, item.subscript_raw, scopes)
    if sub is not None:
        items.append(sub)
    items.append(_word_datum(item.word, scopes))
    return tuple(items)


def _cmd_datum(cmd: Command, scopes: bool) -> Datum:
    if isinstance(cmd, Simple):
        items: list[Datum] = ["simple"]
        for assign in cmd.assigns:
            items.append(_assign_datum(assign, scopes))
        for word in cmd.words:
            items.append(_word_datum(word, scopes))
        for redirect in cmd.redirects:
            items.append(_redirect_datum(redirect, scopes))
        return tuple(items)
    if isinstance(cmd, Pipeline):
        items = ["pipeline"]
        if cmd.time_id is not None:
            items.append(_id_datum(cmd.time_id, scopes))
            if cmd.time_p:
                items.append("-p")
        if cmd.bang_id is not None:
            items.append(_id_datum(cmd.bang_id, scopes))
        for i, sub in enumerate(cmd.cmds):
            if i > 0:
                items.append(_id_datum(cmd.pipe_ids[i - 1], scopes))
            items.append(_cmd_datum(sub, scopes))
        return tuple(items)
    if isinstance(cmd, AndOr):
        return (
            "and-or",
            _cmd_datum(cmd.left, scopes),
            _id_datum(cmd.op_id, scopes),
            _cmd_datum(cmd.right, scopes),
        )
    if isinstance(cmd, Async):
        return ("async", _cmd_datum(cmd.cmd, scopes))
    if isinstance(cmd, Subshell):
        return _with_redirects(
            _body_datum("subshell", cmd.body, scopes), cmd.redirects, scopes
        )
    if isinstance(cmd, BraceGroup):
        return _with_redirects(
            _body_datum("brace-group", cmd.body, scopes), cmd.redirects, scopes
        )
    if isinstance(cmd, If):
        items = ["if"]
        for test, body in cmd.clauses:
            items.append(_body_datum("test", test, scopes))
            items.append(_body_datum("then", body, scopes))
        if cmd.else_body is not None:
            items.append(_body_datum("else", cmd.else_body, scopes))
        return _with_redirects(tuple(items), cmd.redirects, scopes)
    if isinstance(cmd, While):
        return _with_redirects(
            (
                "while",
                _body_datum("test", cmd.test, scopes),
                _body_datum("body", cmd.body, scopes),
            ),
            cmd.redirects,
            scopes,
        )
    if isinstance(cmd, Until):
        return _with_redirects(
            (
                "until",
                _body_datum("test", cmd.test, scopes),
                _body_datum("body", cmd.body, scopes),
            ),
            cmd.redirects,
            scopes,
        )
    if isinstance(cmd, (For, Select)):
        tag = "for" if isinstance(cmd, For) else "select"
        items = [tag, _id_datum(cmd.var_id, scopes)]
        if cmd.in_words is not None:
            items.append(("in", *(_word_datum(w, scopes) for w in cmd.in_words)))
        items.append(_body_datum("body", cmd.body, scopes))
        return _with_redirects(tuple(items), cmd.redirects, scopes)
    if isinstance(cmd, ForArith):
        items = ["for-arith"]
        for tag, expr in (("init", cmd.init), ("cond", cmd.cond), ("step", cmd.step)):
            if expr is None:
                items.append((tag,))
            else:
                items.append((tag, _arith_datum(expr, scopes)))
        items.append(_body_datum("body", cmd.body, scopes))
        return _with_redirects(tuple(items), cmd.redirects, scopes)
    if isinstance(cmd, ArithCommand):
        return _with_redirects(
            ("arith-cmd", _arith_datum(cmd.expr, scopes)), cmd.redirects, scopes
        )
    if isinstance(cmd, Coproc):
        items = ["coproc"]
        if cmd.name_id is not None:
            items.append(_id_datum(cmd.name_id, scopes))
        items.append(_cmd_datum(cmd.cmd, scopes))
        return tuple(items)
    if isinstance(cmd, CondCmd):
        return _with_redirects(
            ("cond", _cond_datum(cmd.expr, scopes)), cmd.redirects, scopes
        )
    if isinstance(cmd, Case):
        items = ["case", _word_datum(cmd.subject, scopes)]
        for item in cmd.items:
            item_datum: tuple[Datum, ...] = (
                "item",
                ("patterns", *(_word_datum(w, scopes) for w in item.patterns)),
                _body_datum("body", item.body, scopes),
            )
            if item.terminator in (";&", ";;&"):
                item_datum = (*item_datum, item.terminator)
            items.append(item_datum)
        return _with_redirects(tuple(items), cmd.redirects, scopes)
    return _with_redirects(
        (
            "fundef",
            _id_datum(cmd.name_id, scopes),
            _cmd_datum(cmd.body, scopes),
        ),
        cmd.redirects,
        scopes,
    )


def _with_redirects(datum: Datum, redirects: list[Redirect], scopes: bool) -> Datum:
    if not redirects:
        return datum
    assert isinstance(datum, tuple)
    return (*datum, *(_redirect_datum(r, scopes) for r in redirects))
