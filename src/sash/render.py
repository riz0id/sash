"""S-expression-style rendering and identifier traversal.

`program_to_datum` produces nested tuples of strings for inspection;
`iter_ids`/`find_ids` yield every ShId leaf in a program, including those
inside nested command-substitution programs, here-document bodies, and
arithmetic expansions — so questions like "which `|` leaves resolve to the
pipe operator?" are one filter away.
"""

from __future__ import annotations

from typing import Iterator, Union

from .nodes import (
    AndOr,
    Arith,
    Async,
    BraceGroup,
    Case,
    CmdSub,
    CmdSubStyle,
    Command,
    DQuote,
    Escape,
    For,
    FunDef,
    HereDoc,
    IdKind,
    If,
    Lit,
    Param,
    Pipeline,
    Program,
    Redirect,
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
            yield assign.name_id
            yield from _parts_ids(assign.word.parts)
        for word in cmd.words:
            yield from _parts_ids(word.parts)
        yield from _redirect_ids(cmd.redirects)
    elif isinstance(cmd, Pipeline):
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
    elif isinstance(cmd, For):
        yield from cmd.keywords
        yield cmd.var_id
        for word in cmd.in_words or []:
            yield from _parts_ids(word.parts)
        for sub_cmd in cmd.body:
            yield from _cmd_ids(sub_cmd)
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


def _redirect_ids(redirects: list[Redirect]) -> Iterator[ShId]:
    for redirect in redirects:
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
            if part.word is not None:
                yield from _parts_ids(part.word)
        elif isinstance(part, DQuote):
            yield from _parts_ids(list(part.parts))
        elif isinstance(part, CmdSub):
            if part.program is not None:
                yield from iter_ids(part.program)
        elif isinstance(part, Arith):
            for sub in part.parts:
                if isinstance(sub, ShId):
                    yield sub
                elif not isinstance(sub, Lit):
                    yield from _parts_ids([sub])


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


def _part_datum(part: WordPart, scopes: bool) -> Datum:
    if isinstance(part, Lit):
        return ("lit", repr(part.text))
    if isinstance(part, Escape):
        return ("esc", repr(part.ch))
    if isinstance(part, SQuote):
        return ("squote", repr(part.text))
    if isinstance(part, DQuote):
        return ("dquote", *(_part_datum(p, scopes) for p in part.parts))
    if isinstance(part, Param):
        items: list[Datum] = ["param", _id_datum(part.id, scopes)]
        if part.is_length:
            items.append("length")
        if part.op is not None:
            items.append(part.op)
        if part.word is not None:
            items.append(("word", *(_part_datum(p, scopes) for p in part.word)))
        return tuple(items)
    if isinstance(part, CmdSub):
        tag = "cmdsub" if part.style is CmdSubStyle.DOLLAR else "backtick"
        if part.program is None:
            return (tag, repr(part.raw))
        return (tag, program_to_datum(part.program, scopes=scopes))
    items = ["arith"]
    for sub in part.parts:
        if isinstance(sub, ShId):
            items.append(_id_datum(sub, scopes))
        else:
            items.append(_part_datum(sub, scopes))
    return tuple(items)


def _redirect_datum(redirect: Redirect, scopes: bool) -> Datum:
    items: list[Datum] = ["redirect"]
    if redirect.n is not None:
        items.append(str(redirect.n))
    items.append(_id_datum(redirect.op_id, scopes))
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


def _cmd_datum(cmd: Command, scopes: bool) -> Datum:
    if isinstance(cmd, Simple):
        items: list[Datum] = ["simple"]
        for assign in cmd.assigns:
            items.append(
                (
                    "assign",
                    _id_datum(assign.name_id, scopes),
                    _word_datum(assign.word, scopes),
                )
            )
        for word in cmd.words:
            items.append(_word_datum(word, scopes))
        for redirect in cmd.redirects:
            items.append(_redirect_datum(redirect, scopes))
        return tuple(items)
    if isinstance(cmd, Pipeline):
        items = ["pipeline"]
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
    if isinstance(cmd, For):
        items = ["for", _id_datum(cmd.var_id, scopes)]
        if cmd.in_words is not None:
            items.append(("in", *(_word_datum(w, scopes) for w in cmd.in_words)))
        items.append(_body_datum("body", cmd.body, scopes))
        return _with_redirects(tuple(items), cmd.redirects, scopes)
    if isinstance(cmd, Case):
        items = ["case", _word_datum(cmd.subject, scopes)]
        for item in cmd.items:
            items.append(
                (
                    "item",
                    ("patterns", *(_word_datum(w, scopes) for w in item.patterns)),
                    _body_datum("body", item.body, scopes),
                )
            )
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
