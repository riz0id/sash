"""Data structures for the sash reader/parser.

Tokens, word parts, and AST nodes. Identifier-like leaves are ``ShId``:
classification (identifier vs. string data) is decided by the reader, so
every ``ShId`` is born with an empty scope set; the bind pass paints scope
sets in place. A shell metacharacter inside quotes only ever appears as a
character in string data and never becomes a ``ShId``.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .scopes import Scope


@dataclass(frozen=True)
class Loc:
    src: str
    line: int
    col: int
    pos: int
    span: int
    synthetic: bool = False


class ShParseError(Exception):
    def __init__(self, message: str, loc: Loc) -> None:
        super().__init__(f"{loc.src}:{loc.line}:{loc.col}: {message}")
        self.message = message
        self.loc = loc


class IdKind(enum.Enum):
    COMMAND = "command"
    OPERATOR = "operator"
    VARIABLE_REF = "variable-ref"
    VARIABLE_BINDER = "variable-binder"
    FOR_VARIABLE = "for-variable"
    FUNCTION_NAME = "function-name"
    SPECIAL_PARAM = "special-param"
    POSITIONAL = "positional"


@dataclass(eq=False)
class ShId:
    sym: str
    kind: IdKind
    loc: Loc
    # Painted in place by the bind pass; empty at read time because
    # classification precedes resolution.
    scopes: frozenset[Scope] = frozenset()


# --------------------------------------------------------------------------
# Word parts


@dataclass(eq=False)
class Lit:
    text: str
    loc: Loc


@dataclass(eq=False)
class Escape:
    ch: str
    loc: Loc


@dataclass(eq=False)
class SQuote:
    text: str
    loc: Loc


@dataclass(eq=False)
class DQuote:
    parts: list[DQuotePart]
    loc: Loc


@dataclass(eq=False)
class Param:
    id: ShId
    op: str | None  # ':-', '-', ':=', '=', ':?', '?', ':+', '+', '%', '%%', '#', '##'
    word: list[WordPart] | None
    is_length: bool
    loc: Loc


class CmdSubStyle(enum.Enum):
    DOLLAR = "dollar"
    BACKTICK = "backtick"


@dataclass(eq=False)
class CmdSub:
    raw: str
    style: CmdSubStyle
    loc: Loc
    # Location of the first character of `raw` in the outer source, so the
    # recursive parse reports absolute positions. Synthetic for backticks,
    # whose escape-stripping shifts offsets.
    raw_loc: Loc | None = None
    # Filled by the parser via recursive parse of `raw`.
    program: Program | None = None


@dataclass(eq=False)
class Arith:
    # Variable references surface as ShId/Param; everything else stays Lit.
    parts: list[ArithPart]
    loc: Loc


DQuotePart = Union[Lit, Escape, Param, CmdSub, Arith]
WordPart = Union[Lit, Escape, SQuote, DQuote, Param, CmdSub, Arith]
ArithPart = Union[Lit, Param, CmdSub, ShId]


@dataclass(eq=False)
class Word:
    parts: list[WordPart]
    loc: Loc


# --------------------------------------------------------------------------
# Tokens


@dataclass(eq=False)
class TokOp:
    id: ShId
    loc: Loc


@dataclass(eq=False)
class TokWord:
    word: Word
    loc: Loc


@dataclass(eq=False)
class TokIoNumber:
    n: int
    loc: Loc


@dataclass(eq=False)
class TokNewline:
    loc: Loc


@dataclass(eq=False)
class TokEof:
    loc: Loc


Token = Union[TokOp, TokWord, TokIoNumber, TokNewline, TokEof]


# --------------------------------------------------------------------------
# AST


@dataclass(eq=False)
class HereDoc:
    op_id: ShId
    delim: Word
    quoted: bool
    strip_tabs: bool
    loc: Loc
    # Filled by the lexer when the pending here-doc body is collected at the
    # next newline; the same object is shared with the Redirect target.
    body: list[WordPart] | None = None


@dataclass(eq=False)
class Redirect:
    n: int | None
    op_id: ShId
    target: Union[Word, HereDoc]
    loc: Loc


@dataclass(eq=False)
class Assign:
    name_id: ShId
    word: Word
    loc: Loc


@dataclass(eq=False)
class Simple:
    assigns: list[Assign]
    words: list[Word]
    # Synthesized when the first word is a single unquoted literal, so
    # "does this name resolve to a function?" is well-posed.
    cmd_id: ShId | None
    redirects: list[Redirect]
    loc: Loc


@dataclass(eq=False)
class Pipeline:
    bang_id: ShId | None
    cmds: list[Command]
    pipe_ids: list[ShId]
    loc: Loc


@dataclass(eq=False)
class AndOr:
    left: Command
    op_id: ShId  # '&&' or '||'
    right: Command
    loc: Loc


@dataclass(eq=False)
class Async:
    cmd: Command
    amp_id: ShId
    loc: Loc


@dataclass(eq=False)
class Subshell:
    body: list[Command]
    redirects: list[Redirect]
    keywords: list[ShId]  # '(' and ')'
    loc: Loc


@dataclass(eq=False)
class BraceGroup:
    body: list[Command]
    redirects: list[Redirect]
    keywords: list[ShId]  # '{' and '}'
    loc: Loc


@dataclass(eq=False)
class If:
    clauses: list[tuple[list[Command], list[Command]]]  # (test, then) pairs
    else_body: list[Command] | None
    redirects: list[Redirect]
    keywords: list[ShId]
    loc: Loc


@dataclass(eq=False)
class While:
    test: list[Command]
    body: list[Command]
    redirects: list[Redirect]
    keywords: list[ShId]
    loc: Loc


@dataclass(eq=False)
class Until:
    test: list[Command]
    body: list[Command]
    redirects: list[Redirect]
    keywords: list[ShId]
    loc: Loc


@dataclass(eq=False)
class For:
    var_id: ShId
    in_words: list[Word] | None  # None => implicit "$@"
    body: list[Command]
    redirects: list[Redirect]
    keywords: list[ShId]
    loc: Loc


@dataclass(eq=False)
class CaseItem:
    patterns: list[Word]
    body: list[Command]
    keywords: list[ShId]
    loc: Loc


@dataclass(eq=False)
class Case:
    subject: Word
    items: list[CaseItem]
    redirects: list[Redirect]
    keywords: list[ShId]
    loc: Loc


@dataclass(eq=False)
class FunDef:
    name_id: ShId
    body: Command
    redirects: list[Redirect]
    keywords: list[ShId]  # '(' and ')'
    loc: Loc


Command = Union[
    Simple,
    Pipeline,
    AndOr,
    Async,
    Subshell,
    BraceGroup,
    If,
    While,
    Until,
    For,
    Case,
    FunDef,
]


@dataclass(eq=False)
class Program:
    commands: list[Command]
    loc: Loc


# --------------------------------------------------------------------------
# Predicates

RESERVED_WORDS = frozenset(
    [
        "if",
        "then",
        "elif",
        "else",
        "fi",
        "for",
        "in",
        "while",
        "until",
        "do",
        "done",
        "case",
        "esac",
        "{",
        "}",
        "!",
    ]
)

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

SPECIAL_PARAMS = frozenset(["@", "*", "#", "?", "-", "$", "!", "0"])


def is_valid_name(s: str) -> bool:
    return _NAME_RE.match(s) is not None


def word_single_unquoted_lit(word: Word) -> str | None:
    """The word's text iff it is exactly one unquoted literal part.

    Reserved-word promotion and assignment/NAME recognition require this:
    any quoting (Escape/SQuote/DQuote) or expansion defeats recognition.
    """
    if len(word.parts) == 1 and isinstance(word.parts[0], Lit):
        return word.parts[0].text
    return None


def word_to_reserved(word: Word) -> str | None:
    text = word_single_unquoted_lit(word)
    if text is not None and text in RESERVED_WORDS:
        return text
    return None
