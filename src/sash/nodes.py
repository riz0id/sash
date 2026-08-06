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


class Dialect(enum.Enum):
    POSIX = "posix"
    BASH = "bash"


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
    locale: bool = False  # bash $"..."


@dataclass(eq=False)
class AnsiCQuote:
    text: str  # decoded value
    raw: str  # source text between the quotes
    loc: Loc


@dataclass(eq=False)
class Param:
    id: ShId
    # POSIX: ':-', '-', ':=', '=', ':?', '?', ':+', '+', '%', '%%', '#', '##'
    # bash: '/', '//', '/#', '/%', ':', '^', '^^', ',', ',,', '@X', '!*', '!@'
    op: str | None
    word: list[WordPart] | None
    is_length: bool
    loc: Loc
    pattern: list[WordPart] | None = None  # bash replace ops: text before '/'
    offset: ArithExpr | None = None  # bash ${x:off}
    length: ArithExpr | None = None  # bash ${x:off:len}
    indirect: bool = False  # bash ${!x}
    # bash ${a[i]}: '@' and '*' stored as Lit; None with a non-None
    # subscript_raw when the raw text is not arithmetic (assoc-style key)
    subscript: Union[ArithExpr, Lit, None] = None
    subscript_raw: str | None = None


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
class ProcSub:
    direction: str  # '<' for <(cmd), '>' for >(cmd)
    raw: str
    loc: Loc
    # Filled by the parser via recursive parse of `raw`.
    program: Program | None = None


@dataclass(eq=False)
class ArithNum:
    text: str
    loc: Loc


@dataclass(eq=False)
class ArithVar:
    id: ShId
    subscript: ArithExpr | None
    loc: Loc


@dataclass(eq=False)
class ArithUnary:
    op: str
    operand: ArithExpr
    loc: Loc
    postfix: bool = False


@dataclass(eq=False)
class ArithBinary:
    op: str
    left: ArithExpr
    right: ArithExpr
    loc: Loc


@dataclass(eq=False)
class ArithAssign:
    target: ArithVar
    op: str
    value: ArithExpr
    loc: Loc


@dataclass(eq=False)
class ArithTernary:
    cond: ArithExpr
    then: ArithExpr
    otherwise: ArithExpr
    loc: Loc


@dataclass(eq=False)
class ArithGroup:
    inner: ArithExpr
    loc: Loc


@dataclass(eq=False)
class ArithPart:
    part: Union[Param, CmdSub, Arith]
    loc: Loc


@dataclass(eq=False)
class Arith:
    expr: ArithExpr
    loc: Loc
    deprecated: bool = False  # bash $[...]


@dataclass(eq=False)
class BraceExpand:
    alternates: list[list[WordPart]] | None  # {a,b,...}; exactly one payload set
    range: tuple[str, str, str | None] | None  # {start..end} / {start..end..step}
    loc: Loc


ArithExpr = Union[
    ArithNum,
    ArithVar,
    ArithUnary,
    ArithBinary,
    ArithAssign,
    ArithTernary,
    ArithGroup,
    ArithPart,
]
DQuotePart = Union[Lit, Escape, Param, CmdSub, Arith]
WordPart = Union[
    Lit, Escape, SQuote, DQuote, AnsiCQuote, Param, CmdSub, Arith, BraceExpand, ProcSub
]


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
    here_string: bool = False  # bash <<<
    move: bool = False  # bash >&n- / <&n-
    fd_var: ShId | None = None  # bash {var}> forms


@dataclass(eq=False)
class ArrayItem:
    # bash `[i]=w` element of a compound assignment; subscript_raw is kept
    # because associative-array subscripts are strings, not arithmetic
    subscript: ArithExpr | None
    subscript_raw: str | None
    word: Word
    loc: Loc


@dataclass(eq=False)
class Assign:
    name_id: ShId
    word: Word
    loc: Loc
    subscript: ArithExpr | None = None  # bash a[expr]=v
    subscript_raw: str | None = None  # raw text; assoc keys are not arithmetic
    array_items: list[ArrayItem] | None = None  # bash a=(w1 [i]=w2 ...)
    append: bool = False  # bash += for scalars and arrays


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
    time_id: ShId | None = None  # bash `time` pipeline prefix
    time_p: bool = False  # bash `time -p`


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
class Select:
    var_id: ShId
    in_words: list[Word] | None  # None => implicit "$@"
    body: list[Command]
    redirects: list[Redirect]
    keywords: list[ShId]
    loc: Loc


@dataclass(eq=False)
class ForArith:
    init: ArithExpr | None
    cond: ArithExpr | None
    step: ArithExpr | None
    body: list[Command]
    redirects: list[Redirect]
    keywords: list[ShId]
    loc: Loc


@dataclass(eq=False)
class ArithCommand:
    expr: ArithExpr
    redirects: list[Redirect]
    keywords: list[ShId]  # '((' and '))'
    loc: Loc


@dataclass(eq=False)
class Coproc:
    name_id: ShId | None  # bash names the coprocess variable; COPROC when None
    cmd: Command
    keywords: list[ShId]  # 'coproc'
    loc: Loc


@dataclass(eq=False)
class CondUnary:
    op_id: ShId  # e.g. '-f', '-z'
    operand: Word
    loc: Loc


@dataclass(eq=False)
class CondBinary:
    op_id: ShId  # '==', '=', '!=', '=~', '<', '>', '-eq', ...
    left: Word
    right: Word
    loc: Loc


@dataclass(eq=False)
class CondNot:
    bang_id: ShId
    operand: CondExpr
    loc: Loc


@dataclass(eq=False)
class CondAnd:
    left: CondExpr
    right: CondExpr
    op_id: ShId
    loc: Loc


@dataclass(eq=False)
class CondOr:
    left: CondExpr
    right: CondExpr
    op_id: ShId
    loc: Loc


@dataclass(eq=False)
class CondGroup:
    inner: CondExpr
    lparen_id: ShId
    rparen_id: ShId
    loc: Loc


# A bare Word leaf tests non-emptiness of its expansion.
CondExpr = Union[Word, CondUnary, CondBinary, CondNot, CondAnd, CondOr, CondGroup]


@dataclass(eq=False)
class CondCmd:
    expr: CondExpr
    redirects: list[Redirect]
    keywords: list[ShId]  # '[[' and ']]'
    loc: Loc


@dataclass(eq=False)
class CaseItem:
    patterns: list[Word]
    body: list[Command]
    keywords: list[ShId]
    loc: Loc
    # ';;', bash ';&' / ';;&', or '' for a last item with no terminator
    terminator: str = ";;"


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
    Select,
    ForArith,
    ArithCommand,
    Case,
    FunDef,
    Coproc,
    CondCmd,
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

# Recognized positionally only in the BASH dialect.
BASH_RESERVED_WORDS = frozenset(["function", "select", "time", "coproc", "[["])

COND_UNARY_OPS = frozenset(
    "-a -b -c -d -e -f -g -h -k -p -r -s -t -u -w -x "
    "-G -L -N -O -S -o -v -R -z -n".split()
)

# '<' and '>' are also binary cond operators but arrive as operator tokens.
COND_BINARY_OPS = frozenset(
    [
        "==",
        "=",
        "!=",
        "=~",
        "-eq",
        "-ne",
        "-lt",
        "-le",
        "-gt",
        "-ge",
        "-nt",
        "-ot",
        "-ef",
    ]
)

_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")

SPECIAL_PARAMS = frozenset(["@", "*", "#", "?", "-", "$", "!", "0"])


def is_valid_name(s: str) -> bool:
    return _NAME_RE.match(s) is not None


def matching_bracket(text: str, start: int) -> int | None:
    """Index of the `]` matching the `[` at `start`, tracking nesting."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                return i
    return None


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
