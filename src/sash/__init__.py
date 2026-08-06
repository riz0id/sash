"""sash — POSIX sh reader/parser with scope-set variable binding.

The reader classifies every identifier-like leaf (commands, operators,
variable references and binders) before any binding machinery runs; the
bind pass paints scope sets in place and registers binders, so resolution
certifies — never re-derives — the reader's classification.
"""

from __future__ import annotations

from .bind import sh_bind
from .lexer import Lexer
from .nodes import (
    AndOr,
    Arith,
    Assign,
    Async,
    BraceGroup,
    Case,
    CaseItem,
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
    Loc,
    Param,
    Pipeline,
    Program,
    Redirect,
    ShId,
    ShParseError,
    Simple,
    SQuote,
    Subshell,
    Token,
    TokEof,
    Until,
    While,
    Word,
    WordPart,
)
from .parser import parse
from .render import find_ids, iter_ids, pretty, program_to_datum
from .scopes import (
    Binding,
    BindingKind,
    BindingStore,
    Namespace,
    Scope,
    ScopeKind,
    ShAmbiguityError,
    Site,
    bound_identifier_eq,
    free_identifier_eq,
    resolve,
)

__all__ = [
    "AndOr",
    "Arith",
    "Assign",
    "Async",
    "Binding",
    "BindingKind",
    "BindingStore",
    "BraceGroup",
    "Case",
    "CaseItem",
    "CmdSub",
    "CmdSubStyle",
    "Command",
    "DQuote",
    "Escape",
    "For",
    "FunDef",
    "HereDoc",
    "IdKind",
    "If",
    "Lit",
    "Loc",
    "Namespace",
    "Param",
    "Pipeline",
    "Program",
    "Redirect",
    "Scope",
    "ScopeKind",
    "ShAmbiguityError",
    "ShId",
    "ShParseError",
    "SQuote",
    "Simple",
    "Site",
    "Subshell",
    "Until",
    "While",
    "Word",
    "WordPart",
    "bound_identifier_eq",
    "find_ids",
    "free_identifier_eq",
    "identifier_binding",
    "iter_ids",
    "namespace_for",
    "pretty",
    "program_to_datum",
    "resolve",
    "resolve_identifier",
    "sh_bind",
    "sh_lex",
    "sh_read",
    "sh_read_program",
]


def sh_read(text: str, src: str = "<sh>") -> Program:
    return parse(text, src)


def sh_lex(text: str, src: str = "<sh>") -> list[Token]:
    lexer = Lexer(text, src)
    tokens: list[Token] = []
    while True:
        tok = lexer.next_token()
        tokens.append(tok)
        if isinstance(tok, TokEof):
            return tokens


def sh_read_program(
    text: str, src: str = "<sh>", *, recognize_local: bool = False
) -> tuple[Program, BindingStore]:
    program = parse(text, src)
    store = sh_bind(program, recognize_local=recognize_local)
    return program, store


def namespace_for(sh_id: ShId) -> Namespace:
    if sh_id.kind is IdKind.OPERATOR:
        return Namespace.OPERATOR
    if sh_id.kind in (IdKind.COMMAND, IdKind.FUNCTION_NAME):
        return Namespace.FUNCTION
    return Namespace.VARIABLE


def resolve_identifier(
    sh_id: ShId, ns: Namespace | None = None, *, ambiguous_ok: bool = False
) -> Binding | None:
    if ns is None:
        ns = namespace_for(sh_id)
    return resolve(sh_id, ns, ambiguous_ok=ambiguous_ok)


def identifier_binding(
    sh_id: ShId, ns: Namespace | None = None
) -> tuple[str, str, list[str], list[Loc | str]] | None:
    """Descriptive view of a resolved binding, echoing `identifier-binding`.

    Returns (binding-kind, sym, sorted scope names, site locations) or None
    when unbound. Implicit/unset sites appear as the strings "implicit" and
    "unset".
    """
    binding = resolve_identifier(sh_id, ns, ambiguous_ok=True)
    if binding is None:
        return None
    scope_names = sorted(sc.name for sc in sh_id.scopes)
    sites: list[Loc | str] = []
    for site in binding.sites:
        if site.id is None:
            sites.append(site.kind)
        else:
            sites.append(site.id.loc)
    return (binding.kind.value, sh_id.sym, scope_names, sites)
