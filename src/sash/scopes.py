"""Binding as sets of scopes (Flatt, POPL 2016) for POSIX sh.

A minimal mirror of the Racket expander's scope machinery
(racket/src/expander/syntax/scope.rkt): identifiers carry a set of scopes; a
binder binds a reference iff the names match and the binder's scope set is a
subset of the reference's; the binder with the largest such set wins, and the
absence of a unique maximum is an ambiguity error.

Deliberately omitted relative to the expander: phases, multi-scopes,
flip-scope (there is no macro expansion, so scope painting is monotonic),
laziness, and caching.

A binding for (namespace, sym, scope-set) is stored in the table of the
maximum scope of that set, so resolution only consults tables reachable from
the reference's own scopes. Re-registering the same cell appends a site
rather than replacing: ``x=1; x=2`` are two writes to one variable cell.
Cells are never removed — scope sets model *which cell* a name denotes, not
whether it is currently set.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Literal

from .nodes import ShId


class Namespace(enum.Enum):
    """sh's separate name spaces, plus one for control operators."""

    VARIABLE = "variable"
    FUNCTION = "function"
    OPERATOR = "operator"


class ScopeKind(enum.Enum):
    CORE = "core"
    TOP = "top"
    FUNCTION = "function"
    SUBSHELL = "subshell"
    COMMAND_SUB = "command-sub"


class BindingKind(enum.Enum):
    ASSIGNMENT = "assignment"
    FOR = "for"
    FUNCTION = "function"
    OPERATOR = "operator"
    SPECIAL_BUILTIN = "special-builtin"
    POSITIONAL = "positional"
    SPECIAL_PARAM = "special-param"
    LOCAL = "local"
    EXPORT = "export"
    READONLY = "readonly"


SiteKind = Literal["binder", "unset", "implicit"]


@dataclass(eq=False)
class Site:
    kind: SiteKind
    id: ShId | None  # None for implicit sites


@dataclass(eq=False)
class Binding:
    kind: BindingKind
    sites: list[Site] = field(default_factory=list)


@dataclass(eq=False)  # identity hash: scopes are object identities
class Scope:
    id: int
    kind: ScopeKind
    name: str
    table: dict[Namespace, dict[str, dict[frozenset[Scope], Binding]]] = field(
        default_factory=dict
    )


class ShAmbiguityError(Exception):
    def __init__(
        self,
        sym: str,
        candidates: list[tuple[frozenset[Scope], Binding]],
    ) -> None:
        names = [sorted(sc.name for sc in scopes) for scopes, _ in candidates]
        super().__init__(f"identifier {sym!r} is ambiguous among scope sets {names}")
        self.sym = sym
        self.candidates = candidates


class BindingStore:
    """Scope allocator plus the core and top scopes of one bound program."""

    def __init__(self) -> None:
        self._counter = 0
        self.core = self.new_scope(ScopeKind.CORE, "core")
        self.top = self.new_scope(ScopeKind.TOP, "top")

    def new_scope(self, kind: ScopeKind, name: str) -> Scope:
        self._counter += 1
        return Scope(self._counter, kind, name)

    def add_binding(
        self,
        ns: Namespace,
        sym: str,
        scopes: frozenset[Scope],
        kind: BindingKind,
        site: Site,
    ) -> Binding:
        """Create or extend the cell for (ns, sym, scopes)."""
        max_scope = max(scopes, key=lambda sc: sc.id)
        by_sym = max_scope.table.setdefault(ns, {})
        by_scopes = by_sym.setdefault(sym, {})
        binding = by_scopes.get(scopes)
        if binding is None:
            binding = Binding(kind)
            by_scopes[scopes] = binding
        binding.sites.append(site)
        return binding


def paint(id: ShId, scopes: frozenset[Scope]) -> None:
    id.scopes |= scopes


def resolve(
    id: ShId,
    ns: Namespace = Namespace.VARIABLE,
    *,
    ambiguous_ok: bool = False,
) -> Binding | None:
    """Largest-subset-wins resolution, transcribing the expander's fold.

    Returns None when unbound. Raises ShAmbiguityError when no candidate's
    scope set is a superset of all others, unless ambiguous_ok is true (then
    returns None).
    """
    candidates: list[tuple[frozenset[Scope], Binding]] = []
    for sc in id.scopes:
        cells = sc.table.get(ns, {}).get(id.sym, {})
        for binder_scopes, binding in cells.items():
            if binder_scopes <= id.scopes:
                candidates.append((binder_scopes, binding))
    if not candidates:
        return None
    best = candidates[0]
    for cand in candidates[1:]:
        if best[0] <= cand[0]:
            best = cand
    for cand in candidates:
        if not (cand[0] <= best[0]):
            if ambiguous_ok:
                return None
            raise ShAmbiguityError(id.sym, candidates)
    return best[1]


def bound_identifier_eq(a: ShId, b: ShId) -> bool:
    """Same name and identical scope sets: each would bind the other."""
    return a.sym == b.sym and a.scopes == b.scopes


def free_identifier_eq(a: ShId, b: ShId, ns: Namespace = Namespace.VARIABLE) -> bool:
    """Both resolve to the same binding cell (or both unbound, same name)."""
    ba = resolve(a, ns, ambiguous_ok=True)
    bb = resolve(b, ns, ambiguous_ok=True)
    if ba is None and bb is None:
        return a.sym == b.sym
    return ba is bb
