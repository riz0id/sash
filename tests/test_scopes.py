"""Unit tests for the scope-set machinery: the resolve fold in isolation."""

from __future__ import annotations

import pytest

from sash import (
    BindingKind,
    BindingStore,
    IdKind,
    Loc,
    Namespace,
    Scope,
    ScopeKind,
    ShAmbiguityError,
    ShId,
    Site,
    bound_identifier_eq,
    free_identifier_eq,
    resolve,
)
from sash.scopes import paint

LOC = Loc("<test>", 1, 0, 0, 0)


def make_id(sym: str, scopes: frozenset[Scope]) -> ShId:
    sh_id = ShId(sym, IdKind.VARIABLE_REF, LOC)
    paint(sh_id, scopes)
    return sh_id


def binder_site() -> Site:
    return Site("binder", None)


def test_unbound_resolves_to_none() -> None:
    store = BindingStore()
    ref = make_id("x", frozenset({store.core, store.top}))
    assert resolve(ref) is None


def test_simple_resolution() -> None:
    store = BindingStore()
    base = frozenset({store.core, store.top})
    binding = store.add_binding(
        Namespace.VARIABLE, "x", base, BindingKind.ASSIGNMENT, binder_site()
    )
    ref = make_id("x", base)
    assert resolve(ref) is binding


def test_binder_scope_set_must_be_subset() -> None:
    store = BindingStore()
    sub = store.new_scope(ScopeKind.SUBSHELL, "sub")
    inner = frozenset({store.core, store.top, sub})
    store.add_binding(
        Namespace.VARIABLE, "x", inner, BindingKind.ASSIGNMENT, binder_site()
    )
    outer_ref = make_id("x", frozenset({store.core, store.top}))
    assert resolve(outer_ref) is None


def test_largest_subset_wins() -> None:
    store = BindingStore()
    sub = store.new_scope(ScopeKind.SUBSHELL, "sub")
    base = frozenset({store.core, store.top})
    inner = base | {sub}
    global_binding = store.add_binding(
        Namespace.VARIABLE, "x", base, BindingKind.ASSIGNMENT, binder_site()
    )
    local_binding = store.add_binding(
        Namespace.VARIABLE, "x", inner, BindingKind.LOCAL, binder_site()
    )
    inner_ref = make_id("x", inner)
    outer_ref = make_id("x", base)
    assert resolve(inner_ref) is local_binding
    assert resolve(outer_ref) is global_binding


def test_ambiguity_raises() -> None:
    store = BindingStore()
    a = store.new_scope(ScopeKind.SUBSHELL, "a")
    b = store.new_scope(ScopeKind.SUBSHELL, "b")
    base = frozenset({store.core, store.top})
    store.add_binding(
        Namespace.VARIABLE, "x", base | {a}, BindingKind.ASSIGNMENT, binder_site()
    )
    store.add_binding(
        Namespace.VARIABLE, "x", base | {b}, BindingKind.ASSIGNMENT, binder_site()
    )
    ref = make_id("x", base | {a, b})
    with pytest.raises(ShAmbiguityError):
        resolve(ref)
    assert resolve(ref, ambiguous_ok=True) is None


def test_sites_append_on_reregistration() -> None:
    store = BindingStore()
    base = frozenset({store.core, store.top})
    first = store.add_binding(
        Namespace.VARIABLE, "x", base, BindingKind.ASSIGNMENT, binder_site()
    )
    second = store.add_binding(
        Namespace.VARIABLE, "x", base, BindingKind.ASSIGNMENT, binder_site()
    )
    assert first is second
    assert len(first.sites) == 2


def test_binding_stored_in_max_scope() -> None:
    store = BindingStore()
    sub = store.new_scope(ScopeKind.SUBSHELL, "sub")
    scopes = frozenset({store.core, store.top, sub})
    store.add_binding(
        Namespace.VARIABLE, "x", scopes, BindingKind.ASSIGNMENT, binder_site()
    )
    assert "x" in sub.table[Namespace.VARIABLE]
    assert Namespace.VARIABLE not in store.core.table


def test_namespaces_are_disjoint() -> None:
    store = BindingStore()
    base = frozenset({store.core, store.top})
    store.add_binding(
        Namespace.FUNCTION, "x", base, BindingKind.FUNCTION, binder_site()
    )
    ref = make_id("x", base)
    assert resolve(ref, Namespace.VARIABLE) is None
    assert resolve(ref, Namespace.FUNCTION) is not None


def test_bound_identifier_eq() -> None:
    store = BindingStore()
    base = frozenset({store.core, store.top})
    sub = store.new_scope(ScopeKind.SUBSHELL, "sub")
    assert bound_identifier_eq(make_id("x", base), make_id("x", base))
    assert not bound_identifier_eq(make_id("x", base), make_id("y", base))
    assert not bound_identifier_eq(make_id("x", base), make_id("x", base | {sub}))


def test_free_identifier_eq_same_cell() -> None:
    store = BindingStore()
    sub = store.new_scope(ScopeKind.SUBSHELL, "sub")
    base = frozenset({store.core, store.top})
    store.add_binding(
        Namespace.VARIABLE, "x", base, BindingKind.ASSIGNMENT, binder_site()
    )
    # Both refs reach the same cell even though their scope sets differ.
    assert free_identifier_eq(make_id("x", base), make_id("x", base | {sub}))


def test_free_identifier_eq_unbound_falls_back_to_sym() -> None:
    store = BindingStore()
    base = frozenset({store.core, store.top})
    assert free_identifier_eq(make_id("q", base), make_id("q", base))
    assert not free_identifier_eq(make_id("q", base), make_id("r", base))
