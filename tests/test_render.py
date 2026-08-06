"""Identifier traversal and datum rendering."""

from __future__ import annotations

import dataclasses

from sash import (
    Dialect,
    IdKind,
    ShId,
    find_ids,
    iter_ids,
    pretty,
    sh_read,
    sh_read_program,
)
from sash.render import program_to_datum


def test_iter_ids_reaches_nested_sites() -> None:
    src = "cat <<E\n$a\nE\necho $((b + 1)) $(echo $c)\n"
    syms = {i.sym for i in iter_ids(sh_read(src))}
    assert {"a", "b", "c"} <= syms


def test_find_ids_filters_by_kind_and_sym() -> None:
    program = sh_read("x=1\necho $x | cat")
    assert [i.sym for i in find_ids(program, kind=IdKind.VARIABLE_BINDER)] == ["x"]
    assert [i.kind for i in find_ids(program, sym="|")] == [IdKind.OPERATOR]
    assert find_ids(program, kind=IdKind.OPERATOR, sym="cat") == []


def test_keywords_are_traversed() -> None:
    program = sh_read("if a; then b; fi")
    syms = [i.sym for i in iter_ids(program)]
    assert "if" in syms and "then" in syms and "fi" in syms


def test_pretty_smoke() -> None:
    assert pretty(sh_read("echo hi")) == (
        "(program\n  (simple (word (lit 'echo')) (word (lit 'hi'))))"
    )


def test_datum_ids_without_scopes() -> None:
    datum = program_to_datum(sh_read("a | b"))
    assert datum == (
        "program",
        (
            "pipeline",
            ("simple", ("word", ("lit", "'a'"))),
            ("id", "|", "operator"),
            ("simple", ("word", ("lit", "'b'"))),
        ),
    )


_META_SRC = """\
declare -A m
m[k]=v
a=([1]=x [k2]=y z)
b+=(1 2)
f() { local n=$1; echo "hi $n" >&2; }
function g { :; }
for ((i=0; i<3; i++)); do :; done
select s in p q; do break; done
if [[ -n ${a[0]} && ( $b == p* || 2 -gt 1 ) && ! x =~ y ]]; then :; fi
case $v in one) ;; two) ;& three) ;;& *) : ;; esac
(( i += 2 ))
coproc CP { cat; }
time ! cat <(sort f) >(tee g) &>> log {fd}>out 3>&1- <<< "here $x"
echo $'a\\tb' $"loc" ${!ind} ${!pre@} ${#a[@]} ${v:1:2} ${v/p/q} ${v^^} {r,s}{1..5..2} $[i+1]
cat <<-EOF
\tbody $w
EOF
x=1 y+=2 cmd | tr a b && { :; } || (exit 1) &
"""


def _collect_ids(obj: object, seen: set[int], out: dict[int, ShId]) -> None:
    if id(obj) in seen:
        return
    seen.add(id(obj))
    if isinstance(obj, ShId):
        out[id(obj)] = obj
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for field in dataclasses.fields(obj):
            _collect_ids(getattr(obj, field.name), seen, out)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            _collect_ids(item, seen, out)


def test_iter_ids_meta_coverage() -> None:
    """Every ShId reachable by a full-field walk is yielded by iter_ids."""
    program = sh_read(_META_SRC, dialect=Dialect.BASH)
    walked: dict[int, ShId] = {}
    _collect_ids(program, set(), walked)
    yielded = {id(i) for i in iter_ids(program)}
    missing = sorted(walked[k].sym for k in set(walked) - yielded)
    assert not missing
    assert yielded <= set(walked)


def test_meta_src_datum_and_pretty_run() -> None:
    program, _store = sh_read_program(_META_SRC, dialect=Dialect.BASH)
    text = pretty(program, scopes=True)
    for tag in ("select", "for-arith", "coproc", "cond", "proc-sub", "fd-var"):
        assert tag in text
    assert isinstance(program_to_datum(program, scopes=True), tuple)


def test_datum_ids_with_scope_names() -> None:
    program, _store = sh_read_program("x=1")
    datum = program_to_datum(program, scopes=True)
    assert datum == (
        "program",
        (
            "simple",
            (
                "assign",
                ("id", "x", "variable-binder", ("core", "top")),
                ("word", ("lit", "'1'")),
            ),
        ),
    )
