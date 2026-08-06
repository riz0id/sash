"""Whole-script bash smoke test: parse, bind, resolve, and render together."""

from __future__ import annotations

import pytest

from sash import (
    Binding,
    BindingKind,
    Dialect,
    IdKind,
    Program,
    ShId,
    ShParseError,
    find_ids,
    iter_ids,
    pretty,
    program_to_datum,
    resolve,
    sh_read,
    sh_read_program,
)

SCRIPT = """\
#!/usr/bin/env bash
declare -A map
map[key]="v"
function greet { local name=$1; echo "hi $name"; }
for ((i=0; i<3; i++)); do arr+=("$i"); done
if [[ ${arr[0]} =~ ^[0-9]+$ ]]; then echo "${BASH_REMATCH[0]}"; fi
case $x in a) ;; b) ;& c) echo both ;; esac
diff <(sort f1) <(sort f2) &> log
echo $'tab\\there' ${!map[@]} ${arr[@]/0/zero} {a,b}{1..3}
coproc UP { tr a-z A-Z; }
time read -r line <<< "$input"
echo "${UP[0]}" "$line"
"""


def bound() -> Program:
    program, _store = sh_read_program(SCRIPT, dialect=Dialect.BASH)
    return program


def ref(program: Program, sym: str, index: int = 0) -> ShId:
    return find_ids(program, kind=IdKind.VARIABLE_REF, sym=sym)[index]


def ref_binding(program: Program, sym: str, index: int = 0) -> Binding:
    binding = resolve(ref(program, sym, index))
    assert binding is not None
    return binding


def test_parses_and_paints_every_id() -> None:
    program = bound()
    assert all(i.scopes for i in iter_ids(program))


def test_local_name_resolves_local() -> None:
    program = bound()
    assert ref_binding(program, "name").kind is BindingKind.LOCAL


def test_arr_refs_resolve_to_the_append_binder() -> None:
    program = bound()
    binders = find_ids(program, kind=IdKind.VARIABLE_BINDER, sym="arr")
    binding = ref_binding(program, "arr")
    assert binding.kind is BindingKind.ASSIGNMENT
    assert binding.sites[0].id is binders[0]
    assert resolve(ref(program, "arr", 1)) is binding


def test_c_for_cond_i_resolves_to_the_init_binder() -> None:
    program = bound()
    binders = find_ids(program, kind=IdKind.VARIABLE_BINDER, sym="i")
    binding = ref_binding(program, "i")
    assert binding.sites[0].id is binders[0]


def test_map_refs_resolve_to_the_declare_binder() -> None:
    program = bound()
    binding = ref_binding(program, "map")
    assert binding.kind is BindingKind.ASSIGNMENT
    site = binding.sites[0].id
    assert site is not None
    assert site.loc.line == 2
    assert len(binding.sites) == 2


def test_bash_rematch_resolves() -> None:
    program = bound()
    assert ref_binding(program, "BASH_REMATCH").kind is BindingKind.SPECIAL_PARAM


def test_coproc_name_and_read_operand_resolve() -> None:
    program = bound()
    assert ref_binding(program, "UP").kind is BindingKind.ASSIGNMENT
    assert ref_binding(program, "line").kind is BindingKind.ASSIGNMENT


def test_render_covers_the_whole_script() -> None:
    program = bound()
    text = pretty(program, scopes=True)
    for tag in ("for-arith", "cond", "proc-sub", "coproc", "brace", "ansi-c", ";&"):
        assert tag in text
    assert isinstance(program_to_datum(program, scopes=True), tuple)


def test_posix_dialect_rejects_the_script() -> None:
    with pytest.raises(ShParseError):
        sh_read(SCRIPT)
