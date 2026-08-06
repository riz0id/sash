"""Stage 7 binding extensions: declare/typeset, namerefs, binder-operand
builtins, {var}> redirects, coproc names, and bash special variables."""

from __future__ import annotations

import pytest

from sash import (
    Binding,
    BindingKind,
    BindingStore,
    Dialect,
    IdKind,
    Namespace,
    Program,
    ShId,
    find_ids,
    identifier_binding,
    resolve,
    sh_read_program,
)

BASH_SPECIALS = [
    "RANDOM",
    "SECONDS",
    "LINENO",
    "FUNCNAME",
    "BASH_SOURCE",
    "BASH_REMATCH",
    "PIPESTATUS",
    "REPLY",
    "OPTARG",
    "OPTIND",
]


def bash(src: str) -> Program:
    program, _store = sh_read_program(src, dialect=Dialect.BASH)
    return program


def posix(src: str) -> Program:
    program, _store = sh_read_program(src)
    return program


def ref(program: Program, sym: str, index: int = 0) -> ShId:
    return find_ids(program, kind=IdKind.VARIABLE_REF, sym=sym)[index]


def ref_binding(program: Program, sym: str, index: int = 0) -> Binding:
    binding = resolve(ref(program, sym, index))
    assert binding is not None
    return binding


# -- declare / typeset --------------------------------------------------------


def test_declare_toplevel_is_assignment() -> None:
    program = bash("declare x=1\necho $x")
    assert ref_binding(program, "x").kind is BindingKind.ASSIGNMENT


def test_declare_in_function_is_local() -> None:
    program = bash("f() { declare x=1; echo $x; }\necho $x")
    assert ref_binding(program, "x", 0).kind is BindingKind.LOCAL
    assert resolve(ref(program, "x", 1)) is None


def test_typeset_in_function_is_local() -> None:
    program = bash("f() { typeset y=2; echo $y; }\necho $y")
    assert ref_binding(program, "y", 0).kind is BindingKind.LOCAL
    assert resolve(ref(program, "y", 1)) is None


def test_declare_g_escapes_function() -> None:
    program = bash("f() { declare -g x=1; }\necho $x")
    assert ref_binding(program, "x").kind is BindingKind.ASSIGNMENT


def test_declare_r_is_readonly() -> None:
    program = bash("declare -r A=1\necho $A")
    assert ref_binding(program, "A").kind is BindingKind.READONLY


def test_declare_x_is_export() -> None:
    program = bash("declare -x B=2\necho $B")
    assert ref_binding(program, "B").kind is BindingKind.EXPORT


def test_declare_array_shares_the_cell_with_assignment() -> None:
    program = bash("declare -a arr\narr[0]=1\necho ${arr[0]}")
    binding = ref_binding(program, "arr")
    assert binding.kind is BindingKind.ASSIGNMENT
    assert len(binding.sites) == 2


def test_declare_flags_after_double_dash_are_operands() -> None:
    program = bash("declare -- x=1\necho $x")
    assert ref_binding(program, "x").kind is BindingKind.ASSIGNMENT


def test_posix_declare_is_not_special() -> None:
    program = posix("declare x=1\necho $x")
    assert resolve(ref(program, "x")) is None


# -- namerefs -----------------------------------------------------------------


def test_declare_n_is_nameref() -> None:
    program = bash("declare -n ref=target\ntarget=1\necho $ref")
    binding = ref_binding(program, "ref")
    assert binding.kind is BindingKind.NAMEREF
    assert binding.nameref_target is not None


def test_identifier_binding_follows_static_nameref() -> None:
    program = bash("declare -n ref=target\ntarget=1\necho $ref")
    info = identifier_binding(ref(program, "ref"))
    assert info is not None
    kind, sym, _scopes, sites = info
    assert kind == "assignment"
    assert sym == "target"
    assert len(sites) == 1


def test_identifier_binding_follows_nameref_through_indirection() -> None:
    program = bash("declare -n r=t\nt=5\necho ${!r}")
    info = identifier_binding(ref(program, "r"))
    assert info is not None
    assert info[0] == "assignment"
    assert info[1] == "t"


def test_dynamic_nameref_stays_opaque() -> None:
    program = bash("declare -n ref=$1\necho $ref")
    info = identifier_binding(ref(program, "ref"))
    assert info is not None
    assert info[0] == "nameref"
    assert info[1] == "ref"


def test_nameref_to_unbound_target_reports_the_nameref() -> None:
    program = bash("declare -n ref=nothing\necho $ref")
    info = identifier_binding(ref(program, "ref"))
    assert info is not None
    assert info[0] == "nameref"


# -- binder-operand builtins --------------------------------------------------


@pytest.mark.parametrize("dialect", [Dialect.POSIX, Dialect.BASH])
def test_read_operands_bind(dialect: Dialect) -> None:
    program, _store = sh_read_program("read x y\necho $x $y", dialect=dialect)
    assert ref_binding(program, "x").kind is BindingKind.ASSIGNMENT
    assert ref_binding(program, "y").kind is BindingKind.ASSIGNMENT


def test_read_skips_flag_words() -> None:
    program = bash("read -r x\necho $x")
    assert ref_binding(program, "x").kind is BindingKind.ASSIGNMENT


def test_read_without_operands_uses_reply_special() -> None:
    program = bash("read\necho $REPLY")
    assert ref_binding(program, "REPLY").kind is BindingKind.SPECIAL_PARAM


@pytest.mark.parametrize("dialect", [Dialect.POSIX, Dialect.BASH])
def test_getopts_variable_binds(dialect: Dialect) -> None:
    program, _store = sh_read_program("getopts ab: opt\necho $opt", dialect=dialect)
    assert ref_binding(program, "opt").kind is BindingKind.ASSIGNMENT


def test_printf_v_binds_in_bash() -> None:
    program = bash("printf -v out %s hi\necho $out")
    assert ref_binding(program, "out").kind is BindingKind.ASSIGNMENT


def test_printf_v_does_not_bind_in_posix() -> None:
    program = posix("printf -v out %s hi\necho $out")
    assert resolve(ref(program, "out")) is None


def test_mapfile_binds_first_non_flag_operand() -> None:
    program = bash("mapfile -t arr\necho ${arr[0]}")
    assert ref_binding(program, "arr").kind is BindingKind.ASSIGNMENT


def test_mapfile_skips_flag_arguments() -> None:
    program = bash("mapfile -t -d , arr\necho ${arr[0]}")
    assert ref_binding(program, "arr").kind is BindingKind.ASSIGNMENT


def test_readarray_binds() -> None:
    program = bash("readarray lines\necho $lines")
    assert ref_binding(program, "lines").kind is BindingKind.ASSIGNMENT


def test_let_quoted_expression_binds() -> None:
    program = bash('j=1\nlet "i = j + 1"\necho $i')
    assert ref_binding(program, "i").kind is BindingKind.ASSIGNMENT


def test_let_unquoted_increment_binds() -> None:
    program = bash("let x++\necho $x")
    assert ref_binding(program, "x").kind is BindingKind.ASSIGNMENT


def test_let_non_literal_operand_is_skipped() -> None:
    program = bash("let $expr\necho ok")
    assert find_ids(program, sym="expr")


def test_export_f_registers_in_function_namespace() -> None:
    program = bash("f() { :; }\nexport -f f\nf")
    (cmd_ref,) = find_ids(program, kind=IdKind.COMMAND, sym="f")
    binding = resolve(cmd_ref, Namespace.FUNCTION)
    assert binding is not None
    assert len(binding.sites) == 2


def test_export_f_does_not_bind_a_variable() -> None:
    program = bash("f() { :; }\nexport -f f\necho $f")
    assert resolve(ref(program, "f")) is None


def test_posix_readonly_f_stays_variable() -> None:
    program = posix("readonly -f x\necho $x")
    assert ref_binding(program, "x").kind is BindingKind.READONLY


# -- {var}> redirects and coproc ----------------------------------------------


def test_fd_var_redirect_binds() -> None:
    program = bash("exec {fd}>out\necho $fd")
    binding = ref_binding(program, "fd")
    assert binding.kind is BindingKind.ASSIGNMENT
    assert binding.sites[0].kind == "binder"


def test_coproc_name_binds() -> None:
    program = bash("coproc mycop { echo hi; }\necho ${mycop[0]}")
    assert ref_binding(program, "mycop").kind is BindingKind.ASSIGNMENT


# -- bash special variables ---------------------------------------------------


@pytest.mark.parametrize("name", BASH_SPECIALS)
def test_bash_special_resolves_in_bash(name: str) -> None:
    program = bash(f"echo ${name}")
    binding = ref_binding(program, name)
    assert binding.kind is BindingKind.SPECIAL_PARAM
    assert binding.sites[0].kind == "implicit"


@pytest.mark.parametrize("name", BASH_SPECIALS)
def test_bash_special_unbound_in_posix(name: str) -> None:
    program = posix(f"echo ${name}")
    assert resolve(ref(program, name)) is None


def test_bash_specials_are_core_registered() -> None:
    _program, store = sh_read_program("echo hi", dialect=Dialect.BASH)
    assert isinstance(store, BindingStore)
    assert "RANDOM" in store.core.table[Namespace.VARIABLE]


def test_rematch_fork_registration_coexists_with_core() -> None:
    program = bash("[[ $x =~ a ]]\necho $BASH_REMATCH")
    binding = ref_binding(program, "BASH_REMATCH")
    assert binding.kind is BindingKind.SPECIAL_PARAM
    assert len(binding.sites) == 1


def test_rematch_resolves_without_any_match() -> None:
    program = bash("echo $BASH_REMATCH")
    assert ref_binding(program, "BASH_REMATCH").kind is BindingKind.SPECIAL_PARAM


# -- select -------------------------------------------------------------------


def test_select_variable_is_for_binding() -> None:
    program = bash("select x in a b; do :; done\necho $x")
    assert ref_binding(program, "x").kind is BindingKind.FOR
