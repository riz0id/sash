"""Binding-pass semantics: fork boundaries, namespaces, binder recognition."""

from __future__ import annotations

from sash import (
    Binding,
    BindingKind,
    IdKind,
    Namespace,
    Program,
    ShId,
    find_ids,
    free_identifier_eq,
    resolve,
    sh_read_program,
)


def read(src: str, *, recognize_local: bool = False) -> Program:
    program, _store = sh_read_program(src, recognize_local=recognize_local)
    return program


def refs(program: Program, sym: str) -> list[ShId]:
    return find_ids(program, kind=IdKind.VARIABLE_REF, sym=sym)


def must_resolve(sh_id: ShId, ns: Namespace = Namespace.VARIABLE) -> Binding:
    binding = resolve(sh_id, ns)
    assert binding is not None
    return binding


def test_pipe_operator_certified() -> None:
    program = read("echo \"a|b\" 'c|d' | cat")
    pipes = find_ids(program, sym="|")
    assert len(pipes) == 1  # the quoted pipes never become identifiers
    assert pipes[0].kind is IdKind.OPERATOR
    binding = must_resolve(pipes[0], Namespace.OPERATOR)
    assert binding.kind is BindingKind.OPERATOR


def test_assignment_in_function_is_global() -> None:
    program = read("f() { x=1; }\necho $x")
    ref = refs(program, "x")[0]
    binding = must_resolve(ref)
    assert binding.kind is BindingKind.ASSIGNMENT
    assert len(binding.sites) == 1


def test_repeated_assignment_is_one_cell() -> None:
    program = read("x=1\nx=2\necho $x")
    binding = must_resolve(refs(program, "x")[0])
    assert len(binding.sites) == 2


def test_subshell_assignment_shadows_inside_only() -> None:
    program = read("x=1\n(x=2; echo $x)\necho $x")
    inner, outer = refs(program, "x")
    inner_binding = must_resolve(inner)
    outer_binding = must_resolve(outer)
    assert inner_binding is not outer_binding
    assert len(inner_binding.sites) == 1
    assert len(outer_binding.sites) == 1
    assert not free_identifier_eq(inner, outer)


def test_subshell_reads_outer_variable() -> None:
    program = read("x=1\n(echo $x)")
    binding = must_resolve(refs(program, "x")[0])
    assert binding.kind is BindingKind.ASSIGNMENT


def test_cmdsub_assignment_is_isolated() -> None:
    program = read("x=1\necho $(x=2; echo $x)\necho $x")
    inner, outer = refs(program, "x")
    assert not free_identifier_eq(inner, outer)


def test_for_variable_persists_after_loop() -> None:
    program = read("for i in a b; do echo $i; done\necho $i")
    in_loop, after = refs(program, "i")
    binding = must_resolve(in_loop)
    assert binding.kind is BindingKind.FOR
    assert free_identifier_eq(in_loop, after)


def test_function_name_resolves_at_call_site() -> None:
    program = read("f() { :; }\nf")
    calls = find_ids(program, kind=IdKind.COMMAND, sym="f")
    binding = must_resolve(calls[0], Namespace.FUNCTION)
    assert binding.kind is BindingKind.FUNCTION


def test_function_defined_in_subshell_invisible_outside() -> None:
    program = read("(g() { :; })\ng")
    calls = find_ids(program, kind=IdKind.COMMAND, sym="g")
    assert resolve(calls[0], Namespace.FUNCTION) is None


def test_special_builtin_resolves() -> None:
    program = read("eval x")
    calls = find_ids(program, kind=IdKind.COMMAND, sym="eval")
    binding = must_resolve(calls[0], Namespace.FUNCTION)
    assert binding.kind is BindingKind.SPECIAL_BUILTIN


def test_export_operand_is_binder() -> None:
    program = read("export FOO=1\necho $FOO")
    binding = must_resolve(refs(program, "FOO")[0])
    assert binding.kind is BindingKind.EXPORT


def test_readonly_operand_is_binder() -> None:
    program = read("readonly BAR=2\necho $BAR")
    binding = must_resolve(refs(program, "BAR")[0])
    assert binding.kind is BindingKind.READONLY


def test_unset_appends_site_without_removing_cell() -> None:
    program = read("x=1\nunset x\necho $x")
    binding = must_resolve(refs(program, "x")[0])
    assert [site.kind for site in binding.sites] == ["binder", "unset"]


def test_param_default_assignment_operator_binds() -> None:
    program = read("echo ${x:=5}\necho $x")
    binding = must_resolve(refs(program, "x")[-1])
    assert binding.kind is BindingKind.ASSIGNMENT


def test_positional_is_implicit() -> None:
    program = read("echo $1")
    ones = find_ids(program, kind=IdKind.POSITIONAL, sym="1")
    binding = must_resolve(ones[0])
    assert binding.kind is BindingKind.POSITIONAL
    assert binding.sites[0].kind == "implicit"


def test_positionals_rebound_per_function() -> None:
    program = read("echo $1\nf() { echo $1; }")
    top, in_fn = find_ids(program, kind=IdKind.POSITIONAL, sym="1")
    assert not free_identifier_eq(top, in_fn)


def test_special_param_resolves() -> None:
    program = read("echo $?")
    marks = find_ids(program, kind=IdKind.SPECIAL_PARAM, sym="?")
    binding = must_resolve(marks[0])
    assert binding.kind is BindingKind.SPECIAL_PARAM


def test_set_dashdash_binds_positionals() -> None:
    program = read("set -- a b\necho $1")
    ones = find_ids(program, kind=IdKind.POSITIONAL, sym="1")
    binding = must_resolve(ones[0])
    assert binding.kind is BindingKind.POSITIONAL
    assert any(site.kind == "binder" for site in binding.sites)


def test_local_off_by_default() -> None:
    program = read("f() { local x; x=1; echo $x; }")
    binding = must_resolve(refs(program, "x")[0])
    assert binding.kind is BindingKind.ASSIGNMENT


def test_local_shadows_inside_function_only() -> None:
    src = "x=0\nf() { local x; echo $x; }\necho $x"
    program = read(src, recognize_local=True)
    inner, outer = refs(program, "x")
    assert must_resolve(inner).kind is BindingKind.LOCAL
    assert must_resolve(outer).kind is BindingKind.ASSIGNMENT
    assert not free_identifier_eq(inner, outer)


def test_heredoc_ref_binds_to_outer_cell() -> None:
    program = read("x=1\ncat <<EOF\n$x\nEOF\n")
    binding = must_resolve(refs(program, "x")[0])
    assert binding.kind is BindingKind.ASSIGNMENT


def test_arith_ref_binds() -> None:
    program = read("y=3\necho $((y + 1))")
    binding = must_resolve(refs(program, "y")[0])
    assert binding.kind is BindingKind.ASSIGNMENT
