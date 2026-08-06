from sash import (
    BindingKind,
    Dialect,
    IdKind,
    Namespace,
    find_ids,
    program_to_datum,
    resolve,
    sh_read,
    sh_read_program,
)

SRC = "f() { local x; x=1; echo $x; }"


def test_bash_dialect_parses_identically() -> None:
    posix = sh_read("echo hi")
    bash = sh_read("echo hi", dialect=Dialect.BASH)
    assert program_to_datum(bash) == program_to_datum(posix)


def test_bash_dialect_implies_recognize_local() -> None:
    program, _store = sh_read_program(SRC, dialect=Dialect.BASH)
    (ref,) = find_ids(program, kind=IdKind.VARIABLE_REF, sym="x")
    binding = resolve(ref, Namespace.VARIABLE)
    assert binding is not None
    assert binding.kind is BindingKind.LOCAL


def test_posix_default_local_not_recognized() -> None:
    program, _store = sh_read_program(SRC)
    (ref,) = find_ids(program, kind=IdKind.VARIABLE_REF, sym="x")
    binding = resolve(ref, Namespace.VARIABLE)
    assert binding is not None
    assert binding.kind is BindingKind.ASSIGNMENT
