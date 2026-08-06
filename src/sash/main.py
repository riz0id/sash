"""CLI: parse a shell script, bind it, and report every identifier."""

from __future__ import annotations

import sys

from . import namespace_for, resolve_identifier, sh_read_program
from .nodes import Dialect, ShParseError
from .render import iter_ids, pretty
from .scopes import ShAmbiguityError


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in ("-h", "--help"):
        print("usage: sash [--bash] [file]", file=sys.stderr)
        raise SystemExit(0)
    dialect = Dialect.POSIX
    if "--bash" in args:
        args.remove("--bash")
        dialect = Dialect.BASH
    if args:
        src = args[0]
        with open(src, encoding="utf-8") as handle:
            text = handle.read()
    else:
        src = "<stdin>"
        text = sys.stdin.read()

    try:
        program, _store = sh_read_program(text, src, dialect=dialect)
    except ShParseError as err:
        print(f"sash: {err}", file=sys.stderr)
        raise SystemExit(1) from None

    print(pretty(program, scopes=True))
    print()
    print("identifiers:")
    for sh_id in iter_ids(program):
        loc = sh_id.loc
        where = f"{loc.line}:{loc.col}"
        ns = namespace_for(sh_id)
        try:
            binding = resolve_identifier(sh_id, ns)
        except ShAmbiguityError:
            desc = "ambiguous"
        else:
            if binding is None:
                desc = "unbound"
            else:
                n = len(binding.sites)
                desc = f"{binding.kind.value} ({n} site{'s' if n != 1 else ''})"
        print(f"  {where:>8}  {sh_id.kind.value:16} {sh_id.sym!r:12} -> {desc}")


if __name__ == "__main__":
    main()
