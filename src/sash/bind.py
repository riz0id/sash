"""The scope/binding pass: allocate scopes, paint identifiers, register binders.

Faithful-to-sh scope model. Only constructs that genuinely create variable
contexts introduce scopes: the top level, function bodies (a home for
`local`-style binders), subshells, and command substitutions (copy-on-fork).
Loops, `if`, and brace groups introduce none.

The crucial faithfulness point: a plain assignment registers its binder with
the *fork-context* scope set (core + top + enclosing subshell/command-sub
scopes, excluding function scopes), because `x=v` inside a function body is
global in sh. A reference inside the function carries the function scope as
well, so the global binder's set is a subset and resolution succeeds; a
`local` binder registered *with* the function scope is a strict superset and
shadows the global inside while remaining invisible outside. Subshell
assignments carry the subshell scope, so copy-on-fork isolation falls out of
subset resolution for free.

Documented approximations: sh is dynamically scoped; scope sets model the
lexical/textual structure — definition-site contexts, fork boundaries,
shadowing — which is the statically decidable layer. Call-graph effects
(`local` in f affecting callees of f, call-order-dependent visibility) are
semantic residue no static machinery can decide. The pass is likewise
flow-insensitive: a reference before a later same-context assignment
resolves to that cell; whether the cell is *set yet* at the reference is a
dataflow question answerable from the binding's site locations. An array is
likewise one variable cell: `a[i]=v` and `a=(...)` bind `a` itself, with
subscripts as expressions inside the binder or reference, so which element
a subscript names at runtime is semantic residue too. A `declare -n` nameref
is followed to its target only when the initializer is a single unquoted
literal name; dynamic targets stay opaque.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .arith import iter_arith, parse_arith
from .nodes import (
    AndOr,
    Arith,
    ArithCommand,
    ArithExpr,
    ArithPart,
    ArithVar,
    Assign,
    Async,
    BraceExpand,
    BraceGroup,
    Case,
    CmdSub,
    Command,
    COND_BINARY_OPS,
    COND_UNARY_OPS,
    CondAnd,
    CondBinary,
    CondCmd,
    CondExpr,
    CondGroup,
    CondNot,
    CondOr,
    CondUnary,
    Coproc,
    Dialect,
    DQuote,
    For,
    ForArith,
    FunDef,
    HereDoc,
    IdKind,
    If,
    Lit,
    Param,
    Pipeline,
    ProcSub,
    Program,
    Redirect,
    RESERVED_WORDS,
    BASH_RESERVED_WORDS,
    Select,
    ShId,
    ShParseError,
    Simple,
    SQuote,
    Subshell,
    Until,
    While,
    Word,
    WordPart,
    is_valid_name,
    matching_bracket,
    word_single_unquoted_lit,
)
from .scopes import (
    BindingKind,
    BindingStore,
    Namespace,
    Scope,
    ScopeKind,
    Site,
    paint,
    resolve,
)

CONTROL_OPERATORS = (
    frozenset(
        [
            "|",
            "||",
            "&&",
            "&",
            ";",
            ";;",
            "<",
            ">",
            ">>",
            "<<",
            "<<-",
            "<&",
            ">&",
            "<>",
            ">|",
            "(",
            ")",
            "{",
            "}",
            "!",
            ";&",
            ";;&",
            "((",
            "))",
        ]
    )
    | RESERVED_WORDS
    | BASH_RESERVED_WORDS
    | COND_UNARY_OPS
    | COND_BINARY_OPS
    | frozenset(["]]"])
)

SPECIAL_BUILTINS = frozenset(
    [
        "break",
        ":",
        ".",
        "continue",
        "eval",
        "exec",
        "exit",
        "export",
        "readonly",
        "return",
        "set",
        "shift",
        "times",
        "trap",
        "unset",
    ]
)

BASH_SPECIAL_VARS = frozenset(
    [
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
)

_POSITIONAL_SYMS = [str(n) for n in range(1, 10)] + ["@", "*", "#"]

_NAME_PREFIX_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

_MAPFILE_ARG_FLAGS = frozenset("dnOsuCc")


@dataclass(frozen=True)
class _Ctx:
    painted: frozenset[Scope]  # everything to paint onto identifiers here
    fork: frozenset[Scope]  # core + top + enclosing fork boundaries, no fn scopes
    fn_scope: Scope | None  # innermost function scope
    positional_base: frozenset[Scope]  # painted set truncated at the fn boundary


def sh_bind(
    program: Program, *, dialect: Dialect = Dialect.POSIX, recognize_local: bool = False
) -> BindingStore:
    store = BindingStore()
    core = frozenset({store.core})
    for sym in CONTROL_OPERATORS:
        store.add_binding(
            Namespace.OPERATOR, sym, core, BindingKind.OPERATOR, Site("implicit", None)
        )
    for name in SPECIAL_BUILTINS:
        store.add_binding(
            Namespace.FUNCTION,
            name,
            core,
            BindingKind.SPECIAL_BUILTIN,
            Site("implicit", None),
        )
    if dialect is Dialect.BASH:
        for name in BASH_SPECIAL_VARS:
            store.add_binding(
                Namespace.VARIABLE,
                name,
                core,
                BindingKind.SPECIAL_PARAM,
                Site("implicit", None),
            )
    base = frozenset({store.core, store.top})
    ctx = _Ctx(painted=base, fork=base, fn_scope=None, positional_base=base)
    binder = _Binder(store, recognize_local or dialect is Dialect.BASH, dialect)
    binder.walk_body(program.commands, ctx)
    return store


class _Binder:
    def __init__(
        self, store: BindingStore, recognize_local: bool, dialect: Dialect
    ) -> None:
        self.store = store
        self.recognize_local = recognize_local
        self.dialect = dialect

    # -- commands ------------------------------------------------------------

    def walk_body(self, commands: list[Command], ctx: _Ctx) -> None:
        for cmd in commands:
            self.walk_command(cmd, ctx)

    def walk_command(self, cmd: Command, ctx: _Ctx) -> None:
        if isinstance(cmd, Simple):
            self._walk_simple(cmd, ctx)
        elif isinstance(cmd, Pipeline):
            if cmd.time_id is not None:
                paint(cmd.time_id, ctx.painted)
            if cmd.bang_id is not None:
                paint(cmd.bang_id, ctx.painted)
            for pid in cmd.pipe_ids:
                paint(pid, ctx.painted)
            self.walk_body(cmd.cmds, ctx)
        elif isinstance(cmd, AndOr):
            paint(cmd.op_id, ctx.painted)
            self.walk_command(cmd.left, ctx)
            self.walk_command(cmd.right, ctx)
        elif isinstance(cmd, Async):
            paint(cmd.amp_id, ctx.painted)
            self.walk_command(cmd.cmd, ctx)
        elif isinstance(cmd, Subshell):
            self._paint_keywords(cmd.keywords, ctx)
            sub = self.store.new_scope(ScopeKind.SUBSHELL, f"subshell@{cmd.loc.line}")
            inner = _Ctx(
                painted=ctx.painted | {sub},
                fork=ctx.fork | {sub},
                fn_scope=ctx.fn_scope,
                positional_base=ctx.positional_base,
            )
            self.walk_body(cmd.body, inner)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, BraceGroup):
            self._paint_keywords(cmd.keywords, ctx)
            self.walk_body(cmd.body, ctx)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, If):
            self._paint_keywords(cmd.keywords, ctx)
            for test, body in cmd.clauses:
                self.walk_body(test, ctx)
                self.walk_body(body, ctx)
            if cmd.else_body is not None:
                self.walk_body(cmd.else_body, ctx)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, (While, Until)):
            self._paint_keywords(cmd.keywords, ctx)
            self.walk_body(cmd.test, ctx)
            self.walk_body(cmd.body, ctx)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, (For, Select)):
            self._paint_keywords(cmd.keywords, ctx)
            paint(cmd.var_id, ctx.painted)
            # for-loops create no scope; the variable persists after the loop
            self.store.add_binding(
                Namespace.VARIABLE,
                cmd.var_id.sym,
                ctx.fork,
                BindingKind.FOR,
                Site("binder", cmd.var_id),
            )
            if cmd.in_words is not None:
                for word in cmd.in_words:
                    self._walk_parts(word.parts, ctx)
            else:
                self._ensure_positional("@", IdKind.SPECIAL_PARAM, ctx)  # implicit "$@"
            self.walk_body(cmd.body, ctx)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, ForArith):
            self._paint_keywords(cmd.keywords, ctx)
            for expr in (cmd.init, cmd.cond, cmd.step):
                if expr is not None:
                    self._walk_arith(expr, ctx)
            self.walk_body(cmd.body, ctx)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, ArithCommand):
            self._paint_keywords(cmd.keywords, ctx)
            self._walk_arith(cmd.expr, ctx)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, CondCmd):
            self._paint_keywords(cmd.keywords, ctx)
            self._walk_cond(cmd.expr, ctx)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, Coproc):
            self._paint_keywords(cmd.keywords, ctx)
            if cmd.name_id is not None:
                # the coprocess array cell; the unnamed default COPROC is not
                # registered (runtime concern)
                paint(cmd.name_id, ctx.painted)
                self.store.add_binding(
                    Namespace.VARIABLE,
                    cmd.name_id.sym,
                    ctx.fork,
                    BindingKind.ASSIGNMENT,
                    Site("binder", cmd.name_id),
                )
            self.walk_command(cmd.cmd, ctx)
        elif isinstance(cmd, Case):
            self._paint_keywords(cmd.keywords, ctx)
            self._walk_parts(cmd.subject.parts, ctx)
            for item in cmd.items:
                self._paint_keywords(item.keywords, ctx)
                for pattern in item.patterns:
                    self._walk_parts(pattern.parts, ctx)
                self.walk_body(item.body, ctx)
            self._walk_redirects(cmd.redirects, ctx)
        elif isinstance(cmd, FunDef):
            self._paint_keywords(cmd.keywords, ctx)
            paint(cmd.name_id, ctx.painted)
            # a function defined in a subshell is local to that subshell
            self.store.add_binding(
                Namespace.FUNCTION,
                cmd.name_id.sym,
                ctx.fork,
                BindingKind.FUNCTION,
                Site("binder", cmd.name_id),
            )
            fn = self.store.new_scope(ScopeKind.FUNCTION, f"function:{cmd.name_id.sym}")
            inner = _Ctx(
                painted=ctx.painted | {fn},
                fork=ctx.fork,  # function scopes are not fork boundaries
                fn_scope=fn,
                positional_base=ctx.painted | {fn},  # positionals rebind per call
            )
            self.walk_command(cmd.body, inner)
            self._walk_redirects(cmd.redirects, ctx)

    def _paint_keywords(self, keywords: list[ShId], ctx: _Ctx) -> None:
        for kw in keywords:
            paint(kw, ctx.painted)

    # -- [[ ]] conditional expressions ----------------------------------------

    def _walk_cond(self, expr: CondExpr, ctx: _Ctx) -> None:
        if isinstance(expr, Word):
            self._walk_parts(expr.parts, ctx)
        elif isinstance(expr, CondUnary):
            paint(expr.op_id, ctx.painted)
            self._walk_parts(expr.operand.parts, ctx)
        elif isinstance(expr, CondBinary):
            paint(expr.op_id, ctx.painted)
            self._walk_parts(expr.left.parts, ctx)
            self._walk_parts(expr.right.parts, ctx)
            if expr.op_id.sym == "=~":
                # a successful match writes BASH_REMATCH in the fork context
                self._add_implicit("BASH_REMATCH", ctx.fork, BindingKind.SPECIAL_PARAM)
        elif isinstance(expr, CondNot):
            paint(expr.bang_id, ctx.painted)
            self._walk_cond(expr.operand, ctx)
        elif isinstance(expr, (CondAnd, CondOr)):
            paint(expr.op_id, ctx.painted)
            self._walk_cond(expr.left, ctx)
            self._walk_cond(expr.right, ctx)
        else:
            paint(expr.lparen_id, ctx.painted)
            paint(expr.rparen_id, ctx.painted)
            self._walk_cond(expr.inner, ctx)

    # -- simple commands -----------------------------------------------------

    def _walk_simple(self, cmd: Simple, ctx: _Ctx) -> None:
        if cmd.cmd_id is not None:
            paint(cmd.cmd_id, ctx.painted)
        for assign in cmd.assigns:
            self._register_assignment(assign, ctx, BindingKind.ASSIGNMENT)
        for word in cmd.words:
            self._walk_parts(word.parts, ctx)
        self._walk_redirects(cmd.redirects, ctx)
        name = cmd.cmd_id.sym if cmd.cmd_id is not None else None
        bash = self.dialect is Dialect.BASH
        if name in ("export", "readonly"):
            kind = BindingKind.EXPORT if name == "export" else BindingKind.READONLY
            operands = cmd.words[1:]
            flags, _ = _leading_flags(operands)
            if bash and "f" in flags:
                self._register_operand_binders(
                    operands, ctx, kind, ctx.fork, ns=Namespace.FUNCTION
                )
            else:
                self._register_operand_binders(operands, ctx, kind, ctx.fork)
        elif name == "unset":
            self._register_unsets(cmd.words[1:], ctx)
        elif name == "local" and self.recognize_local and ctx.fn_scope is not None:
            self._register_operand_binders(
                cmd.words[1:], ctx, BindingKind.LOCAL, ctx.fork | {ctx.fn_scope}
            )
        elif name == "set" and cmd.cmd_id is not None:
            self._register_set_positionals(cmd.cmd_id, cmd.words[1:], ctx)
        elif name in ("declare", "typeset") and bash:
            self._walk_declare(cmd.words[1:], ctx)
        elif name == "read":
            self._register_operand_binders(
                cmd.words[1:], ctx, BindingKind.ASSIGNMENT, ctx.fork
            )
        elif name == "getopts" and len(cmd.words) >= 3:
            self._register_operand_binders(
                [cmd.words[2]], ctx, BindingKind.ASSIGNMENT, ctx.fork
            )
        elif name == "printf" and bash:
            self._register_printf_v(cmd.words[1:], ctx)
        elif name in ("mapfile", "readarray") and bash:
            self._register_mapfile(cmd.words[1:], ctx)
        elif name == "let" and bash:
            self._walk_let(cmd.words[1:], ctx)

    def _register_assignment(
        self, assign: Assign, ctx: _Ctx, kind: BindingKind
    ) -> None:
        paint(assign.name_id, ctx.painted)
        self.store.add_binding(
            Namespace.VARIABLE,
            assign.name_id.sym,
            ctx.fork,
            kind,
            Site("binder", assign.name_id),
        )
        if assign.subscript is not None:
            self._walk_arith(assign.subscript, ctx)
        if assign.array_items is not None:
            for item in assign.array_items:
                if item.subscript is not None:
                    self._walk_arith(item.subscript, ctx)
                self._walk_parts(item.word.parts, ctx)
        self._walk_parts(assign.word.parts, ctx)

    def _register_operand_binders(
        self,
        operands: list[Word],
        ctx: _Ctx,
        kind: BindingKind,
        scopes: frozenset[Scope],
        ns: Namespace = Namespace.VARIABLE,
    ) -> None:
        for word in operands:
            named = _operand_name(word, bash=self.dialect is Dialect.BASH)
            if named is None:
                continue
            name, name_id = named
            if ns is Namespace.FUNCTION:
                name_id.kind = IdKind.FUNCTION_NAME
            paint(name_id, ctx.painted)
            self.store.add_binding(ns, name, scopes, kind, Site("binder", name_id))

    def _walk_declare(self, operands: list[Word], ctx: _Ctx) -> None:
        flags, start = _leading_flags(operands)
        # flag priority (last-wins is unnecessary): n > r > x > local/assignment
        if "n" in flags:
            kind = BindingKind.NAMEREF
        elif "r" in flags:
            kind = BindingKind.READONLY
        elif "x" in flags:
            kind = BindingKind.EXPORT
        elif ctx.fn_scope is not None and "g" not in flags:
            kind = BindingKind.LOCAL
        else:
            kind = BindingKind.ASSIGNMENT
        if ctx.fn_scope is not None and "g" not in flags:
            scopes = ctx.fork | {ctx.fn_scope}
        else:
            scopes = ctx.fork
        for word in operands[start:]:
            named = _operand_name(word, bash=True)
            if named is None:
                continue
            name, name_id = named
            paint(name_id, ctx.painted)
            binding = self.store.add_binding(
                Namespace.VARIABLE, name, scopes, kind, Site("binder", name_id)
            )
            if kind is BindingKind.NAMEREF:
                target = _nameref_target(word)
                if target is not None:
                    paint(target, ctx.painted)
                    if binding.nameref_target is None:
                        binding.nameref_target = target

    def _register_printf_v(self, operands: list[Word], ctx: _Ctx) -> None:
        for i, word in enumerate(operands):
            if word_single_unquoted_lit(word) == "-v" and i + 1 < len(operands):
                self._register_operand_binders(
                    [operands[i + 1]], ctx, BindingKind.ASSIGNMENT, ctx.fork
                )
                return

    def _register_mapfile(self, operands: list[Word], ctx: _Ctx) -> None:
        i = 0
        while i < len(operands):
            text = word_single_unquoted_lit(operands[i])
            if (
                text is not None
                and len(text) > 1
                and text[0] == "-"
                and text[1:].isalpha()
            ):
                i += 2 if text[-1] in _MAPFILE_ARG_FLAGS else 1
                continue
            break
        if i < len(operands):
            self._register_operand_binders(
                [operands[i]], ctx, BindingKind.ASSIGNMENT, ctx.fork
            )

    def _walk_let(self, operands: list[Word], ctx: _Ctx) -> None:
        for word in operands:
            text = _let_operand_text(word)
            if text is None:
                continue
            try:
                expr = parse_arith(text, word.loc, dialect=self.dialect)
            except ShParseError:
                continue  # malformed let operand is a runtime concern
            self._walk_arith(expr, ctx)

    def _register_unsets(self, operands: list[Word], ctx: _Ctx) -> None:
        ns = Namespace.VARIABLE
        for word in operands:
            text = word_single_unquoted_lit(word)
            if text is None:
                continue
            if text.startswith("-"):
                ns = Namespace.FUNCTION if "f" in text else Namespace.VARIABLE
                continue
            if not is_valid_name(text):
                continue
            kind = (
                IdKind.FUNCTION_NAME
                if ns is Namespace.FUNCTION
                else IdKind.VARIABLE_REF
            )
            uid = ShId(text, kind, word.loc)
            paint(uid, ctx.painted)
            binding = resolve(uid, ns, ambiguous_ok=True)
            if binding is not None:
                binding.sites.append(Site("unset", uid))

    def _register_set_positionals(
        self, set_id: ShId, operands: list[Word], ctx: _Ctx
    ) -> None:
        saw_terminator = False
        has_operands = False
        for word in operands:
            text = word_single_unquoted_lit(word)
            if not saw_terminator and text == "--":
                saw_terminator = True
                has_operands = True  # `set --` alone still resets positionals
                continue
            if not saw_terminator and text is not None and text.startswith("-"):
                continue  # option word, e.g. `set -e`
            has_operands = True
        if not has_operands:
            return
        for sym in _POSITIONAL_SYMS:
            kind = (
                BindingKind.POSITIONAL if sym.isdigit() else BindingKind.SPECIAL_PARAM
            )
            self.store.add_binding(
                Namespace.VARIABLE,
                sym,
                ctx.positional_base,
                kind,
                Site("binder", set_id),
            )

    # -- words and parts -----------------------------------------------------

    def _walk_redirects(self, redirects: list[Redirect], ctx: _Ctx) -> None:
        for redirect in redirects:
            paint(redirect.op_id, ctx.painted)
            if redirect.fd_var is not None:
                # {var}>file writes the allocated fd into var, like x=v
                paint(redirect.fd_var, ctx.painted)
                self.store.add_binding(
                    Namespace.VARIABLE,
                    redirect.fd_var.sym,
                    ctx.fork,
                    BindingKind.ASSIGNMENT,
                    Site("binder", redirect.fd_var),
                )
            if isinstance(redirect.target, HereDoc):
                hd = redirect.target
                paint(hd.op_id, ctx.painted)
                if hd.body is not None:
                    self._walk_parts(hd.body, ctx)
            else:
                self._walk_parts(redirect.target.parts, ctx)

    def _walk_parts(self, parts: list[WordPart], ctx: _Ctx) -> None:
        for part in parts:
            if isinstance(part, Param):
                self._walk_param(part, ctx)
            elif isinstance(part, DQuote):
                self._walk_parts(list(part.parts), ctx)
            elif isinstance(part, CmdSub):
                self._walk_cmdsub(part, ctx)
            elif isinstance(part, ProcSub):
                self._walk_procsub(part, ctx)
            elif isinstance(part, Arith):
                self._walk_arith(part.expr, ctx)
            elif isinstance(part, BraceExpand):
                if part.alternates is not None:
                    for alt in part.alternates:
                        self._walk_parts(alt, ctx)

    def _walk_arith(self, expr: ArithExpr, ctx: _Ctx) -> None:
        for node in iter_arith(expr):
            if isinstance(node, ArithVar):
                paint(node.id, ctx.painted)
                if node.id.kind is IdKind.VARIABLE_BINDER:
                    # $((x=5)) and ++/-- write in the fork context, like x=v
                    self.store.add_binding(
                        Namespace.VARIABLE,
                        node.id.sym,
                        ctx.fork,
                        BindingKind.ASSIGNMENT,
                        Site("binder", node.id),
                    )
            elif isinstance(node, ArithPart):
                self._walk_parts([node.part], ctx)

    def _walk_param(self, param: Param, ctx: _Ctx) -> None:
        paint(param.id, ctx.painted)
        if param.id.kind in (IdKind.POSITIONAL, IdKind.SPECIAL_PARAM):
            self._ensure_positional(param.id.sym, param.id.kind, ctx)
        elif param.op in (":=", "="):
            # ${x:=v} assigns x as a side effect, in the fork context
            self.store.add_binding(
                Namespace.VARIABLE,
                param.id.sym,
                ctx.fork,
                BindingKind.ASSIGNMENT,
                Site("binder", param.id),
            )
        if param.subscript is not None and not isinstance(param.subscript, Lit):
            self._walk_arith(param.subscript, ctx)
        if param.pattern is not None:
            self._walk_parts(param.pattern, ctx)
        if param.word is not None:
            self._walk_parts(param.word, ctx)
        if param.offset is not None:
            self._walk_arith(param.offset, ctx)
        if param.length is not None:
            self._walk_arith(param.length, ctx)

    def _walk_cmdsub(self, part: CmdSub, ctx: _Ctx) -> None:
        if part.program is None:
            return
        sub = self.store.new_scope(ScopeKind.COMMAND_SUB, f"cmdsub@{part.loc.line}")
        inner = _Ctx(
            painted=ctx.painted | {sub},
            fork=ctx.fork | {sub},
            fn_scope=ctx.fn_scope,
            positional_base=ctx.positional_base,
        )
        self.walk_body(part.program.commands, inner)

    def _walk_procsub(self, part: ProcSub, ctx: _Ctx) -> None:
        if part.program is None:
            return
        # a proc-sub body runs in a subshell: same fork treatment as $(...)
        sub = self.store.new_scope(ScopeKind.PROC_SUB, f"procsub@{part.loc.line}")
        inner = _Ctx(
            painted=ctx.painted | {sub},
            fork=ctx.fork | {sub},
            fn_scope=ctx.fn_scope,
            positional_base=ctx.positional_base,
        )
        self.walk_body(part.program.commands, inner)

    # -- positional / special parameters -------------------------------------

    def _ensure_positional(self, sym: str, kind: IdKind, ctx: _Ctx) -> None:
        bkind = (
            BindingKind.POSITIONAL
            if kind is IdKind.POSITIONAL
            else BindingKind.SPECIAL_PARAM
        )
        self._add_implicit(sym, ctx.positional_base, bkind)

    def _add_implicit(
        self, sym: str, scopes: frozenset[Scope], kind: BindingKind
    ) -> None:
        # Register only on first reference in this context: an implicit cell
        # models "the positionals of this function/top-level", not a write.
        max_scope = max(scopes, key=lambda sc: sc.id)
        existing = max_scope.table.get(Namespace.VARIABLE, {}).get(sym, {})
        if scopes in existing:
            return
        self.store.add_binding(
            Namespace.VARIABLE, sym, scopes, kind, Site("implicit", None)
        )


def _leading_flags(operands: list[Word]) -> tuple[str, int]:
    """Letters of leading `-abc` flag words and the index of the first operand.

    `+x` flag words are consumed but contribute no letters (they unset
    attributes); `--` ends flag parsing.
    """
    letters = ""
    i = 0
    while i < len(operands):
        text = word_single_unquoted_lit(operands[i])
        if text == "--":
            return (letters, i + 1)
        if (
            text is None
            or len(text) < 2
            or text[0] not in "-+"
            or not text[1:].isalpha()
        ):
            break
        if text[0] == "-":
            letters += text[1:]
        i += 1
    return (letters, i)


def _nameref_target(word: Word) -> ShId | None:
    """The `target` of a `ref=target` operand when it is a static plain name."""
    text = word_single_unquoted_lit(word)
    if text is None:
        return None
    _, eq, value = text.partition("=")
    if eq and is_valid_name(value):
        return ShId(value, IdKind.VARIABLE_REF, word.loc)
    return None


def _let_operand_text(word: Word) -> str | None:
    """A let operand's text: unquoted literal or wholly quoted literal text."""
    text = word_single_unquoted_lit(word)
    if text is not None:
        return text
    if len(word.parts) == 1:
        part = word.parts[0]
        if isinstance(part, SQuote):
            return part.text
        if isinstance(part, DQuote) and all(isinstance(p, Lit) for p in part.parts):
            return "".join(p.text for p in part.parts if isinstance(p, Lit))
    return None


def _operand_name(word: Word, *, bash: bool = False) -> tuple[str, ShId] | None:
    """NAME or NAME=... operand of export/readonly/local: (name, minted binder id)."""
    text = word_single_unquoted_lit(word)
    if text is not None and is_valid_name(text):
        return (text, ShId(text, IdKind.VARIABLE_BINDER, word.loc))
    if not word.parts or not isinstance(word.parts[0], Lit):
        return None
    head = word.parts[0].text
    if bash:
        m = _NAME_PREFIX_RE.match(head)
        if m is None:
            return None
        name = m.group(0)
        i = m.end()
        if head.startswith("[", i):
            end = matching_bracket(head, i)
            if end is None:
                return None
            i = end + 1
        if head.startswith("+=", i) or head.startswith("=", i):
            return (name, ShId(name, IdKind.VARIABLE_BINDER, word.parts[0].loc))
        return None
    eq = head.find("=")
    if eq <= 0:
        return None
    name = head[:eq]
    if not is_valid_name(name):
        return None
    return (name, ShId(name, IdKind.VARIABLE_BINDER, word.parts[0].loc))
