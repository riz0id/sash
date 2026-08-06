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
dataflow question answerable from the binding's site locations.
"""

from __future__ import annotations

from dataclasses import dataclass

from .nodes import (
    AndOr,
    Arith,
    Assign,
    Async,
    BraceGroup,
    Case,
    CmdSub,
    Command,
    DQuote,
    For,
    FunDef,
    HereDoc,
    IdKind,
    If,
    Lit,
    Param,
    Pipeline,
    Program,
    Redirect,
    RESERVED_WORDS,
    ShId,
    Simple,
    Subshell,
    Until,
    While,
    Word,
    WordPart,
    is_valid_name,
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
        ]
    )
    | RESERVED_WORDS
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

_POSITIONAL_SYMS = [str(n) for n in range(1, 10)] + ["@", "*", "#"]


@dataclass(frozen=True)
class _Ctx:
    painted: frozenset[Scope]  # everything to paint onto identifiers here
    fork: frozenset[Scope]  # core + top + enclosing fork boundaries, no fn scopes
    fn_scope: Scope | None  # innermost function scope
    positional_base: frozenset[Scope]  # painted set truncated at the fn boundary


def sh_bind(program: Program, *, recognize_local: bool = False) -> BindingStore:
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
    base = frozenset({store.core, store.top})
    ctx = _Ctx(painted=base, fork=base, fn_scope=None, positional_base=base)
    binder = _Binder(store, recognize_local)
    binder.walk_body(program.commands, ctx)
    return store


class _Binder:
    def __init__(self, store: BindingStore, recognize_local: bool) -> None:
        self.store = store
        self.recognize_local = recognize_local

    # -- commands ------------------------------------------------------------

    def walk_body(self, commands: list[Command], ctx: _Ctx) -> None:
        for cmd in commands:
            self.walk_command(cmd, ctx)

    def walk_command(self, cmd: Command, ctx: _Ctx) -> None:
        if isinstance(cmd, Simple):
            self._walk_simple(cmd, ctx)
        elif isinstance(cmd, Pipeline):
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
        elif isinstance(cmd, For):
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
        if name in ("export", "readonly"):
            kind = BindingKind.EXPORT if name == "export" else BindingKind.READONLY
            self._register_operand_binders(cmd.words[1:], ctx, kind, ctx.fork)
        elif name == "unset":
            self._register_unsets(cmd.words[1:], ctx)
        elif name == "local" and self.recognize_local and ctx.fn_scope is not None:
            self._register_operand_binders(
                cmd.words[1:], ctx, BindingKind.LOCAL, ctx.fork | {ctx.fn_scope}
            )
        elif name == "set" and cmd.cmd_id is not None:
            self._register_set_positionals(cmd.cmd_id, cmd.words[1:], ctx)

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
        self._walk_parts(assign.word.parts, ctx)

    def _register_operand_binders(
        self,
        operands: list[Word],
        ctx: _Ctx,
        kind: BindingKind,
        scopes: frozenset[Scope],
    ) -> None:
        for word in operands:
            named = _operand_name(word)
            if named is None:
                continue
            name, name_id = named
            paint(name_id, ctx.painted)
            self.store.add_binding(
                Namespace.VARIABLE, name, scopes, kind, Site("binder", name_id)
            )

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
            elif isinstance(part, Arith):
                for sub in part.parts:
                    if isinstance(sub, ShId):
                        paint(sub, ctx.painted)
                    elif isinstance(sub, Lit):
                        pass
                    else:
                        self._walk_parts([sub], ctx)

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
        if param.word is not None:
            self._walk_parts(param.word, ctx)

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


def _operand_name(word: Word) -> tuple[str, ShId] | None:
    """NAME or NAME=... operand of export/readonly/local: (name, minted binder id)."""
    text = word_single_unquoted_lit(word)
    if text is not None and is_valid_name(text):
        return (text, ShId(text, IdKind.VARIABLE_BINDER, word.loc))
    if not word.parts or not isinstance(word.parts[0], Lit):
        return None
    head = word.parts[0].text
    eq = head.find("=")
    if eq <= 0:
        return None
    name = head[:eq]
    if not is_valid_name(name):
        return None
    return (name, ShId(name, IdKind.VARIABLE_BINDER, word.parts[0].loc))
