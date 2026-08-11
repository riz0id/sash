"""Claude Code PreToolUse hook: decide Bash permissions with sash.

Parses the proposed command with sash and emits an "allow" decision when
every subcommand of the chain is a literal-only simple command whose text
matches a Bash(...) allow rule in Claude settings. If any subcommand matches
a deny rule, emits an explicit "deny" decision whose reason aggregates the
advice from every matching deny rule across the chain — a denied stage
anywhere in a &&/||/| chain denies the whole command. Anything else —
parse errors, expansions, redirects, unlisted commands — stays silent so the
normal permission flow runs. The sole redirect exception is `2>&1`: merging
stderr into stdout touches no files, so it cannot widen what an allow rule
already permits.

Beyond Claude Code's native rule forms (exact, `prefix:*`, `prefix *`),
rules containing other glob metacharacters are matched with fnmatch, e.g.
`Bash(sed -n [0-9]*,[0-9]*p*)`. Glob rules are additionally matched against
the stage's dequoted argv text, so quoting cannot dodge them, while the
pattern occurring inside another command's argument does not match.

The optional `permissions.denyAdvice` settings map keys the full Bash(...)
deny rule to remediation advice appended to that rule's denial reason.
"""

from __future__ import annotations

import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sash import (
    AndOr,
    Command,
    Dialect,
    DQuote,
    Escape,
    Lit,
    Pipeline,
    Program,
    Redirect,
    ShParseError,
    Simple,
    SQuote,
    Word,
    sh_read,
)

LITERAL_PARTS = (Lit, Escape, SQuote)


def load_rules(project_dir: str | None) -> tuple[list[str], list[str], dict[str, str]]:
    paths = [Path.home() / ".claude" / "settings.json"]
    if project_dir:
        paths.append(Path(project_dir) / ".claude" / "settings.json")
        paths.append(Path(project_dir) / ".claude" / "settings.local.json")
    allow: list[str] = []
    deny: list[str] = []
    advice: dict[str, str] = {}
    for path in paths:
        try:
            perms = json.loads(path.read_text()).get("permissions", {})
        except (OSError, ValueError):
            continue
        for key, out in (("allow", allow), ("deny", deny)):
            for rule in perms.get(key, []):
                if (
                    isinstance(rule, str)
                    and rule.startswith("Bash(")
                    and rule.endswith(")")
                ):
                    out.append(rule[len("Bash(") : -1])
        deny_advice = perms.get("denyAdvice", {})
        if isinstance(deny_advice, dict):
            for rule, text in deny_advice.items():
                if (
                    isinstance(rule, str)
                    and isinstance(text, str)
                    and rule.startswith("Bash(")
                    and rule.endswith(")")
                ):
                    advice[rule[len("Bash(") : -1]] = text
    return allow, deny, advice


def redirect_is_stderr_merge(redirect: Redirect) -> bool:
    """True iff the redirect is exactly `2>&1`."""
    return (
        redirect.n == 2
        and redirect.op_id.sym == ">&"
        and not redirect.move
        and redirect.fd_var is None
        and isinstance(redirect.target, Word)
        and len(redirect.target.parts) == 1
        and isinstance(redirect.target.parts[0], Lit)
        and redirect.target.parts[0].text == "1"
    )


def word_is_literal(word: Word) -> bool:
    for part in word.parts:
        if isinstance(part, LITERAL_PARTS):
            continue
        if isinstance(part, DQuote):
            if all(isinstance(p, (Lit, Escape)) for p in part.parts):
                continue
        return False
    return True


def stage_text(simple: Simple, source: str) -> str:
    first, last = simple.words[0].loc, simple.words[-1].loc
    return source[first.pos : last.pos + last.span]


def word_text(word: Word) -> str:
    """Dequoted text of a literal-only word: the argv string it executes as."""
    out: list[str] = []
    for part in word.parts:
        if isinstance(part, (Lit, SQuote)):
            out.append(part.text)
        elif isinstance(part, Escape):
            out.append(part.ch)
        elif isinstance(part, DQuote):
            out.extend(p.text if isinstance(p, Lit) else p.ch for p in part.parts)
    return "".join(out)


def stage_argv_text(simple: Simple) -> str:
    return " ".join(word_text(w) for w in simple.words)


def rule_matches(rule: str, text: str, argv_text: str) -> bool:
    texts = (text, " ".join(text.split()))
    if rule.endswith(":*"):
        prefix = rule[:-2]
        return any(t == prefix or t.startswith(prefix + " ") for t in texts)
    if rule.endswith(" *"):
        prefix = rule[:-2]
        return any(t.startswith(prefix + " ") for t in texts)
    if any(ch in rule for ch in "*?["):
        return any(fnmatch.fnmatchcase(t, rule) for t in (*texts, argv_text))
    return any(t == rule for t in texts)


def collect_stages(cmd: Command) -> list[Simple] | None:
    """Flatten a command into its simple-command stages, None if unsupported."""
    if isinstance(cmd, Simple):
        return [cmd]
    if isinstance(cmd, Pipeline):
        if cmd.bang_id is not None or cmd.time_id is not None:
            return None
        stages: list[Simple] = []
        for c in cmd.cmds:
            got = collect_stages(c)
            if got is None:
                return None
            stages.extend(got)
        return stages
    if isinstance(cmd, AndOr):
        left = collect_stages(cmd.left)
        right = collect_stages(cmd.right)
        if left is None or right is None:
            return None
        return left + right
    return None


def decision(kind: str, reason: str) -> dict[str, Any]:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": kind,
            "permissionDecisionReason": reason,
        }
    }


def decide(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("tool_name") != "Bash":
        return None
    command = payload.get("tool_input", {}).get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    allow, deny, deny_advice = load_rules(
        os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd")
    )
    if not allow and not deny:
        return None
    try:
        program: Program = sh_read(command, dialect=Dialect.BASH)
    except ShParseError:
        return None
    stages: list[Simple] = []
    for cmd in program.commands:
        got = collect_stages(cmd)
        if got is None:
            return None
        stages.extend(got)
    if not stages:
        return None
    denials: list[str] = []
    for stage in stages:
        if not stage.words or not all(word_is_literal(w) for w in stage.words):
            continue
        text = stage_text(stage, command)
        argv_text = stage_argv_text(stage)
        for rule in deny:
            if rule_matches(rule, text, argv_text):
                advice = f"subcommand '{text}' matches deny rule Bash({rule})"
                extra = deny_advice.get(rule)
                if extra:
                    advice += f": {extra}"
                if advice not in denials:
                    denials.append(advice)
    if denials:
        return decision(
            "deny",
            "sash: "
            + "; ".join(denials)
            + "; every command in a chain must be permitted",
        )
    for stage in stages:
        if stage.assigns or not stage.words:
            return None
        if not all(redirect_is_stderr_merge(r) for r in stage.redirects):
            return None
        if not all(word_is_literal(w) for w in stage.words):
            return None
        text = stage_text(stage, command)
        argv_text = stage_argv_text(stage)
        if not any(rule_matches(r, text, argv_text) for r in allow):
            return None
    return decision(
        "allow",
        "sash: all commands allowlisted; no unquoted "
        "operators/expansions/redirects beyond 2>&1",
    )


def main() -> None:
    try:
        result = decide(json.load(sys.stdin))
    except Exception:
        result = None
    if result is not None:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
