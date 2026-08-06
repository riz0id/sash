"""Claude Code PreToolUse hook: auto-allow Bash commands proven safe by sash.

Parses the proposed command with sash and emits an "allow" decision when
every pipeline stage is a literal-only simple command whose text matches an
existing Bash(...) allow rule in Claude settings. Anything else — parse
errors, expansions, redirects, compound commands, deny-rule matches — stays
silent so the normal permission flow runs.
"""

from __future__ import annotations

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
    ShParseError,
    Simple,
    SQuote,
    Word,
    sh_read,
)

LITERAL_PARTS = (Lit, Escape, SQuote)


def load_rules(project_dir: str | None) -> tuple[list[str], list[str]]:
    paths = [Path.home() / ".claude" / "settings.json"]
    if project_dir:
        paths.append(Path(project_dir) / ".claude" / "settings.json")
        paths.append(Path(project_dir) / ".claude" / "settings.local.json")
    allow: list[str] = []
    deny: list[str] = []
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
    return allow, deny


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


def rule_matches(rule: str, text: str) -> bool:
    texts = (text, " ".join(text.split()))
    if rule.endswith(":*"):
        prefix = rule[:-2]
        return any(t == prefix or t.startswith(prefix + " ") for t in texts)
    if rule.endswith(" *"):
        prefix = rule[:-2]
        return any(t.startswith(prefix + " ") for t in texts)
    if "*" in rule:
        return False
    return any(t == rule for t in texts)


def check(cmd: Command, source: str, allow: list[str], deny: list[str]) -> bool:
    """True iff every stage is a literal-only allowlisted simple command.

    Raises PermissionError when a stage matches a deny rule, aborting the
    whole decision rather than merely failing this branch.
    """
    if isinstance(cmd, Simple):
        if cmd.assigns or cmd.redirects or not cmd.words:
            return False
        if not all(word_is_literal(w) for w in cmd.words):
            return False
        text = stage_text(cmd, source)
        if any(rule_matches(r, text) for r in deny):
            raise PermissionError
        return any(rule_matches(r, text) for r in allow)
    if isinstance(cmd, Pipeline):
        if cmd.bang_id is not None or cmd.time_id is not None:
            return False
        return all(check(c, source, allow, deny) for c in cmd.cmds)
    if isinstance(cmd, AndOr):
        return check(cmd.left, source, allow, deny) and check(
            cmd.right, source, allow, deny
        )
    return False


def decide(payload: dict[str, Any]) -> dict[str, Any] | None:
    if payload.get("tool_name") != "Bash":
        return None
    command = payload.get("tool_input", {}).get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    allow, deny = load_rules(os.environ.get("CLAUDE_PROJECT_DIR") or payload.get("cwd"))
    if not allow:
        return None
    try:
        program: Program = sh_read(command, dialect=Dialect.BASH)
    except ShParseError:
        return None
    try:
        if not program.commands or not all(
            check(c, command, allow, deny) for c in program.commands
        ):
            return None
    except PermissionError:
        return None
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "permissionDecisionReason": (
                "sash: all commands allowlisted; no unquoted "
                "operators/expansions/redirects"
            ),
        }
    }


def main() -> None:
    try:
        decision = decide(json.load(sys.stdin))
    except Exception:
        decision = None
    if decision is not None:
        print(json.dumps(decision))


if __name__ == "__main__":
    main()
