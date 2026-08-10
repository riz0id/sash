"""Claude Code PreToolUse hook: allow decisions and permission fall-through."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any, Protocol

import pytest

_HOOK_PATH = Path(__file__).resolve().parent.parent / "tools" / "permission_hook.py"
_spec = importlib.util.spec_from_file_location("permission_hook", _HOOK_PATH)
assert _spec is not None and _spec.loader is not None
hook = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hook)

ALLOW = ["Bash(rg)", "Bash(git status)", "Bash(ls:*)", "Bash(cat *)"]
DENY = ["Bash(grep *)", "Bash(ls -R /)"]


class Decide(Protocol):
    def __call__(
        self, command: str, tool_name: str = "Bash"
    ) -> dict[str, Any] | None: ...


@pytest.fixture
def decide(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Decide:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    (project / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": ALLOW, "deny": DENY}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))

    def run(command: str, tool_name: str = "Bash") -> dict[str, Any] | None:
        result: dict[str, Any] | None = hook.decide(
            {"tool_name": tool_name, "tool_input": {"command": command}}
        )
        return result

    return run


def assert_allowed(decision: dict[str, Any] | None) -> None:
    assert decision is not None
    out = decision["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "allow"


def assert_denied(decision: dict[str, Any] | None, rule: str) -> None:
    assert decision is not None
    out = decision["hookSpecificOutput"]
    assert out["hookEventName"] == "PreToolUse"
    assert out["permissionDecision"] == "deny"
    assert f"Bash({rule})" in out["permissionDecisionReason"]


# --- allow: literal-only commands covered by allow rules ---


def test_allows_exact_rule(decide: Decide) -> None:
    assert_allowed(decide("rg"))


def test_allows_multiword_exact_rule(decide: Decide) -> None:
    assert_allowed(decide("git status"))


def test_allows_colon_star_prefix_rule(decide: Decide) -> None:
    assert_allowed(decide("ls -la src"))


def test_allows_colon_star_rule_bare_command(decide: Decide) -> None:
    assert_allowed(decide("ls"))


def test_allows_space_star_prefix_rule(decide: Decide) -> None:
    assert_allowed(decide("cat foo.txt"))


def test_allows_single_quoted_literal_args(decide: Decide) -> None:
    assert_allowed(decide("ls 'a b'"))


def test_allows_double_quoted_literal_args(decide: Decide) -> None:
    assert_allowed(decide('ls "a b"'))


def test_allows_pipeline_of_allowlisted_stages(decide: Decide) -> None:
    assert_allowed(decide("ls -l | rg"))


def test_allows_and_or_chain(decide: Decide) -> None:
    assert_allowed(decide("ls -a && rg || git status"))


def test_allows_extra_whitespace_via_normalization(decide: Decide) -> None:
    assert_allowed(decide("ls   -la    src"))


# --- deny: any subcommand matching a deny rule denies the whole command ---


def test_deny_rule_blocks_allow(decide: Decide) -> None:
    assert_denied(decide("grep foo"), "grep *")


def test_deny_exact_rule_overrides_allow_prefix(decide: Decide) -> None:
    assert_denied(decide("ls -R /"), "ls -R /")


def test_deny_stage_blocks_whole_pipeline(decide: Decide) -> None:
    assert_denied(decide("ls | grep foo"), "grep *")


def test_deny_stage_blocks_and_chain(decide: Decide) -> None:
    assert_denied(decide("ls && grep foo"), "grep *")


def test_deny_stage_blocks_or_chain(decide: Decide) -> None:
    assert_denied(decide("grep foo || ls"), "grep *")


def test_deny_wins_over_unlisted_stage_in_chain(decide: Decide) -> None:
    assert_denied(decide("whoami && grep foo"), "grep *")


# --- fall-through: non-literals and unprovable commands ---


def test_unlisted_command_falls_through(decide: Decide) -> None:
    assert decide("whoami") is None


def test_space_star_rule_requires_argument(decide: Decide) -> None:
    assert decide("cat") is None


def test_variable_expansion_falls_through(decide: Decide) -> None:
    assert decide("ls $HOME") is None


def test_expansion_in_double_quotes_falls_through(decide: Decide) -> None:
    assert decide('ls "$HOME"') is None


def test_command_substitution_falls_through(decide: Decide) -> None:
    assert decide("ls $(whoami)") is None


def test_redirect_falls_through(decide: Decide) -> None:
    assert decide("ls > /tmp/out") is None


def test_assignment_prefix_falls_through(decide: Decide) -> None:
    assert decide("FOO=1 ls") is None


def test_bang_pipeline_falls_through(decide: Decide) -> None:
    assert decide("! ls") is None


def test_unlisted_stage_blocks_pipeline(decide: Decide) -> None:
    assert decide("ls | whoami") is None


def test_parse_error_falls_through(decide: Decide) -> None:
    assert decide("ls (") is None


def test_empty_command_falls_through(decide: Decide) -> None:
    assert decide("   ") is None


def test_non_bash_tool_falls_through(decide: Decide) -> None:
    assert decide("rg", tool_name="Read") is None


def test_no_allow_rules_falls_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", lambda: home)
    project = tmp_path / "project"
    (project / ".claude").mkdir(parents=True)
    (project / ".claude" / "settings.json").write_text(
        json.dumps({"permissions": {"allow": [], "deny": []}}), encoding="utf-8"
    )
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", str(project))
    payload = {"tool_name": "Bash", "tool_input": {"command": "ls"}}
    assert hook.decide(payload) is None
