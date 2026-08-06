"""Regression checks for bugs found during development.

The lexer once looped forever at end of input: `_peek()` returns "" at EOF
and, in Python, `"" in " \\t"` is True (the empty string is a substring of
everything), so blank-skipping loops spun without advancing. Every test here
runs under a deadline so a reintroduced non-termination fails instead of
hanging the suite.
"""

from __future__ import annotations

import signal
from collections.abc import Iterator
from contextlib import contextmanager
from types import FrameType

import pytest

from sash import ShParseError, sh_lex, sh_read
from sash.nodes import TokEof


@contextmanager
def deadline(seconds: int = 5) -> Iterator[None]:
    def on_alarm(signum: int, frame: FrameType | None) -> None:
        raise TimeoutError("lexer/parser did not terminate")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def test_lexing_terminates_at_eof_after_word() -> None:
    with deadline():
        toks = sh_lex("echo hi")
    assert isinstance(toks[-1], TokEof)


def test_lexing_terminates_on_empty_and_blank_input() -> None:
    with deadline():
        for src in ("", " ", "\t", " \t ", "\n", "\\\n"):
            toks = sh_lex(src)
            assert isinstance(toks[-1], TokEof)


def test_param_op_scan_at_eof_does_not_hang() -> None:
    # `_read_param_op` peeks past `${x`; "" must not satisfy `in "-=?+"`.
    for src in ("echo ${x", "echo ${x:", "echo ${x:-", "echo ${x%%"):
        with deadline():
            with pytest.raises(ShParseError):
                sh_read(src)


def test_heredoc_operator_at_eof_inside_cmdsub_does_not_hang() -> None:
    # The blank-skip after `<<` inside `$(...)` raw capture hit the same
    # empty-string-membership trap.
    for src in ("echo $(cat <<", "echo $(cat << ", "echo $(cat <<E"):
        with deadline():
            with pytest.raises(ShParseError):
                sh_read(src)


def test_case_pattern_paren_does_not_end_cmdsub_capture() -> None:
    with deadline():
        program = sh_read("echo $(case a in b) echo c ;; esac) tail")
    assert program.commands


def test_empty_program_parses() -> None:
    with deadline():
        program = sh_read("")
    assert program.commands == []


def test_comment_only_program_parses() -> None:
    with deadline():
        program = sh_read("# just a comment\n")
    assert program.commands == []
