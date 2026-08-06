"""CLI entry point, including the --bash dialect flag."""

from __future__ import annotations

from pathlib import Path

import pytest

from sash.main import main


def test_bash_flag_enables_bash_grammar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "s.sh"
    script.write_text("[[ -f x ]]\n", encoding="utf-8")
    main(["--bash", str(script)])
    out = capsys.readouterr().out
    assert "(cond" in out


def test_default_dialect_is_posix(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "s.sh"
    script.write_text("[[ -f x ]]\n", encoding="utf-8")
    main([str(script)])
    out = capsys.readouterr().out
    assert "(simple" in out
    assert "(cond" not in out


def test_posix_mode_rejects_bash_grammar(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "s.sh"
    script.write_text("for ((i=0; i<3; i++)); do :; done\n", encoding="utf-8")
    with pytest.raises(SystemExit) as exc:
        main([str(script)])
    assert exc.value.code == 1
    assert "sash:" in capsys.readouterr().err


def test_bash_flag_after_file_argument(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    script = tmp_path / "s.sh"
    script.write_text("coproc c { :; }\n", encoding="utf-8")
    main([str(script), "--bash"])
    assert "(coproc" in capsys.readouterr().out


def test_help_mentions_bash_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0
    assert "--bash" in capsys.readouterr().err
