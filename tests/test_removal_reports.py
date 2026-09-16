"""Removing something that was never there is not a success.

`omavoi names rm nosuchname` printed "removed 0" in green and exited 0, so a
typo read exactly like the work being done. `omavoi dict rm` on the same kind
of word list names the one it could not find and exits 1. Two commands over
two lists of words in one config file, answering the same question two ways.

A partial removal is its own case: `names rm Alice nosuchname` should remove
Alice — refusing the whole batch over one typo is worse — say which one it
could not find, and exit non-zero, because something the caller asked for did
not happen.
"""
from __future__ import annotations

import argparse

from omavoi import config
from omavoi.commands import words


def _names(cfg=None):
    cfg = cfg or config.load()
    return [e.get("name") if isinstance(e, dict) else e
            for e in cfg.get("dictionary", {}).get("names", [])]


def _run(action, names):
    return words.cmd_names(argparse.Namespace(
        action=action, names=names, group="", json=False, apply=False))


def test_removing_a_name_that_is_not_there_fails(home, capsys):
    assert _run("add", ["Alice"]) == 0
    capsys.readouterr()

    assert _run("rm", ["nosuchname"]) == 1, "a name that was never there"
    out = capsys.readouterr().out
    assert "nosuchname" in out, "it must say which one"
    assert "removed" not in out, 'nothing was removed; do not print "removed 0"'
    assert _names() == ["Alice"], "it removed something it should not have"


def test_a_partial_removal_does_the_work_and_still_reports(home, capsys):
    assert _run("add", ["Alice", "Bob"]) == 0
    capsys.readouterr()

    assert _run("rm", ["Alice", "nosuchname"]) == 1
    out = capsys.readouterr().out
    assert "removed 1" in out, "Alice should have gone"
    assert "nosuchname" in out, "and the miss should be named"
    assert _names() == ["Bob"]


def test_removing_a_name_that_is_there_succeeds(home, capsys):
    assert _run("add", ["Alice"]) == 0
    capsys.readouterr()
    assert _run("rm", ["Alice"]) == 0
    assert _names() == []


def test_rm_with_no_names_is_usage(home, capsys):
    assert _run("rm", []) == 1
    assert "usage" in capsys.readouterr().err


def test_a_missing_file_is_not_ffmpegs_fault(home, tmp_path):
    """`transcribe /nonexistent.wav` reported ffmpeg's own diagnostics.

    `Error opening input: No such file or directory`, under a line of ffmpeg
    banner — an answer about ffmpeg to a question about a filename.
    """
    import pytest

    from omavoi.commands._support import load_wav

    with pytest.raises(SystemExit) as exc:
        load_wav(tmp_path / "gone.wav")
    assert "no such file" in str(exc.value)
    assert "ffmpeg" not in str(exc.value)

    with pytest.raises(SystemExit) as exc:
        load_wav(tmp_path)
    assert "is a directory" in str(exc.value)
