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


# -- the seed budget -------------------------------------------------------


def test_names_past_the_seed_budget_are_reported(home, capsys):
    """seed_text broke at the budget and said nothing about the rest.

    The names are listed on the Dictionary page and by `names list` as
    seeded; the ones past 224 characters were handed to nothing. A user who
    adds forty names gets whatever fits and no sign that the other twelve
    are inert.
    """
    from omavoi import names

    cfg = config.load()
    cfg.setdefault("dictionary", {})["names"] = [
        {"name": f"Name{i:02d}", "seed": True, "enabled": False, "group": ""}
        for i in range(120)
    ]
    index = names.NameIndex(cfg)
    seeded, dropped = index.seed_split()

    # Three tokens each by the estimate, so 120 of them overrun 224.
    assert seeded and dropped, "120 short names should overrun a 224-token budget"
    assert len(seeded) + len(dropped) == 120, "every name must be in one list"
    assert index.seed_chars() <= index.budget, "the seeded names must fit"
    assert set(seeded).isdisjoint(dropped)
    assert index.seed_text() == ", ".join(seeded)


def test_the_budget_is_spent_not_abandoned_at_the_first_overrun(home):
    """It was `break`, so one long name hid every short one behind it.

    Ordered most-used first, a 210-character name with 50 hits stopped the
    loop, and a three-letter name with 10 hits was dropped for no reason
    other than its position.
    """
    from omavoi import names

    cfg = config.load()
    cfg.setdefault("dictionary", {})["names"] = [
        # ~71 tokens each by the estimate, so three fit in 224 and the
        # fourth does not — while Kim, at three tokens, always does.
        *[{"name": letter * 210, "seed": True, "enabled": False,
           "group": "", "hits": 99 - i}
          for i, letter in enumerate("ABCD")],
        {"name": "Kim", "seed": True, "enabled": False, "group": "", "hits": 1},
    ]
    seeded, dropped = names.NameIndex(cfg).seed_split()
    assert "Kim" in seeded, "a short name after a long one still fits the budget"
    assert dropped, "four long names must overrun 224 tokens"
    assert all(len(name) == 210 for name in dropped), dropped


def test_names_list_says_what_did_not_fit(home, capsys):
    cfg = config.load()
    cfg.setdefault("dictionary", {})["names"] = [
        {"name": f"Name{i:02d}", "seed": True, "enabled": False, "group": ""}
        for i in range(120)
    ]
    config.write(cfg)
    capsys.readouterr()

    assert _run("list", []) == 0
    out = capsys.readouterr().out
    assert "did not fit" in out, "the cap must be mentioned"
    assert "seed_budget_tokens" in out, "and how to raise it"
