"""`omavoi mode set` reaches the fields a two-line chat answer needs.

`rules.joiner = keep` and `newline_key = SHIFT+RETURN` are what let an LLM's
two lines survive to a chat window without sending the first. The first was
reachable only through `omavoi config set modes.<m>.rules.joiner`, which
nothing pointed at; the second is new. Both go through `mode set` now, with
the same refusals as the other fields.
"""
from __future__ import annotations

import argparse

from omavoi import config
from omavoi.commands.modes import cmd_mode


def _run(*rest: str) -> int:
    return cmd_mode(argparse.Namespace(action="set", rest=list(rest), json=False, force=False))


def test_rules_joiner_keep_is_written(home, capsys):
    assert _run("prose", "rules.joiner", "keep") == 0
    assert config.load()["modes"]["prose"]["rules"]["joiner"] == "keep"
    assert "prose.rules.joiner set" in capsys.readouterr().out


def test_rules_flag_takes_on_off_and_refuses_else(home, capsys):
    assert _run("prose", "rules.fillers", "off") == 0
    assert config.load()["modes"]["prose"]["rules"]["fillers"] is False
    assert _run("prose", "rules.fillers", "maybe") == 1
    assert "on or off" in capsys.readouterr().err


def test_rules_punctuation_is_keep_or_strip(home, capsys):
    assert _run("prose", "rules.punctuation", "strip") == 0
    assert config.load()["modes"]["prose"]["rules"]["punctuation"] == "strip"
    assert _run("prose", "rules.punctuation", "loud") == 1


def test_unknown_rule_is_refused_with_the_list(home, capsys):
    assert _run("prose", "rules.nonsense", "x") == 1
    err = capsys.readouterr().err
    assert "unknown rule 'nonsense'" in err and "joiner" in err


def test_newline_key_is_a_field(home, capsys):
    assert _run("prose", "newline_key", "SHIFT+RETURN") == 0
    assert config.load()["modes"]["prose"]["newline_key"] == "SHIFT+RETURN"


def test_unknown_field_names_the_rules_form(home, capsys):
    assert _run("prose", "bogus", "x") == 1
    assert "rules.<" in capsys.readouterr().err


def test_other_rules_untouched_when_one_is_set(home):
    before = dict(config.load()["modes"]["prose"].get("rules") or config.load()["modes"]["default"]["rules"])
    _run("prose", "rules.joiner", "keep")
    after = config.load()["modes"]["prose"]["rules"]
    for k, v in before.items():
        if k != "joiner":
            assert after.get(k, v) == v
