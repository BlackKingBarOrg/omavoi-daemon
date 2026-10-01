"""The mode list says, per step, whether its instructions are somebody's own.

The console's "no rewrite" removes a mode's one step. A shipped prompt is what
the next step it adds will carry anyway, so only an edited one is worth asking
about before it goes.
"""
from __future__ import annotations

import argparse
import json

from omavoi import config
from omavoi.commands import modes
from omavoi.window import Window


def _mode(*rest: str) -> int:
    return modes.cmd_mode(argparse.Namespace(action="step", rest=list(rest),
                                             json=False, force=False))


def _steps(monkeypatch, capsys, name: str) -> list[dict]:
    monkeypatch.setattr(modes.ipc, "ping", lambda: None)
    monkeypatch.setattr("omavoi.window.active_window", lambda: Window())
    capsys.readouterr()
    assert modes.cmd_mode(argparse.Namespace(action="list", rest=[], json=True)) == 0
    rows = {r["name"]: r for r in json.loads(capsys.readouterr().out)["modes"]}
    return rows[name]["steps"]


def _plain_default(home):
    cfg = config.defaults()
    cfg["modes"]["default"]["steps"] = []
    config.write(cfg)


def test_a_step_added_with_the_shipped_prompt_is_not_edited(home, monkeypatch, capsys):
    _plain_default(home)
    assert _mode("default", "add", "local") == 0
    [step] = _steps(monkeypatch, capsys, "default")
    assert step["llm"] == "local"
    assert step["prompt_edited"] is False


def test_instructions_somebody_wrote_are_edited(home, monkeypatch, capsys):
    _plain_default(home)
    assert _mode("default", "add", "local") == 0
    assert _mode("default", "prompt", "0", "Fix the grammar and nothing else.") == 0
    [step] = _steps(monkeypatch, capsys, "default")
    assert step["prompt_edited"] is True
    # Back to the shipped one: nothing of theirs is left to lose.
    assert _mode("default", "prompt", "0") == 0
    [step] = _steps(monkeypatch, capsys, "default")
    assert step["prompt_edited"] is False


def test_a_shipped_prompt_in_another_language_is_not_edited(home, monkeypatch, capsys):
    """Written while the console was in Chinese and read with it in English:
    still the shipped text, and still nobody's own."""
    cfg = config.defaults()
    cfg["ui"]["language"] = "en"
    cfg["modes"]["default"]["steps"] = [
        {"llm": "local", "prompt": config.default_step_prompt("zh")}]
    config.write(cfg)
    [step] = _steps(monkeypatch, capsys, "default")
    assert step["prompt_edited"] is False


def test_every_other_step_field_is_still_listed(home, monkeypatch, capsys):
    _plain_default(home)
    assert _mode("default", "add", "local") == 0
    [step] = _steps(monkeypatch, capsys, "default")
    assert {"llm", "model", "prompt", "timeout", "prompt_edited"} <= set(step)
