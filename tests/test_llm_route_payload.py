"""The AI-rewrite routes say which one the next take uses, and how fast it is.

Three routes are in use at once by design, one per mode's step, and the cards
said only which modes used each and whether it was "running" — which a coding
agent always is, having no process to stop. Someone on one mode could not tell
from the page which route their dictation went through.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout

from omavoi import config, paths
from omavoi.commands import catalogue


def payload() -> dict:
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert catalogue.cmd_model(argparse.Namespace(action="list", rest=[], json=True, force=False)) == 0
    return json.loads(buf.getvalue())


def _setup(by_window=False):
    cfg = config.defaults()
    cfg["modes"]["prose"]["steps"] = [{"llm": "agent", "prompt": "x"}]
    cfg["modes"]["default"]["steps"] = [{"llm": "local", "prompt": "x"}]
    cfg["switching"] = {"by_window": by_window, "mode": "default"}
    config.write(cfg)


def _history(steps):
    path = paths.history_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for llm, seconds, error in steps:
            fh.write(json.dumps({"ts": 1.0, "steps": [{"llm": llm, "seconds": seconds, "error": error}]}) + "\n")


def test_the_mode_every_take_uses_is_named(home):
    _setup()
    data = payload()
    assert data["current_mode"] == "default" and data["following_window"] is False
    rows = {r["name"]: r for r in data["llm"]}
    assert rows["local"]["used_by"] == ["default"]
    assert rows["agent"]["used_by"] == ["prose"]


def test_no_current_mode_while_it_follows_the_window(home):
    _setup(by_window=True)
    data = payload()
    assert data["current_mode"] == "" and data["following_window"] is True


def test_nothing_is_resident_without_a_daemon(home):
    _setup()
    assert not any(r["live_resident"] for r in payload()["llm"])


def test_each_route_reports_its_recent_speed(home):
    _setup()
    _history([("local", 1.0, ""), ("local", 2.0, ""), ("local", 3.0, ""),
              ("agent", 9.0, ""), ("agent", 30.0, "timed out"),   # failed steps do not count
              ("agent", 7.0, ""), ("local", True, "")])            # nor a value that is not a time
    rows = {r["name"]: r for r in payload()["llm"]}
    assert (rows["local"]["recent_seconds"], rows["local"]["recent_steps"]) == (2.0, 3)
    assert (rows["agent"]["recent_seconds"], rows["agent"]["recent_steps"]) == (8.0, 2)
    assert (rows["api"]["recent_seconds"], rows["api"]["recent_steps"]) == (None, 0)


def test_a_coding_agent_is_never_resident():
    engines = {"llm": [{"name": "agent", "live": True, "pid": 0},
                       {"name": "local", "live": True, "pid": 4242}]}
    assert catalogue._llm_live(engines, "agent")["live_resident"] is False
    assert catalogue._llm_live(engines, "agent")["live_running"] is True
    assert catalogue._llm_live(engines, "local")["live_resident"] is True
