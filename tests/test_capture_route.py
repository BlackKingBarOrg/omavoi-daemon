"""`hotkey capture` must ask the daemon before it tries the devices itself.

The console's "press a key" button is a child of the desktop session. A
session that joined the `input` group after logging in cannot open a keyboard,
so the button failed on exactly the machines where rebinding was the thing
being attempted — while the daemon, started later or through newgrp, could
read the key perfectly well.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout

import pytest

from omavoi import daemon as daemon_mod
from omavoi import ipc
from omavoi.commands import keys


def _run(monkeypatch, reply, *, raises=None):
    def fake_request(payload, timeout=30.0):
        assert payload["cmd"] == "capture", "it did not ask the daemon"
        if raises is not None:
            raise raises
        return reply
    monkeypatch.setattr(ipc, "request", fake_request)
    args = argparse.Namespace(action="capture", timeout=2.0, json=True)
    buf = io.StringIO()
    with redirect_stdout(buf):
        code = keys.cmd_hotkey(args)
    return code, buf.getvalue().strip()


def test_the_daemons_answer_is_used(monkeypatch, home):
    code, out = _run(monkeypatch, {"ok": True, "key": "RIGHTCTRL"})
    assert code == 0
    assert json.loads(out) == {"ok": True, "key": "RIGHTCTRL"}


def test_a_running_daemon_that_cannot_read_is_the_answer(monkeypatch, home):
    """Not a reason to retry locally with strictly less access."""
    code, out = _run(monkeypatch, {"ok": False, "error": "no key pressed within 2s"})
    assert code == 1
    assert json.loads(out) == {"ok": False, "error": "no key pressed within 2s"}


def test_no_daemon_falls_through_to_the_local_path(monkeypatch, home):
    """Without a daemon there is nothing to ask, so it opens the devices
    itself — which is the only case where that was ever the right order."""
    called = {}

    def fake_capture(timeout=10.0):
        called["timeout"] = timeout
        return "F13"
    # cmd_hotkey imports it inside the function, so patch it at the source.
    from omavoi import hotkey as hotkey_mod
    monkeypatch.setattr(hotkey_mod, "capture", fake_capture)
    code, out = _run(monkeypatch, None,
                     raises=ConnectionError("the omavoi daemon is not running"))
    assert called, "it never reached the local path"
    assert code == 0
    assert json.loads(out) == {"ok": True, "key": "F13"}


@pytest.mark.parametrize("asked,expected", [(0.5, 1.0), (8.0, 8.0), (600.0, 15.0)])
def test_the_daemon_caps_the_timeout(asked, expected):
    """A capture blocks the one socket the daemon serves, so a client must not
    be able to hold it for as long as it likes."""
    import inspect
    src = inspect.getsource(daemon_mod.Daemon.capture_key)
    assert "min(float(timeout), 15.0)" in src
    assert "max(1.0," in src
