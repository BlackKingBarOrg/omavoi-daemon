"""The hotkey report must not call a dead key bound.

Twice now the daemon has held a HotkeyListener that opened no device — the
start path did not drop it on failure, the way the rebind path always had —
and status() then reported enabled:true with an empty device list. `omavoi
hotkey check` read that and answered "the daemon is listening on it:" over a
key that could not be read at all, which is the one thing that command exists
not to do.
"""
from __future__ import annotations

import glob

from omavoi.commands import keys
from omavoi.hotkey import explain_missing, key_code


def _report(monkeypatch, hotkey_block, **over):
    monkeypatch.setattr(keys, "config", keys.config)
    # Patched on ipc, which is where the client half lives and where
    # keys.py reads it; daemon.ping is a re-export of this one.
    from omavoi import ipc
    monkeypatch.setattr(ipc, "ping", lambda: {"hotkey": hotkey_block})
    return keys._hotkey_report()


def test_a_listener_with_no_devices_is_not_listening(monkeypatch, home):
    r = _report(monkeypatch, {"enabled": True, "key": "RIGHTALT", "devices": []})
    assert r["listener"] is False, "an empty device list is not a binding"
    assert r["bound_devices"] == []


def test_the_configured_key_is_not_read_as_a_binding(monkeypatch, home):
    """status() falls back to the configured key for display when there is no
    listener; the report must not take that for a live binding."""
    r = _report(monkeypatch, {"enabled": False, "key": "RIGHTALT", "devices": []})
    assert r["bound"] == "", "the config value was read as if it were live"
    assert r["matches"] is False


def test_a_real_binding_is_reported_as_one(monkeypatch, home):
    r = _report(monkeypatch, {"enabled": True, "key": "RIGHTALT",
                              "devices": ["/dev/input/event5 A Keyboard"]})
    assert r["listener"] is True
    assert r["bound"] == "RIGHTALT"


def test_unreadable_devices_are_not_reported_as_absent():
    """evdev.list_devices() drops what it cannot open, so building the message
    on it said "there are no input devices at all" about a machine with
    twenty-five of them."""
    nodes = glob.glob("/dev/input/event*")
    if not nodes:
        return  # a machine with genuinely none; nothing to distinguish
    why = explain_missing(key_code("RIGHTALT"), "RIGHTALT")
    assert "no input devices at all" not in why, (
        f"{len(nodes)} device nodes exist and the message denies it: {why!r}")


def test_the_json_exit_status_answers_the_same_question_as_the_text(monkeypatch, home):
    """`hotkey check --json` exited 1 on a hotkey that was working.

    It was `matches and not devices_problem`, and devices_problem describes
    this process's own access to /dev/input — which is absent on every
    machine where the input group was granted after login and the daemon was
    started with it. So the text output said "the daemon is reading it on:
    /dev/input/event7, /dev/input/event6" and returned 0, and --json returned
    1 about the same daemon reading the same two devices.

    The sixth instance of one mistake: reporting what the checking process can
    do instead of what the running one is doing.
    """
    import argparse
    import json

    from omavoi.commands import keys as keys_mod

    working = {
        "configured": "RIGHTALT", "mode": "push_to_talk", "enabled": True,
        "code": 100, "name_ok": True,
        "group_listed": True, "group_held": False,
        # Set, because this shell cannot open a device — and irrelevant,
        # because the daemon can.
        "devices_problem": "you are in the `input` group but this process "
                           "started before that took effect",
        "daemon": "running", "bound": "RIGHTALT",
        "bound_devices": ["/dev/input/event7 SONiX USB DEVICE"],
        "listener": True, "matches": True,
    }
    monkeypatch.setattr(keys_mod, "_hotkey_report", lambda: working)
    args = argparse.Namespace(action="check", json=True, timeout=10)
    assert keys_mod.cmd_hotkey(args) == 0, (
        "a daemon reading the configured key must exit 0"
    )

    # And it must still fail when the daemon is reading nothing.
    for broken in ({**working, "listener": False},
                   {**working, "matches": False, "bound": "RIGHTCTRL"}):
        monkeypatch.setattr(keys_mod, "_hotkey_report", lambda b=broken: b)
        assert keys_mod.cmd_hotkey(args) == 1, json.dumps(broken)
