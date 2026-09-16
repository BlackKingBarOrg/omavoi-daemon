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
    from omavoi import daemon as daemon_mod
    monkeypatch.setattr(daemon_mod, "ping", lambda: {"hotkey": hotkey_block})
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
