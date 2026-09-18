"""A typed newline is the mode's newline_key, not a bare Return.

`xdotool type` and wtype both turn a literal newline into Return. In an
editor that is a newline; in a chat window it is the send key, and a two-line
take -- the original on one line, its translation on the next -- posted its
first line and left the second in the box. The earlier answer folded every
newline away after the LLM, which made the second line impossible rather than
deliverable. Now the injector types line by line and sends the mode's
newline_key between them: RETURN by default, SHIFT+RETURN for a chat mode.
"""
from __future__ import annotations

import shutil
import subprocess
from types import SimpleNamespace

import pytest

from omavoi import inject as inject_mod
from omavoi.inject import Injector, PartiallyTyped


class Ran:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kw):
        self.calls.append(list(argv))
        return subprocess.CompletedProcess(argv, 0, b"", b"")


@pytest.fixture
def ran(monkeypatch):
    r = Ran()
    monkeypatch.setattr(subprocess, "run", r)
    monkeypatch.setattr(shutil, "which", lambda name: f"/usr/bin/{name}")
    return r


def _xdotool_calls(ran: Ran) -> list[tuple[str, str]]:
    out = []
    for argv in ran.calls:
        if argv[0] == "xdotool" and argv[1] == "type":
            out.append(("type", argv[-1]))
        elif argv[0] == "xdotool" and argv[1] == "key":
            out.append(("key", argv[-1]))
    return out


def test_xdotool_two_lines_default_is_return(ran):
    Injector({"inject": {}})._xdotool("原文\nคำแปล", {})
    assert _xdotool_calls(ran) == [("type", "原文"), ("key", "Return"), ("type", "คำแปล")]


def test_xdotool_chat_mode_sends_shift_return(ran):
    Injector({"inject": {}})._xdotool("原文\nคำแปล", {"newline_key": "SHIFT+RETURN"})
    assert _xdotool_calls(ran) == [
        ("type", "原文"), ("key", "shift+Return"), ("type", "คำแปล")]


def test_xdotool_global_default_from_config(ran):
    Injector({"inject": {"newline_key": "SHIFT+RETURN"}})._xdotool("a\nb", {})
    assert [c for c in _xdotool_calls(ran) if c[0] == "key"] == [("key", "shift+Return")]


def test_xdotool_single_line_is_one_type_call(ran):
    Injector({"inject": {}})._xdotool("one line", {"newline_key": "SHIFT+RETURN"})
    assert _xdotool_calls(ran) == [("type", "one line")]


def test_xdotool_blank_line_is_two_keys_and_no_empty_type(ran):
    # "a\n\nb": a blank line between paragraphs is two newlines, and there is
    # nothing to type between them -- xdotool type "" would still be a call.
    Injector({"inject": {}})._xdotool("a\n\nb", {})
    assert _xdotool_calls(ran) == [
        ("type", "a"), ("key", "Return"), ("key", "Return"), ("type", "b")]


def test_wtype_two_lines_sends_key_between_segments(ran):
    Injector({"inject": {}})._wtype("first\nsecond", {"newline_key": "SHIFT+RETURN"})
    wtype = [argv for argv in ran.calls if argv[0] == "wtype"]
    # type, key, type
    assert wtype[0][-1] == "-" and wtype[2][-1] == "-"
    assert wtype[1] == ["wtype", "-M", "shift", "-k", "Return", "-m", "shift"]


def test_clipboard_route_keeps_newlines_in_the_text(ran, monkeypatch):
    # A paste carries the text whole; no newline key is involved.
    monkeypatch.setattr(inject_mod.time, "sleep", lambda s: None)
    inj = Injector({"inject": {"restore_clipboard_after": 0}})
    inj._clipboard("a\nb", {}, "xdotool")
    copied = [argv for argv in ran.calls if argv[0] == "wl-copy"]
    assert copied == [["wl-copy", "--"]]
    assert not [argv for argv in ran.calls if argv[:2] == ["xdotool", "type"]]


@pytest.mark.parametrize("combo,expect", [
    ("SHIFT+RETURN", "shift+Return"),
    ("shift+return", "shift+Return"),
    ("CTRL+V", "ctrl+v"),
    ("CTRL+SHIFT+V", "ctrl+shift+v"),
    ("KP_ENTER", "KP_Enter"),
    ("SHIFT+SPACE", "shift+space"),
    ("TAB", "Tab"),
])
def test_keysym_spelling_for_xdotool(ran, combo, expect):
    # X keysyms are case-sensitive: "return" is not a key, "Return" is. The
    # lowering that made "CTRL+V" into "ctrl+v" would have made SHIFT+RETURN
    # into shift+return, which xdotool cannot send.
    Injector._send_paste(combo, "xdotool")
    assert ran.calls[-1][-1] == expect


# -- what is owed after a partial typing, and chat windows by default --------


def _win(cls: str, xwayland: bool = True):
    return SimpleNamespace(cls=cls, xwayland=xwayland, title="", app_id=cls)


def test_chat_window_defaults_to_shift_return(ran):
    Injector({"inject": {"chat_classes": ["wechat", "slack"]}})._xdotool(
        "原文\nคำแปล", {}, _win("wechat"))
    assert [c for c in _xdotool_calls(ran) if c[0] == "key"] == [("key", "shift+Return")]


def test_mode_newline_key_beats_the_chat_default(ran):
    Injector({"inject": {"chat_classes": ["wechat"]}})._xdotool(
        "a\nb", {"newline_key": "RETURN"}, _win("wechat"))
    assert [c for c in _xdotool_calls(ran) if c[0] == "key"] == [("key", "Return")]


def test_editor_window_keeps_return(ran):
    Injector({"inject": {"chat_classes": ["wechat"]}})._xdotool("a\nb", {}, _win("code"))
    assert [c for c in _xdotool_calls(ran) if c[0] == "key"] == [("key", "Return")]


def test_failure_after_first_line_owes_only_the_rest(ran, monkeypatch):
    inj = Injector({"inject": {}})
    calls = {"n": 0}

    def flaky_key(combo, method="shortcut"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("xdotool key failed")
    monkeypatch.setattr(inj, "_send_paste", flaky_key)
    with pytest.raises(PartiallyTyped) as info:
        inj._xdotool("第一行\n第二行", {})
    assert info.value.remaining == "第二行"
    assert _xdotool_calls(ran) == [("type", "第一行")]


def test_failure_typing_the_first_line_is_a_plain_error(ran, monkeypatch):
    # Nothing reached the window, so the caller may retry with everything.
    inj = Injector({"inject": {}})
    monkeypatch.setattr(inj, "_xdotool_type", lambda text: (_ for _ in ()).throw(RuntimeError("XTEST refused")))
    with pytest.raises(RuntimeError) as info:
        inj._xdotool("第一行\n第二行", {})
    assert not isinstance(info.value, PartiallyTyped)


def test_inject_falls_back_with_the_remainder_not_the_whole(ran, monkeypatch):
    # xdotool types line one, the newline key fails; the fallback for an X11
    # window is the clipboard, and what it must carry is line two alone --
    # otherwise line one lands twice.
    monkeypatch.setattr(inject_mod.time, "sleep", lambda s: None)
    inj = Injector({"inject": {"restore_clipboard_after": 0}})
    calls = {"n": 0}

    def flaky_key(combo, method="shortcut"):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("xdotool key failed")
    monkeypatch.setattr(inj, "_send_paste", flaky_key)
    copied: list[bytes] = []
    real_run = subprocess.run

    def run(argv, **kw):
        if argv[0] == "wl-copy":
            copied.append(kw.get("input", b""))
        return real_run(argv, **kw)
    monkeypatch.setattr(subprocess, "run", run)
    win = _win("wechat", xwayland=True)
    result = inj.inject("第一行\n第二行", win, {})
    assert result.ok and result.fell_back and result.method == "clipboard"
    assert copied == ["第二行".encode()]
