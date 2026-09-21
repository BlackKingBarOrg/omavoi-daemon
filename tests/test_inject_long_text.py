"""Long Wayland dictation must survive keymap changes and partial failures."""
from __future__ import annotations

import subprocess
from unittest.mock import Mock

import pytest

from omavoi import inject as inject_mod
from omavoi.inject import Injector, PartiallyTyped
from omavoi.window import Window


@pytest.mark.parametrize("length", [63, 64, 65, 99, 100, 101, 200, 1001])
def test_long_text_survives_a_client_retaining_the_first_stdin_keymap(monkeypatch, length):
    # Every character is new: repeating the same sentence would conceal the
    # original bug, since previously mapped characters still arrive.
    text = "".join(chr(0x4E00 + i) for i in range(length))
    received = []

    def stale_keymap(argv, *, input, **kwargs):
        assert argv == ["wtype", "-"]  # dictation stays out of process argv
        payload = input.decode("utf-8")
        known = set(payload[:100])
        received.append(payload[:100] + "".join(c for c in payload[100:] if c in known))
        return subprocess.CompletedProcess(argv, 0, b"", b"")

    monkeypatch.setattr(inject_mod.subprocess, "run", stale_keymap)
    Injector({"inject": {}})._wtype_type(text)
    assert "".join(received) == text


def test_batches_preserve_utf8_literals_delay_and_conservative_keycodes(monkeypatch):
    text = ("-M ctrl 中文繁體日本語ภาษาไทย🙂👩🏽‍💻e\u0301 $HOME `literal` ") * 12
    run = Mock(return_value=subprocess.CompletedProcess([], 0, b"", b""))
    monkeypatch.setattr(inject_mod.subprocess, "run", run)
    Injector({"inject": {"wtype_delay_ms": 500}})._wtype_type(text)
    payloads = []
    for call in run.call_args_list:
        assert call.args[0] == ["wtype", "-d", "500", "-"]
        payload = call.kwargs["input"].decode("utf-8")
        assert len(payload) < 99  # don't allocate the physical Print keycode
        assert call.kwargs["timeout"] > len(payload) * .5
        payloads.append(payload)
    assert "".join(payloads) == text


def test_empty_text_needs_no_virtual_keyboard(monkeypatch):
    run = Mock()
    monkeypatch.setattr(inject_mod.subprocess, "run", run)
    Injector({"inject": {}})._wtype_type("")
    run.assert_not_called()


@pytest.mark.parametrize("failure", ["exit", "timeout", "spawn"])
@pytest.mark.parametrize("prefix", ["", "上一行\n"])
@pytest.mark.parametrize("suffix", ["", "\n末行"])
def test_fallback_omits_completed_batches_and_keeps_following_lines(monkeypatch, failure, prefix, suffix):
    text = prefix + "甲" * 80 + "乙丙丁" * 60 + suffix
    sent = []
    copied = []

    def run(argv, **kwargs):
        payload = kwargs.get("input", b"").decode()
        if argv[0] == "wl-copy":
            copied.append(payload)
        elif argv[0] == "wtype" and argv[-1] == "-":
            # Fail after the first successful batch within the long line.
            if "甲" in "".join(sent):
                if failure == "timeout":
                    raise subprocess.TimeoutExpired(argv, 30)
                if failure == "spawn":
                    raise OSError("cannot spawn")
                return subprocess.CompletedProcess(argv, 1, b"", b"disconnected")
            sent.append(payload)
        return subprocess.CompletedProcess(argv, 0, b"ok", b"")

    monkeypatch.setattr(inject_mod.subprocess, "run", run)
    monkeypatch.setattr(inject_mod.shutil, "which", lambda name: f"/usr/bin/{name}")
    monkeypatch.setattr(inject_mod.time, "sleep", lambda _: None)
    result = Injector({"inject": {"restore_clipboard_after": 0}}).inject(text, Window(cls="foot"))
    assert result.ok and result.fell_back and result.method == "clipboard"
    completed = "".join(sent) if not prefix else sent[0] + "\n" + "".join(sent[1:])
    assert len(copied) == 1
    assert completed + copied[0] == text
    assert copied[0].endswith(suffix)


def test_first_batch_failure_does_not_claim_a_completed_prefix(monkeypatch):
    monkeypatch.setattr(inject_mod.subprocess, "run", Mock(
        return_value=subprocess.CompletedProcess([], 1, b"", b"disconnected")))
    with pytest.raises(RuntimeError, match="disconnected") as error:
        Injector({"inject": {}})._wtype_type("甲" * 101)
    assert not isinstance(error.value, PartiallyTyped)
