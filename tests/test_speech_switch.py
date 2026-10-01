"""One voice model for every mode, switched without a restart.

A Daemon is not built here -- that opens the microphone and loads a model --
but made without __init__ and given only what switching touches: the config,
the engine, the pipeline that holds it, and the state it reports.
"""
from __future__ import annotations

import argparse
import copy
import threading

from omavoi import config
from omavoi import daemon as daemon_mod
from omavoi.commands import modes as modes_cmd


class FakeEngine:
    def __init__(self, cfg, log, *, fails=False, port=0):
        self.model = cfg["speech"]["model"]
        self.port = port
        self.log = log
        self.fails = fails

    def load(self):
        self.log.append(("load", self.model))
        if self.fails:
            raise RuntimeError("weights would not load")

    def close(self):
        self.log.append(("close", self.model))

    def describe(self):
        return self.model


class Holder:
    def __init__(self, backend):
        self.backend = backend


def _daemon(home, monkeypatch, model="ggml:qwen3-asr-1.7b"):
    monkeypatch.setattr(daemon_mod.notify, "send", lambda *a, **k: None)
    cfg = config.defaults()
    cfg["speech"]["model"] = model
    log: list = []
    d = object.__new__(daemon_mod.Daemon)
    d.cfg = cfg
    d.backend = FakeEngine(cfg, log)
    d.pipeline = Holder(d.backend)
    d.state_file = home / "state-file"
    d._state = daemon_mod.IDLE
    d._mode_hint = "default"
    d._ticker = None
    d._subs, d._subs_lock = [], threading.Lock()
    d._lock = threading.Lock()
    d._speech_cfg = copy.deepcopy(cfg["speech"])
    d._speech_failed = None
    d._speech_pending = False
    d._speech_lock = threading.Lock()
    return d, log


def _build_with(monkeypatch, log, **kw):
    built = []

    def build(cfg):
        engine = FakeEngine(cfg, log, **kw)
        built.append(engine)
        return engine

    monkeypatch.setattr(daemon_mod.asr, "build", build)
    return built


def test_a_new_voice_model_is_loaded_before_the_old_one_is_closed(home, monkeypatch):
    d, log = _daemon(home, monkeypatch)
    built = _build_with(monkeypatch, log)
    d.cfg["speech"]["model"] = "ggml:large-v3-turbo"
    assert d._apply_speech() == ""
    assert log == [("load", "ggml:large-v3-turbo"), ("close", "ggml:qwen3-asr-1.7b")]
    assert d.backend is built[0] and d.pipeline.backend is built[0]
    assert d._state == daemon_mod.IDLE


def test_nothing_changed_means_nothing_is_loaded(home, monkeypatch):
    d, log = _daemon(home, monkeypatch)
    built = _build_with(monkeypatch, log)
    assert d._apply_speech() == ""
    assert built == [] and log == []


def test_weights_that_will_not_load_leave_the_old_ones_working(home, monkeypatch):
    d, log = _daemon(home, monkeypatch)
    old = d.backend
    built = _build_with(monkeypatch, log, fails=True)
    d.cfg["speech"]["model"] = "ggml:large-v3"
    error = d._apply_speech()
    assert "ggml:large-v3" in error and "would not load" in error
    assert d.backend is old and d.pipeline.backend is old
    assert ("close", "ggml:qwen3-asr-1.7b") not in log
    assert d._state == daemon_mod.IDLE
    # Tried once per change, not on every reload after it.
    assert d._apply_speech() == ""
    assert len(built) == 1


def test_a_switch_during_a_take_waits_for_it(home, monkeypatch):
    d, log = _daemon(home, monkeypatch)
    built = _build_with(monkeypatch, log)
    d._state = daemon_mod.RECORDING
    d.cfg["speech"]["model"] = "ggml:large-v3-turbo"
    assert d._apply_speech() == ""
    assert built == [] and d._speech_pending is True
    d._state = daemon_mod.IDLE
    assert d._apply_speech() == ""
    assert d.backend is built[0] and d._speech_pending is False


def test_one_fixed_port_closes_the_old_server_first_and_restores_it(home, monkeypatch):
    d, log = _daemon(home, monkeypatch)
    d.backend.port = 8178
    _build_with(monkeypatch, log, fails=True, port=8178)
    d.cfg["speech"]["model"] = "ggml:qwen3-asr-0.6b"
    assert d._apply_speech() != ""
    assert log == [("close", "ggml:qwen3-asr-1.7b"), ("load", "ggml:qwen3-asr-0.6b"),
                   ("load", "ggml:qwen3-asr-1.7b"), ("close", "ggml:qwen3-asr-0.6b")]


def test_a_mode_of_its_own_voice_model_is_taken_out_of_the_config():
    cfg = config.defaults()
    cfg["modes"]["prose"]["speech_model"] = "ggml:small"
    cfg["modes"]["code"]["speech_model"] = ""
    notes = config.retire_mode_speech_model(cfg)
    assert all("speech_model" not in m for m in cfg["modes"].values())
    assert any("prose" in n and "ggml:small" in n for n in notes)
    assert config.retire_mode_speech_model(cfg) == []


def test_mode_set_speech_model_says_where_the_voice_model_is_chosen(home, capsys):
    config.write(config.defaults())
    code = modes_cmd.cmd_mode(argparse.Namespace(
        action="set", rest=["default", "speech_model", "ggml:large-v3-turbo"], json=False))
    assert code == 1
    assert "same for every mode" in capsys.readouterr().err
    assert "speech_model" not in config.load()["modes"]["default"]


def test_mode_list_no_longer_reports_a_voice_model_per_mode(home, monkeypatch, capsys):
    import json

    from omavoi.window import Window

    config.write(config.defaults())
    monkeypatch.setattr("omavoi.window.active_window", lambda: Window())
    assert modes_cmd.cmd_mode(argparse.Namespace(action="list", rest=[], json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["modes"] and all("speech_model" not in m for m in payload["modes"])
