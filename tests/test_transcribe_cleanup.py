"""`omavoi transcribe` stops everything it started, the LLM servers included.

A mode with a local LLM step starts a llama-server for the take. The command
closed the speech engine and not the registry, so that server outlived it,
parented to init and still holding its weights.
"""
from __future__ import annotations

import argparse

import numpy as np

from omavoi import config, pipeline
from omavoi.commands import run


class FakeBackend:
    closed = False

    def load(self):
        pass

    def close(self):
        FakeBackend.closed = True


def test_the_llm_servers_stop_with_the_command(home, monkeypatch, tmp_path):
    from omavoi import asr

    wav = tmp_path / "take.wav"
    from omavoi.asr.api_whisper import encode_wav
    wav.write_bytes(encode_wav(np.zeros(16000, dtype=np.float32), 16000))

    closed = []
    monkeypatch.setattr(asr, "build", lambda cfg: FakeBackend())
    monkeypatch.setattr(pipeline.Pipeline, "process", lambda self, capture, **kw: {"text": ""})
    monkeypatch.setattr(pipeline.Registry, "close", lambda self: closed.append(True))
    monkeypatch.setattr(run, "_print_entry", lambda entry, verbose=False: None)
    config.write(config.defaults())

    args = argparse.Namespace(file=str(wav), inject=False, mode="prose", json=False,
                              verbose=False, log_level="")
    assert run.cmd_transcribe(args) == 0
    assert FakeBackend.closed, "the speech engine was not stopped"
    assert closed, "the LLM servers the take started were left running"
