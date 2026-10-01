"""Qwen3-ASR as a second local speech engine, beside whisper.cpp.

The parts that are ours: reading its reply, keeping two files together as one
model, sending each set of weights to the engine that can run them, and never
stopping a working engine for weights it cannot load. No server is started
here; the HTTP client is replaced with one that keeps what it was sent.
"""
from __future__ import annotations

import numpy as np
import pytest

from omavoi import asr, config, models
from omavoi.asr import local_llama_asr
from omavoi.asr.base import NotReady
from omavoi.asr.local_llama_asr import LlamaAsrBackend, parse
from omavoi.asr.local_whispercpp import WhisperCppBackend

QWEN = "ggml:qwen3-asr-1.7b"


# -- the reply --------------------------------------------------------------

def test_the_language_prefix_is_read_and_removed():
    assert parse("language Chinese<asr_text>这个 PR 我已经 review 过了。") == (
        "这个 PR 我已经 review 过了。", "chinese")


@pytest.mark.parametrize("reply", [
    "language Chinese<asr_text>",
    "language Chinese<asr_text>Transcribe audio to text.",
    "language English<asr_text> transcribe audio to text ",
    "language None<asr_text>",
])
def test_silence_comes_back_empty(reply):
    """What it wrote for silent and noise-only clips: nothing, or its own
    instruction. Typed, the second is a sentence nobody said."""
    assert parse(reply)[0] == ""


def test_a_reply_without_the_prefix_is_kept_as_it_is():
    assert parse(" hello ") == ("hello", "")


def test_a_sentence_that_mentions_the_phrase_is_not_silence():
    text, _ = parse("language English<asr_text>Can it transcribe audio to text offline?")
    assert text == "Can it transcribe audio to text offline?"


# -- the catalogue ----------------------------------------------------------

def test_it_is_a_speech_model_on_its_own_engine():
    spec = models.spec(QWEN)
    assert spec is not None and spec.kind == models.SPEECH
    assert spec.backend == "local-llama-asr"
    # Listed by the console with the other speech models, which it finds by
    # this format.
    assert spec.fmt == models.GGML
    assert spec.extra_files, "the audio encoder is part of the model"


def _ggml_dir():
    d = models.model_root() / "ggml"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_the_model_file_alone_is_not_downloaded(home):
    spec = models.spec(QWEN)
    (_ggml_dir() / spec.filename).write_bytes(b"model")
    assert not models.is_downloaded(QWEN)
    (_ggml_dir() / spec.extra_files[0]).write_bytes(b"encoder")
    assert models.is_downloaded(QWEN)


def test_removing_it_removes_both_files(home):
    spec = models.spec(QWEN)
    for name in spec.files:
        (_ggml_dir() / name).write_bytes(b"x")
    assert models.remove(QWEN)
    assert not any((_ggml_dir() / name).exists() for name in spec.files)


def test_progress_counts_the_files_already_finished(home):
    spec = models.spec(QWEN)
    (_ggml_dir() / spec.extra_files[0]).write_bytes(b"e" * 300)
    part = _ggml_dir() / ".cache" / "huggingface" / "download"
    part.mkdir(parents=True)
    (part / f"{spec.filename}.incomplete").write_bytes(b"m" * 1000)
    assert models.bytes_in_flight(QWEN) == 1300
    # Nothing in flight is "not downloading", whatever is on disk.
    (part / f"{spec.filename}.incomplete").unlink()
    assert models.bytes_in_flight(QWEN) == 0


# -- which engine runs what -------------------------------------------------

def _cfg(backend, model):
    cfg = config.defaults()
    cfg["speech"]["backend"] = backend
    cfg["speech"]["model"] = model
    return cfg


@pytest.mark.parametrize("backend,model,engine", [
    ("local-llama-asr", QWEN, "local-llama-asr"),
    # What the console's "on this computer" card writes: the backend only.
    ("local-whispercpp", QWEN, "local-llama-asr"),
    ("local-llama-asr", "ggml:large-v3-turbo", "local-whispercpp"),
    ("local-whispercpp", "ggml:large-v3-turbo", "local-whispercpp"),
    # Remote stays remote: speech.model is not what an API runs.
    ("api", QWEN, "api"),
])
def test_the_weights_choose_the_local_engine(backend, model, engine):
    assert asr.engine_for(_cfg(backend, model)) == engine


def test_build_returns_the_engine_that_can_run_the_weights():
    assert isinstance(asr.build(_cfg("local-whispercpp", QWEN)), LlamaAsrBackend)
    assert isinstance(asr.build(_cfg("local-llama-asr", "ggml:large-v3-turbo")),
                      WhisperCppBackend)


def test_whisper_refuses_qwen_weights_before_stopping_anything(monkeypatch):
    """The daemon calls use() on every config reload, so this is what runs
    the moment Qwen3-ASR is chosen in the console."""
    backend = WhisperCppBackend(_cfg("local-whispercpp", "ggml:large-v3-turbo"))
    stopped = []
    monkeypatch.setattr(backend, "close", lambda: stopped.append(True))
    with pytest.raises(NotReady, match="local-llama-asr"):
        backend.use(QWEN)
    assert not stopped
    assert backend.model_key == "ggml:large-v3-turbo"


def test_qwen_refuses_whisper_weights_before_stopping_anything(monkeypatch):
    backend = LlamaAsrBackend(_cfg("local-llama-asr", QWEN))
    stopped = []
    monkeypatch.setattr(backend, "close", lambda: stopped.append(True))
    with pytest.raises(NotReady, match="local-whispercpp"):
        backend.use("ggml:large-v3-turbo")
    assert not stopped


def test_it_needs_llama_server(monkeypatch):
    from omavoi.llm import llama_local

    monkeypatch.setattr(llama_local, "find_server", lambda: None)
    assert "llama-cpp" in asr.why_unavailable("local-llama-asr")
    monkeypatch.setattr(llama_local, "find_server", lambda: "/usr/bin/llama-server")
    assert asr.why_unavailable("local-llama-asr") == ""


# -- the request ------------------------------------------------------------

class FakeResponse:
    status_code = 200

    def __init__(self, text):
        self._payload = {"text": text}
        self.text = text

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, text):
        self._response = FakeResponse(text)
        self.sent = None

    def post(self, url, **kw):
        self.sent = {"url": url, **kw}
        return self._response


def _running(reply):
    backend = LlamaAsrBackend(_cfg("local-llama-asr", QWEN))
    client = FakeClient(reply)
    backend._client = client
    backend._url = "http://127.0.0.1:9"
    return backend, client


def test_a_take_becomes_a_transcript_with_its_language():
    backend, client = _running("language Chinese<asr_text>我们明天见。")
    result = backend.transcribe(np.zeros(16000, dtype=np.float32), 16000)
    assert client.sent["url"].endswith("/v1/audio/transcriptions")
    assert result.text == "我们明天见。"
    assert result.language == "chinese"
    assert result.segments == []
    assert result.model == QWEN
    assert result.audio_seconds == pytest.approx(1.0)


def test_the_names_prompt_is_not_sent_as_the_instruction():
    """llama-server puts `prompt` where "Transcribe audio to text" goes."""
    backend, client = _running("language Chinese<asr_text>好的。")
    backend.transcribe(np.zeros(8000, dtype=np.float32), 16000,
                       language="zh", prompt="Omavoi, Hyprland")
    assert "prompt" not in client.sent["data"]
    assert client.sent["data"]["language"] == "zh"


def test_auto_is_not_sent_as_a_language():
    backend, client = _running("language English<asr_text>hello")
    backend.transcribe(np.zeros(8000, dtype=np.float32), 16000, language="auto")
    assert "language" not in client.sent["data"]


def test_module_name_matches_the_registry():
    assert local_llama_asr.NAME in asr.BACKENDS
    assert local_llama_asr.NAME in asr.LOCAL


# -- the default ------------------------------------------------------------

def _speech():
    return [entry for entry in models.CATALOG if entry.kind == models.SPEECH]


def test_qwen3_asr_is_what_a_fresh_install_runs(home):
    """No config file at all: the daemon's own defaults decide."""
    cfg = config.load()
    assert cfg["speech"]["model"] == QWEN
    assert asr.engine_for(cfg) == "local-llama-asr"
    assert isinstance(asr.build(cfg), LlamaAsrBackend)


def test_the_shipped_speech_model_is_the_one_recommended_and_listed_first():
    """The console lists speech models in catalogue order and marks the
    recommended one; the first run offers the default first. All three are
    the same model or none of them means anything."""
    recommended = [entry.key for entry in _speech() if "recommended" in entry.tags]
    assert recommended == [config.DEFAULTS["speech"]["model"]]
    assert _speech()[0].key == config.DEFAULTS["speech"]["model"]
    assert models.spec(config.DEFAULTS["speech"]["model"]).backend == \
        config.DEFAULTS["speech"]["backend"]


def test_a_config_that_already_names_turbo_keeps_it(home):
    """Changing the default reaches nobody who has chosen: every config the
    CLI writes names its model."""
    config.set_path("speech.model", "ggml:large-v3-turbo")
    config.set_path("speech.backend", "local-whispercpp")
    cfg = config.load()
    assert cfg["speech"]["model"] == "ggml:large-v3-turbo"
    assert asr.engine_for(cfg) == "local-whispercpp"


def test_the_language_picker_offers_only_what_qwen3_asr_hears():
    from omavoi import input_language

    qwen = {row["value"] for row in input_language.options(QWEN)}
    whisper = {row["value"] for row in input_language.options("ggml:large-v3-turbo")}
    assert {"auto", "zh", "en", "th", "yue", "ja", "vi"} <= qwen
    assert "sw" not in qwen and "sw" in whisper
    assert len(qwen) == 31, "auto and the 30 languages on its card"
    assert all(code in {r["code"] for r in input_language.catalogue()}
               for code in models.spec(QWEN).language_codes), "spelled as the catalogue spells them"
