"""A language selection must reach ASR without changing translation output."""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from omavoi import config, input_language, vocabulary
from omavoi.asr.base import Segment, Transcript
from omavoi.commands import modes, settings
from omavoi.pipeline import Pipeline
from omavoi.window import Window


@pytest.mark.parametrize("locale", ["en", "zh", "th", "de", "fr", "es", "ja", "vi"])
def test_searchable_catalogue_has_all_languages_and_script_choices(locale):
    options = input_language.options("ggml:large-v3-turbo", locale)
    assert len(options) == len({o["value"] for o in options}) == 102
    assert [o["value"] for o in options[:5]] == ["auto", "zh-Hans", "zh-Hant", "en", "th"]
    english = next(o for o in options if o["value"] == "en")
    searchable = english["label"] + english["description"]
    for query in ("英语", "English", "en"):
        assert query in searchable
    assert all(o["label"] for o in options)


def test_older_models_do_not_offer_the_v3_cantonese_token():
    assert "yue" not in {o["value"] for o in input_language.options("ggml:small")}
    assert "yue" in {o["value"] for o in input_language.options("ggml:large-v3-q5_0")}
    assert {o["value"] for o in input_language.options("base.en")} == {"auto", "en"}


def test_existing_chinese_choice_does_not_silently_become_simplified():
    assert "zh" in {o["value"] for o in input_language.options("ggml:large-v3", current="zh")}
    assert input_language.convert("這是舊設定。", "zh") == "這是舊設定。"


@pytest.mark.parametrize("language,before,after", [
    ("zh-Hans", "這個 API 很穩定，版本 123。", "这个 API 很稳定，版本 123。"),
    ("zh-Hant", "这个 API 很稳定，版本 123。", "這個 API 很穩定，版本 123。"),
    ("zh-Hans", "滑鼠與軟體", "滑鼠与软体"),  # script, not regional wording
    ("zh-Hant", "头发和发展", "頭髮和發展"),  # phrase-aware conversion
    ("auto", "這個 API 很穩定。", "這個 API 很穩定。"),
    ("", "這個 API 很穩定。", "這個 API 很穩定。"),
    ("en", "這個 API 很穩定。", "這個 API 很穩定。"),
    ("zh-Hans", "ภาษาไทย English 123", "ภาษาไทย English 123"),
])
def test_script_conversion_has_narrow_scope(language, before, after):
    assert input_language.convert(before, language) == after


@pytest.mark.parametrize("literal", [
    "`繁體變數`", "```python\n繁體變數 = 1\n```", "https://example.com/繁體",
    "/home/繁體/檔案.txt", "./繁體/檔案.txt", "資料/檔案.txt", r"C:\繁體\檔案.txt",
    "繁體@example.com", "日本語の説明", "繁體工作室",
])
def test_verbatim_text_and_dictionary_spelling_survive(literal):
    assert input_language.convert("請保留：" + literal + "。這是測試。", "zh-Hans", ("繁體工作室",)) == (
        "请保留：" + literal + "。这是测试。")


@pytest.mark.parametrize("value", ["zh-Hans", "zh-Hant", "auto", "", "en", "yue"])
def test_selection_round_trips_through_both_settings_paths(home, value):
    assert modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "language", value])) == 0
    assert config.load()["modes"]["default"]["language"] == value
    assert settings._why_not_that_language("modes.default.language", value) == ""


def test_invalid_selection_is_refused_without_changing_config(home):
    modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "language", "zh-Hans"]))
    assert modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "language", "xx"])) == 1
    assert config.load()["modes"]["default"]["language"] == "zh-Hans"


@pytest.mark.parametrize("backend,count", [("local-whispercpp", 102), ("api", 101)])
def test_menu_payload_follows_the_mode_model_and_interface_locale(home, monkeypatch, capsys, backend, count):
    cfg = config.defaults()
    cfg["ui"]["language"] = "zh"
    cfg["speech"]["backend"] = backend
    cfg["speech"]["api"]["model"] = "whisper-1"
    cfg["modes"]["default"]["language"] = "zh-Hant"
    cfg["modes"]["prose"]["speech_model"] = "ggml:small"
    config.write(cfg)
    monkeypatch.setattr(modes.ipc, "ping", lambda: None)
    monkeypatch.setattr("omavoi.window.active_window", lambda: Window())
    assert modes.cmd_mode(argparse.Namespace(action="list", rest=[], json=True)) == 0
    payload = json.loads(capsys.readouterr().out)
    rows = {r["name"]: r for r in payload["modes"]}
    assert rows["default"]["language"] == "zh-Hant"
    assert rows["default"]["input_languages"][1]["label"] == "简体中文"
    assert len(rows["default"]["input_languages"]) == count
    assert len(rows["prose"]["input_languages"]) == 101


def run_pipeline(cfg, text, monkeypatch, rewrite=None):
    cfg["ui"]["notify"] = False
    received = []
    transcript = Transcript(text, segments=[Segment(0, 1, text, -0.1, 0.01)])

    def transcribe(*args, **kwargs):
        received.append(kwargs)
        return transcript

    pipeline = Pipeline(cfg, SimpleNamespace(transcribe=transcribe),
                        history=SimpleNamespace(record=lambda *a: None))
    if rewrite is not None:
        monkeypatch.setattr(pipeline, "_run_steps", rewrite)
    cap = SimpleNamespace(samples=np.zeros(16000, dtype=np.float32), rate=16000,
                          seconds=1, peak_dbfs=-10, rms_dbfs=-15, preroll_seconds=0,
                          tail_seconds=0, truncated=False)
    return pipeline.process(cap, inject=False, window=Window(), forced_mode="default"), received


def test_pipeline_converts_before_ai_and_keeps_raw_evidence(home, monkeypatch):
    cfg = config.defaults()
    cfg["modes"]["default"].update(language="zh-Hans", prompt="Omarchy", steps=[{"llm": "local"}])

    def translate(text, *args):
        assert text == "这个 API 很稳定。"
        return "日本語の説明：繁體"

    entry, received = run_pipeline(cfg, "這個 API 很穩定。", monkeypatch, translate)
    assert received[0]["language"] == "zh"
    assert received[0]["prompt"].startswith("Omarchy")
    assert entry["raw_text"] == entry["asr"]["segments"][0]["text"] == "這個 API 很穩定。"
    assert entry["rules_text"] == "这个 API 很稳定。"
    assert entry["post"]["changed"] is True
    assert entry["text"] == "日本語の説明：繁體"
    assert "Chinese script: zh-Hans" in entry["post"]["changes"]


def test_auto_clears_script_preference_and_overrides_global_language(home, monkeypatch):
    cfg = config.defaults()
    cfg["speech"]["language"] = "en"
    cfg["modes"]["default"]["language"] = "auto"
    entry, received = run_pipeline(cfg, "這個 API 很穩定。", monkeypatch)
    assert received[0]["language"] == "auto"
    assert entry["text"] == "這個 API 很穩定。"


def test_global_language_remains_effective_for_legacy_empty_modes(home, monkeypatch):
    cfg = config.defaults()
    cfg["speech"]["language"] = "th"
    _, received = run_pipeline(cfg, "ภาษาไทย", monkeypatch)
    assert received[0]["language"] == "th"


def test_dictionary_spelling_wins_over_script_preference(home, monkeypatch):
    cfg = config.defaults()
    cfg["dictionary"] = {"schema_version": 2, "revision": 0,
                         "entries": [vocabulary.word("繁體工作室")]}
    cfg["modes"]["default"]["language"] = "zh-Hans"
    entry, _ = run_pipeline(cfg, "聯繫繁體工作室。", monkeypatch)
    assert entry["text"] == "联系繁體工作室。"


def test_conversion_failure_keeps_the_transcript_and_reports_it(home, monkeypatch):
    cfg = config.defaults()
    cfg["modes"]["default"]["language"] = "zh-Hans"

    def broken(*args):
        raise RuntimeError("dictionary unavailable")

    monkeypatch.setattr(input_language, "convert", broken)
    entry, _ = run_pipeline(cfg, "這是文字。", monkeypatch)
    assert entry["text"] == "這是文字。"
    assert any("dictionary unavailable" in w for w in entry["warnings"])
    assert not entry["rejected"]


@pytest.mark.parametrize("language", ["auto", "zh-Hans", "zh-Hant", "en"])
@pytest.mark.parametrize("local", [True, False])
def test_backend_receives_valid_language_parameters(language, local):
    from omavoi.asr.api_whisper import ApiWhisperBackend
    from omavoi.asr.local_whispercpp import WhisperCppBackend

    cfg = config.defaults()
    cfg["speech"]["language"] = "en"
    backend = WhisperCppBackend(cfg) if local else ApiWhisperBackend(cfg)
    backend._url = "http://whisper.test"
    bodies = []

    def respond(request):
        bodies.append(request.content)
        return httpx.Response(200, json={"text": "hello"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        backend._client = client
        backend.transcribe(np.zeros(160, dtype=np.float32), 16000,
                           language=input_language.recognition_language(language))
    body = bodies[0]
    if not local and language == "auto":
        assert b'name="language"' not in body
    else:
        code = input_language.recognition_language(language).encode()
        assert b'name="language"\r\n\r\n' + code + b'\r\n' in body
    assert b"zh-Hans" not in body and b"zh-Hant" not in body
