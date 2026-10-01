"""A language selection must reach ASR without changing translation output.

Which language is listened for and which Chinese characters come out are two
settings. They were one picker, where Simplified Chinese also meant "listen
for Chinese only" — so auto-detection with Simplified output was the one
combination a trilingual speaker wanted and the one that could not be chosen.
"""

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
def test_searchable_catalogue_lists_languages_not_scripts(locale):
    options = input_language.options("ggml:large-v3-turbo", locale)
    assert len(options) == len({o["value"] for o in options}) == 101
    assert [o["value"] for o in options[:4]] == ["auto", "zh", "en", "th"]
    assert not {"zh-Hans", "zh-Hant"} & {o["value"] for o in options}
    english = next(o for o in options if o["value"] == "en")
    searchable = english["label"] + english["description"]
    for query in ("英语", "English", "en"):
        assert query in searchable
    chinese = next(o for o in options if o["value"] == "zh")
    for query in ("中文", "Chinese", "zh"):
        assert query in chinese["label"] + chinese["description"]
    assert all(o["label"] for o in options)


def test_older_models_do_not_offer_the_v3_cantonese_token():
    assert "yue" not in {o["value"] for o in input_language.options("ggml:small")}
    assert "yue" in {o["value"] for o in input_language.options("ggml:large-v3-q5_0")}
    assert {o["value"] for o in input_language.options("base.en")} == {"auto", "en"}


def test_chinese_on_its_own_converts_nothing():
    """zh used to mean "Chinese, no conversion" beside two converting
    entries. It is plain Chinese now, and still converts nothing without a
    script."""
    assert "zh" in {o["value"] for o in input_language.options("ggml:large-v3")}
    assert input_language.convert("這是舊設定。", "") == "這是舊設定。"
    assert not input_language.applies("", "zh")


@pytest.mark.parametrize("script,language,detected,converts", [
    ("zh-Hans", "zh", "chinese", True),
    ("zh-Hans", "auto", "chinese", True),
    # A Chinese sentence with English in it is often detected as English.
    ("zh-Hans", "auto", "english", True),
    ("zh-Hant", "auto", "", True),
    ("zh-Hans", "auto", "japanese", False),
    ("zh-Hans", "auto", "korean", False),
    ("zh-Hans", "ja", "", False),
    ("zh-Hans", "ko", "korean", False),
    ("", "auto", "chinese", False),
    ("", "zh", "chinese", False),
])
def test_the_script_touches_any_take_but_japanese_and_korean(script, language, detected, converts):
    assert input_language.applies(script, language, detected) is converts


def test_a_legacy_script_language_splits_into_the_two_settings():
    assert input_language.split("zh-Hans") == ("zh", "zh-Hans")
    assert input_language.split("zh-Hant") == ("zh", "zh-Hant")
    assert input_language.split("en") == ("en", "")
    assert input_language.split("") == ("", "")


@pytest.mark.parametrize("script,before,after", [
    ("zh-Hans", "這個 API 很穩定，版本 123。", "这个 API 很稳定，版本 123。"),
    ("zh-Hant", "这个 API 很稳定，版本 123。", "這個 API 很穩定，版本 123。"),
    ("zh-Hans", "滑鼠與軟體", "滑鼠与软体"),  # script, not regional wording
    ("zh-Hant", "头发和发展", "頭髮和發展"),  # phrase-aware conversion
    ("auto", "這個 API 很穩定。", "這個 API 很穩定。"),
    ("", "這個 API 很穩定。", "這個 API 很穩定。"),
    ("en", "這個 API 很穩定。", "這個 API 很穩定。"),
    ("zh-Hans", "ภาษาไทย English 123", "ภาษาไทย English 123"),
])
def test_script_conversion_has_narrow_scope(script, before, after):
    assert input_language.convert(before, script) == after


@pytest.mark.parametrize("literal", [
    "`繁體變數`", "```python\n繁體變數 = 1\n```", "https://example.com/繁體",
    "/home/繁體/檔案.txt", "./繁體/檔案.txt", "資料/檔案.txt", r"C:\繁體\檔案.txt",
    "繁體@example.com", "日本語の説明", "繁體工作室",
])
def test_verbatim_text_and_dictionary_spelling_survive(literal):
    assert input_language.convert("請保留：" + literal + "。這是測試。", "zh-Hans", ("繁體工作室",)) == (
        "请保留：" + literal + "。这是测试。")


@pytest.mark.parametrize("value", ["zh", "auto", "", "en", "yue"])
def test_selection_round_trips_through_both_settings_paths(home, value):
    assert modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "language", value])) == 0
    assert config.load()["modes"]["default"]["language"] == value
    assert settings._why_not_that_language("modes.default.language", value) == ""


@pytest.mark.parametrize("value", ["zh-Hans", "zh-Hant", ""])
def test_script_round_trips(home, value):
    assert modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "script", value])) == 0
    assert config.load()["modes"]["default"]["script"] == value
    assert settings._why_not_that_choice("modes.default.script", value) == ""


@pytest.mark.parametrize("value", ["zh-Hans", "zh-Hant"])
def test_an_old_script_language_still_sets_what_it_meant(home, value):
    """Typed as a language, as it was until now: Chinese, and that script."""
    assert modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "language", value])) == 0
    mode = config.load()["modes"]["default"]
    assert (mode["language"], mode["script"]) == ("zh", value)
    # The config-set path writes it as given; the next load splits it.
    assert settings._why_not_that_language("modes.default.language", value) == ""


def test_invalid_selection_is_refused_without_changing_config(home):
    modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "language", "zh"]))
    modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "script", "zh-Hans"]))
    assert modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "language", "xx"])) == 1
    assert modes.cmd_mode(argparse.Namespace(action="set", rest=["default", "script", "simplified"])) == 1
    mode = config.load()["modes"]["default"]
    assert (mode["language"], mode["script"]) == ("zh", "zh-Hans")


def test_load_writes_an_old_script_language_as_two_settings(home):
    from omavoi import paths

    path = paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[modes.default]\nlanguage = "zh-Hans"\n\n'
                    '[modes.chat]\nlanguage = "zh-Hant"\nscript = "zh-Hans"\n', encoding="utf-8")
    cfg = config.load()
    assert (cfg["modes"]["default"]["language"], cfg["modes"]["default"]["script"]) == ("zh", "zh-Hans")
    # A script the mode already stated is its own.
    assert (cfg["modes"]["chat"]["language"], cfg["modes"]["chat"]["script"]) == ("zh", "zh-Hans")
    # Written back, so the file stops disagreeing with what runs.
    assert "zh-Hant" not in path.read_text(encoding="utf-8")
    assert config.split_chinese_script(config.load()) == []


def test_a_mode_inherits_the_script_unless_it_states_its_own(home):
    cfg = config.defaults()
    cfg["modes"]["default"]["script"] = "zh-Hans"
    cfg["modes"]["prose"]["script"] = ""
    from omavoi import modes as modes_mod

    assert modes_mod.resolve(cfg, forced="code").script == "zh-Hans"
    assert modes_mod.resolve(cfg, forced="prose").script == ""
    # An old in-memory value still means both, wherever load has not run.
    cfg["modes"]["code"]["language"] = "zh-Hant"
    code = modes_mod.resolve(cfg, forced="code")
    assert (code.language, code.script) == ("zh", "zh-Hant")


def mode_list(monkeypatch, capsys) -> dict:
    monkeypatch.setattr(modes.ipc, "ping", lambda: None)
    monkeypatch.setattr("omavoi.window.active_window", lambda: Window())
    assert modes.cmd_mode(argparse.Namespace(action="list", rest=[], json=True)) == 0
    return json.loads(capsys.readouterr().out)


@pytest.mark.parametrize("backend,model,count", [
    ("local-whispercpp", "ggml:large-v3-turbo", 101),
    # Qwen3-ASR's card lists 30 languages, and the menu offers those.
    ("local-llama-asr", "ggml:qwen3-asr-1.7b", 31),
    ("api", "ggml:large-v3-turbo", 100),
])
def test_menu_payload_follows_the_voice_model_and_interface_locale(home, monkeypatch, capsys,
                                                                   backend, model, count):
    cfg = config.defaults()
    cfg["ui"]["language"] = "zh"
    cfg["speech"]["backend"] = backend
    cfg["speech"]["model"] = model
    cfg["speech"]["api"]["model"] = "whisper-1"
    cfg["modes"]["default"]["language"] = "zh-Hant"
    config.write(cfg)
    payload = mode_list(monkeypatch, capsys)
    assert payload["script_supported"] is True
    rows = {r["name"]: r for r in payload["modes"]}
    assert (rows["default"]["language"], rows["default"]["script"]) == ("zh", "zh-Hant")
    assert rows["default"]["input_languages"][1]["label"] == "中文"
    assert len(rows["default"]["input_languages"]) == count
    # Every mode hears through the one voice model, so every mode's menu is
    # that model's. A mode used to be able to name weights of its own.
    assert len(rows["prose"]["input_languages"]) == count


def test_payload_reports_the_prompt_the_decoder_is_given(home, monkeypatch, capsys):
    """terminal, code and prose state no prompt, so they send default's —
    and the console showed them an empty box."""
    cfg = config.defaults()
    cfg["modes"]["default"]["prompt"] = "Hyprland, Omarchy"
    cfg["modes"]["chat"] = {"prompt": "", "steps": []}
    config.write(cfg)
    rows = {r["name"]: r for r in mode_list(monkeypatch, capsys)["modes"]}
    assert (rows["terminal"]["prompt"], rows["terminal"]["prompt_inherited"]) == ("Hyprland, Omarchy", True)
    assert (rows["default"]["prompt"], rows["default"]["prompt_inherited"]) == ("Hyprland, Omarchy", False)
    # An empty prompt the mode states is its own: no prompt at all.
    assert (rows["chat"]["prompt"], rows["chat"]["prompt_inherited"]) == ("", False)


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


def run_detected(cfg, text, detected, monkeypatch):
    """A take whose language the backend reports, as whisper.cpp does."""
    cfg["ui"]["notify"] = False
    transcript = Transcript(text, segments=[Segment(0, 1, text, -0.1, 0.01)], language=detected)
    pipeline = Pipeline(cfg, SimpleNamespace(transcribe=lambda *a, **k: transcript),
                        history=SimpleNamespace(record=lambda *a: None))
    cap = SimpleNamespace(samples=np.zeros(16000, dtype=np.float32), rate=16000,
                          seconds=1, peak_dbfs=-10, rms_dbfs=-15, preroll_seconds=0,
                          tail_seconds=0, truncated=False)
    return pipeline.process(cap, inject=False, window=Window(), forced_mode="default")


def test_pipeline_converts_before_ai_and_keeps_raw_evidence(home, monkeypatch):
    cfg = config.defaults()
    cfg["modes"]["default"].update(language="zh", script="zh-Hans", prompt="Omarchy",
                                   steps=[{"llm": "local"}])

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


def test_an_old_script_language_still_converts(home, monkeypatch):
    """A config that config.load has not split yet means what it meant."""
    cfg = config.defaults()
    cfg["modes"]["default"]["language"] = "zh-Hans"
    entry, received = run_pipeline(cfg, "這個 API 很穩定。", monkeypatch)
    assert received[0]["language"] == "zh"
    assert entry["text"] == "这个 API 很稳定。"


def test_auto_without_a_script_overrides_global_language_and_converts_nothing(home, monkeypatch):
    cfg = config.defaults()
    cfg["speech"]["language"] = "en"
    cfg["modes"]["default"]["language"] = "auto"
    entry, received = run_pipeline(cfg, "這個 API 很穩定。", monkeypatch)
    assert received[0]["language"] == "auto"
    assert entry["text"] == "這個 API 很穩定。"


def test_auto_with_a_script_keeps_detecting_and_converts(home, monkeypatch):
    """The combination the single picker could not express."""
    cfg = config.defaults()
    cfg["modes"]["default"].update(language="auto", script="zh-Hans")
    entry, received = run_pipeline(cfg, "這個 API 很穩定。", monkeypatch)
    assert received[0]["language"] == "auto"
    assert entry["raw_text"] == "這個 API 很穩定。"
    assert entry["text"] == "这个 API 很稳定。"
    assert "Chinese script: zh-Hans" in entry["post"]["changes"]


@pytest.mark.parametrize("detected", ["japanese", "korean"])
def test_the_script_leaves_a_detected_japanese_or_korean_take_alone(home, monkeypatch, detected):
    cfg = config.defaults()
    cfg["modes"]["default"].update(language="auto", script="zh-Hans")
    entry = run_detected(cfg, "東京駅", detected, monkeypatch)
    assert entry["text"] == "東京駅"
    assert not any(c.startswith("Chinese script") for c in entry["post"]["changes"])


def test_the_script_follows_a_detected_chinese_take(home, monkeypatch):
    cfg = config.defaults()
    cfg["modes"]["default"].update(language="auto", script="zh-Hant")
    entry = run_detected(cfg, "这个很稳定", "chinese", monkeypatch)
    assert entry["text"] == "這個很穩定"


def test_global_language_remains_effective_for_legacy_empty_modes(home, monkeypatch):
    cfg = config.defaults()
    cfg["speech"]["language"] = "th"
    _, received = run_pipeline(cfg, "ภาษาไทย", monkeypatch)
    assert received[0]["language"] == "th"


def test_dictionary_spelling_wins_over_script_preference(home, monkeypatch):
    cfg = config.defaults()
    cfg["dictionary"] = {"schema_version": 2, "revision": 0,
                         "entries": [vocabulary.word("繁體工作室")]}
    cfg["modes"]["default"]["script"] = "zh-Hans"
    entry, _ = run_pipeline(cfg, "聯繫繁體工作室。", monkeypatch)
    assert entry["text"] == "联系繁體工作室。"


def test_conversion_failure_keeps_the_transcript_and_reports_it(home, monkeypatch):
    cfg = config.defaults()
    cfg["modes"]["default"]["script"] = "zh-Hans"

    def broken(*args):
        raise RuntimeError("dictionary unavailable")

    monkeypatch.setattr(input_language, "convert", broken)
    entry, _ = run_pipeline(cfg, "這是文字。", monkeypatch)
    assert entry["text"] == "這是文字。"
    assert any("dictionary unavailable" in w for w in entry["warnings"])
    assert not entry["rejected"]


@pytest.mark.parametrize("language", ["auto", "zh", "zh-Hans", "zh-Hant", "en"])
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
