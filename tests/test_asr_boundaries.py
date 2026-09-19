"""Subtitle segmentation must not rewrite the decoder's text."""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from omavoi import config
from omavoi.asr.local_whispercpp import WhisperCppBackend
from omavoi.post import rules


def _transcribe(payload):
    backend = WhisperCppBackend(config.defaults())
    backend._url = "http://whisper.test"
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with httpx.Client(transport=transport) as client:
        backend._client = client
        return backend.transcribe(np.zeros(160, dtype=np.float32), 16000)


@pytest.mark.parametrize("chunks,want", [
    # These boundaries occurred in saved whisper.cpp transcripts.
    (["显示中", "文翻译"], "显示中文翻译"),
    (["mac", "OS"], "macOS"),
    (["hello", " world"], "hello world"),
    (["hello ", "world"], "hello world"),
    (["hello", " ", "world"], "hello world"),
    ([" inter", "national", " flights"], "international flights"),
    (["Hello.", " How are you?"], "Hello. How are you?"),
    (["我今天想说的是", "这个功能很好"], "我今天想说的是这个功能很好"),
    (["-3", " --no-color"], "-3 --no-color"),
    (["첫 번째", " 문장"], "첫 번째 문장"),
])
def test_local_decoder_preserves_original_segment_spacing(chunks, want):
    segments = [{"text": chunk, "start": i, "end": i + 1}
                for i, chunk in enumerate(chunks)]
    # whisper-server's top-level text adds its own subtitle newlines.
    transcript = _transcribe({"text": "\n".join(chunks) + "\n", "segments": segments})
    assert transcript.text == want
    assert [segment.text for segment in transcript.segments] == chunks
    assert [(segment.start, segment.end) for segment in transcript.segments] == [
        (i, i + 1) for i in range(len(chunks))]
    result = rules.run(transcript.text, config.defaults())
    assert result.text == want
    assert not result.rejected


def test_real_line_breaks_in_segment_text_are_preserved_by_the_backend():
    text = "First paragraph.\n\nSecond paragraph."
    transcript = _transcribe({"text": text, "segments": [{"text": text}]})
    assert transcript.text == text
    cfg = config.defaults()
    cfg["post"]["newlines"] = "keep"
    assert rules.run(transcript.text, cfg).text == text


@pytest.mark.parametrize("segments", [[], [{"text": " "}], None])
def test_top_level_text_is_used_when_no_segment_contains_text(segments):
    transcript = _transcribe({"text": "fallback text", "segments": segments})
    assert transcript.text == "fallback text"


def test_missing_confidence_is_not_reported_as_measured_zero():
    transcript = _transcribe({"segments": [
        {"text": "missing"},
        {"text": " null", "avg_logprob": None, "no_speech_prob": None},
        {"text": " zero", "avg_logprob": 0, "no_speech_prob": 0},
    ]})
    assert [(s.avg_logprob, s.no_speech_prob) for s in transcript.segments] == [
        (None, None), (None, None), (0.0, 0.0)]


def test_old_punctuation_argument_cannot_reintroduce_false_commas():
    assert rules.normalise_boundaries("中\n文翻译", add_punctuation=True) == "中文翻译"


def test_old_punctuation_setting_loads_but_reports_its_retirement(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[post]\nadd_missing_punctuation = true\n", encoding="utf-8")
    cfg = config.load(path)
    assert any("add_missing_punctuation is retired" in note for note in config.validate(cfg))
    assert "add_missing_punctuation" not in config.defaults()["post"]
    assert rules.run("中\n文翻译", cfg).text == "中文翻译"
