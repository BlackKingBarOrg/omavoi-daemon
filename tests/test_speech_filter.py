"""Silence evidence belongs to one segment, and missing evidence deletes nothing."""

from __future__ import annotations

from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from omavoi import config
from omavoi.asr.api_whisper import ApiWhisperBackend
from omavoi.asr.base import Segment, Transcript
from omavoi.commands.health import _print_entry
from omavoi.pipeline import Pipeline
from omavoi.post import rules
from omavoi.window import Window


def _segment(text, probability=0.99, logprob=-1.5):
    return Segment(0, 1, text, avg_logprob=logprob, no_speech_prob=probability)


@pytest.mark.parametrize("probability,logprob", [
    (0.01, -0.2), (0.01, -2.0), (0.99, -0.1), (0.99, -1.0),
    (None, -2.0), (0.99, None), (None, None),
    (float("nan"), -2.0), (float("inf"), -2.0), (1.5, -2.0),
    (0.99, float("nan")), (0.99, float("-inf")),
])
def test_genuine_or_uncertain_known_phrase_is_not_deleted(probability, logprob):
    text = "谢谢观看。"
    out = rules.run(text, config.defaults(), segments=[_segment(text, probability, logprob)], quiet=True)
    assert out.text == text
    assert not out.rejected
    assert not out.changes


def test_text_only_backend_cannot_silently_delete_a_known_phrase():
    out = rules.run("谢谢观看。", config.defaults(), quiet=True)
    assert out.text == "谢谢观看。"
    assert not out.rejected


def test_one_silent_tail_does_not_discard_valid_speech():
    segments = [_segment("请保留正文。", 0.01, -0.2), _segment("谢谢观看。")]
    out = rules.run("请保留正文。谢谢观看。", config.defaults(), segments=segments)
    assert out.text == "请保留正文。"
    assert not out.rejected
    assert len(out.changes) == 1
    assert "segment 2" in out.changes[0]
    assert "no_speech_prob=0.99" in out.changes[0]


def test_silent_middle_keeps_both_sides_and_their_spacing():
    text = "First sentence. Thanks for watching! Last sentence."
    segments = [
        _segment("First sentence.", 0.01, -0.2),
        _segment(" Thanks for watching!"),
        _segment(" Last sentence.", 0.01, -0.2),
    ]
    out = rules.run(text, config.defaults(), segments=segments)
    assert out.text == "First sentence. Last sentence."
    assert not out.rejected


def test_high_probability_without_phrase_match_can_drop_only_its_segment():
    segments = [_segment("real words", 0.01, -0.1), _segment(" imagined ending")]
    out = rules.run("real words imagined ending", config.defaults(), segments=segments)
    assert out.text == "real words"
    assert "hallucination" not in out.changes[0]


def test_only_unknown_and_speech_segments_survive_a_silent_one():
    segments = [_segment("low volume", None, None), _segment(" Thank you.")]
    out = rules.run("low volume Thank you.", config.defaults(), segments=segments)
    assert out.text == "low volume"
    assert not out.rejected


def test_whole_take_rejection_requires_all_text_segments_to_be_silent():
    segments = [_segment("谢谢观看。"), _segment("Thanks for watching!")]
    out = rules.run("谢谢观看。Thanks for watching!", config.defaults(), segments=segments)
    assert not out.text
    assert "all text segments" in out.rejected
    assert "segment 1" in out.rejected and "segment 2" in out.rejected


def test_quiet_threshold_still_requires_low_segment_confidence():
    cfg = config.defaults()
    segment = _segment("谢谢观看。", 0.6, -1.5)
    assert rules.run(segment.text, cfg, segments=[segment]).text == segment.text
    assert rules.run(segment.text, cfg, segments=[segment], quiet=True).rejected
    segment.avg_logprob = -0.1
    assert rules.run(segment.text, cfg, segments=[segment], quiet=True).text == segment.text


@pytest.mark.parametrize("text,segments", [
    ("Different provider text.", [_segment("谢谢观看。")]),
    ("Keep this. 谢谢观看。", [_segment("谢谢观看。")]),
    ("谢谢观看。 Keep this.", [_segment("谢谢观看。")]),
    ("We fixed it.", [_segment("We fixed it")]),
])
def test_incomplete_or_mismatched_segment_coverage_keeps_the_transcript(text, segments):
    out = rules.run(text, config.defaults(), segments=segments)
    assert out.text == text
    assert not out.rejected


def test_disabling_thresholds_or_post_processing_keeps_even_silent_text():
    cfg = config.defaults()
    text = "谢谢观看。"
    segment = _segment(text)
    cfg["post"]["no_speech_threshold"] = 0
    cfg["post"]["quiet_no_speech_threshold"] = 0
    assert rules.run(text, cfg, segments=[segment], quiet=True).text == text
    cfg = config.defaults()
    cfg["post"]["enabled"] = False
    assert rules.run(text, cfg, segments=[segment]).text == text


def test_speech_filter_does_not_damage_internal_word_segments():
    segments = [_segment("mac", 0.01, -0.2), _segment("OS", 0.01, -0.2), _segment("谢谢观看。")]
    out = rules.run("macOS谢谢观看。", config.defaults(), segments=segments)
    assert out.text == "macOS"


def test_remote_backend_preserves_absent_null_and_measured_zero_confidence():
    backend = ApiWhisperBackend(config.defaults())
    payload = {"text": "谢谢观看。", "segments": [
        {"text": "missing"},
        {"text": "null", "avg_logprob": None, "no_speech_prob": None},
        {"text": "zero", "avg_logprob": 0, "no_speech_prob": 0},
    ]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload))
    with httpx.Client(transport=transport) as client:
        backend._client = client
        transcript = backend.transcribe(np.zeros(160, dtype=np.float32), 16000)
    assert [(s.avg_logprob, s.no_speech_prob) for s in transcript.segments] == [
        (None, None), (None, None), (0.0, 0.0)]


def test_missing_confidence_is_unknown_in_diagnostics(capsys):
    transcript = Transcript("谢谢观看。", segments=[Segment(0, 1, "谢谢观看。")])
    assert transcript.max_no_speech is None
    assert transcript.min_avg_logprob is None
    assert transcript.warnings() == []
    _print_entry({"text": transcript.text, "asr": transcript.as_dict()}, verbose=True)
    assert "n/a" in capsys.readouterr().out


def test_pipeline_passes_segment_evidence_instead_of_global_max(home):
    cfg = config.defaults()
    cfg["ui"]["notify"] = False
    cfg["modes"]["default"]["rules"]["names"] = False
    transcript = Transcript("真实内容。谢谢观看。", segments=[
        _segment("真实内容。", 0.01, -0.1), _segment("谢谢观看。"),
    ])
    backend = SimpleNamespace(transcribe=lambda *args, **kwargs: transcript)
    recorded = []
    history = SimpleNamespace(record=lambda entry, *args: recorded.append(entry))
    pipeline = Pipeline(cfg, backend, history=history)
    capture = SimpleNamespace(
        samples=np.zeros(16000, dtype=np.float32), rate=16000, seconds=1.0,
        peak_dbfs=-10.0, rms_dbfs=-20.0, preroll_seconds=0.1,
        tail_seconds=0.1, truncated=False,
    )
    entry = pipeline.process(capture, window=Window(), inject=False, forced_mode="default")
    assert entry["raw_text"] == "真实内容。谢谢观看。"
    assert entry["text"] == "真实内容。"
    assert entry["asr"]["segments"][1]["text"] == "谢谢观看。"
    assert not entry["rejected"]
    assert recorded == [entry]
