"""The clipped-onset warning has to measure the onset.

It used to ask the transcript where its first segment began — which is where
whisper.cpp always begins one. Across forty consecutive takes it read 0.00 on
thirty-eight, and all thirty-eight produced text perfectly well. A warning
that fires on healthy takes teaches you to ignore it, and then it is not
there the once it matters.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from omavoi.pipeline import _opens_mid_speech

RATE = 16000


def _speech(seconds: float, level: float = 0.2) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, level, int(RATE * seconds)).astype(np.float32)


def _cap(samples):
    return SimpleNamespace(samples=samples, rate=RATE)


def test_a_quiet_lead_in_is_the_preroll_working():
    room = _speech(0.6, level=0.0008)
    assert _opens_mid_speech(_cap(np.concatenate([room, _speech(3)]))) == ""


def test_opening_at_speech_level_warns():
    why = _opens_mid_speech(_cap(_speech(3)))
    assert why
    assert "first syllable" in why
    assert "preroll" in why, "the warning has to name the knob that fixes it"


def test_too_short_to_judge_says_nothing():
    assert _opens_mid_speech(_cap(_speech(0.05))) == ""


def test_no_audio_says_nothing():
    assert _opens_mid_speech(SimpleNamespace(samples=None, rate=RATE)) == ""
    assert _opens_mid_speech(SimpleNamespace(samples=_speech(3), rate=0)) == ""


def test_silence_throughout_does_not_warn():
    """A silent take is caught by the quiet-input check; this one must not
    also fire, or one problem gets two voices."""
    assert _opens_mid_speech(_cap(_speech(3, level=0.0005))) == ""


def test_the_transcript_no_longer_warns_about_its_own_segmentation():
    from omavoi.asr.base import Segment, Transcript
    t = Transcript(text="x", segments=[Segment(start=0.0, end=1.0, text="x",
                                               avg_logprob=-0.1,
                                               no_speech_prob=0.0,
                                               compression_ratio=1.0,
                                               temperature=0.0)])
    assert t.first_speech_at == 0.0
    assert not [w for w in t.warnings() if "clipped" in w]
