"""There is no clipped-onset warning, and that is the finding.

Two were tried. The first asked the transcript where its first segment began,
which is where whisper.cpp always begins one: it fired on 38 of 40 consecutive
takes, every one of which produced text perfectly well. The second compared
the take's first 120 ms against the take as a whole — better, and it survived
until a take picked up a video playing in the background. Continuous sound
makes the head match the whole whatever the pre-roll did, so it fired again,
correctly and uselessly.

Whether the onset survived is not inferable from the waveform once the room is
not quiet. So the measurement is recorded and no conclusion is drawn from it.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from omavoi import pipeline
from omavoi.asr.base import Segment, Transcript

RATE = 16000


def _noise(seconds: float, level: float) -> np.ndarray:
    rng = np.random.default_rng(0)
    return rng.normal(0, level, int(RATE * seconds)).astype(np.float32)


def _cap(samples):
    return SimpleNamespace(samples=samples, rate=RATE)


def test_the_head_level_is_measured():
    quiet = pipeline._head_level(_cap(_noise(3, 0.0008)))
    loud = pipeline._head_level(_cap(_noise(3, 0.2)))
    assert quiet is not None and loud is not None
    assert quiet < loud - 20, "the two should be tens of dB apart"


def test_unmeasurable_audio_gives_none():
    assert pipeline._head_level(_cap(_noise(0.05, 0.2))) is None
    assert pipeline._head_level(SimpleNamespace(samples=None, rate=RATE)) is None
    assert pipeline._head_level(SimpleNamespace(samples=_noise(3, 0.2), rate=0)) is None


def test_neither_onset_warning_exists_any_more():
    """The point of the file: no code draws a conclusion from the head level."""
    assert not hasattr(pipeline, "_opens_mid_speech")
    t = Transcript(text="x", segments=[Segment(start=0.0, end=1.0, text="x",
                                               avg_logprob=-0.1,
                                               no_speech_prob=0.0,
                                               compression_ratio=1.0,
                                               temperature=0.0)])
    assert not [w for w in t.warnings() if "clip" in w.lower()]
    assert not hasattr(t, "first_speech_at")


def test_a_take_with_continuous_background_would_have_tripped_it():
    """The case that killed the second attempt, kept so the reasoning is
    testable rather than a story in a comment."""
    continuous = _noise(3, 0.2)          # a video playing throughout
    head = pipeline._head_level(_cap(continuous))
    whole = 20.0 * np.log10(max(float(np.sqrt(np.mean(np.square(
        continuous.astype(np.float64))))), 1e-9))
    assert abs(head - whole) < 6.0, (
        "head and whole match under continuous sound, which is exactly why "
        "comparing them cannot answer whether the onset was clipped")
