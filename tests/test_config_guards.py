"""`config set` must refuse what it cannot use — and accept what it can.

Both halves matter equally. A guard that refuses a valid value is worse than
no guard, and this project has shipped one: a check meant for local weights
was applied to every llm.*.model and made the remote API impossible to
configure.
"""
from __future__ import annotations

import pytest

from omavoi import config
from omavoi.commands import settings


def test_every_legal_value_is_accepted(home):
    """No key may refuse a value its own table calls legal."""
    cfg = config.load()
    checked = 0
    for key in ("speech.backend", "llm.api.backend", "hotkey.mode",
                "inject.method", "modes.default.inject",
                "modes.default.rules.punctuation", "ui.hud_dwell",
                "ui.language", "speech.local_whisper.device", "switching.mode"):
        legal, what = settings._legal_values(key, cfg)
        assert legal, f"{key} has no table any more"
        assert what
        for value in legal:
            why = settings._why_not_that_choice(key, value)
            assert why == "", f"{key}={value!r} is legal but refused: {why}"
            checked += 1
    assert checked > 30


@pytest.mark.parametrize("key", [
    "speech.backend", "llm.api.backend", "hotkey.mode", "inject.method",
    "modes.default.inject", "modes.default.rules.punctuation",
    "ui.hud_dwell", "ui.language", "speech.local_whisper.device",
    "switching.mode",
])
def test_a_value_outside_the_set_is_refused(home, key):
    why = settings._why_not_that_choice(key, "definitely-not-a-real-value")
    assert why, f"{key} accepted a value that is not in its set"
    # The message has to name the alternatives, or it is just a refusal.
    assert "One of:" in why


def test_keys_without_a_fixed_set_are_left_alone(home):
    """A free-text key must not be caught by the table."""
    cfg = config.load()
    for key in ("llm.api.model", "llm.api.base_url", "modes.default.prompt",
                "speech.api.base_url"):
        assert settings._legal_values(key, cfg) == ((), "")
        assert settings._why_not_that_choice(key, "anything at all") == ""


def test_switching_mode_follows_the_modes_that_exist(home):
    """The set is read from the config, not from a list that can go stale."""
    cfg = config.load()
    legal, _ = settings._legal_values("switching.mode", cfg)
    assert set(legal) == set(cfg["modes"])
    assert "default" in legal


def test_numbers_outside_their_range_are_refused(home):
    """`audio.preroll_seconds 99` was accepted — ninety-nine seconds of
    pre-roll from a ring buffer that does not hold it — and `warn_rms_dbfs
    500`, a positive number for a quantity that is negative by definition."""
    for key, bad in (("audio.preroll_seconds", "99"),
                     ("audio.warn_rms_dbfs", "500"),
                     ("history.keep_audio", "-5"),
                     ("audio.max_seconds", "1"),
                     ("hotkey.rescan_seconds", "0")):
        why = settings._why_not_that_number(key, bad)
        assert why, f"{key}={bad} was accepted"
        assert key in why, "the message has to name the key it is about"


def test_a_number_that_is_not_one_is_refused(home):
    why = settings._why_not_that_number("audio.max_seconds", "abc")
    assert "not a number" in why


def test_the_shipped_defaults_are_all_inside_their_own_ranges(home):
    """A range that refuses the value the program ships with is the guard
    this project has already shipped once."""
    cfg = config.load()
    for key in settings._RANGES:
        node, parts = cfg, key.split(".")
        for p in parts[:-1]:
            node = node.get(p, {})
        value = node.get(parts[-1])
        assert value is not None, f"{key} is not in the shipped config"
        why = settings._why_not_that_number(key, str(value))
        assert why == "", f"the default {value} for {key} is refused: {why}"
