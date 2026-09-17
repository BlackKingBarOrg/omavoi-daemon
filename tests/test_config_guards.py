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
                "ui.language", "switching.mode",
                "ui.hud_size", "ui.hud_position", "inject.paste_method",
                "modes.default.paste_method", "speech.api.response_format",
                "audio.rate", "hotkey.force_mode"):
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
    "ui.hud_dwell", "ui.language", "switching.mode", "ui.hud_size", "ui.hud_position", "inject.paste_method",
    "modes.default.paste_method", "speech.api.response_format", "audio.rate",
    "hotkey.force_mode",
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


# -- the two that cannot use the tuple table -------------------------------


@pytest.mark.parametrize("value", ["DEBUG", "debug", "Info", "WARNING", "warn",
                                   "error", "CRITICAL", "fatal"])
def test_every_spelling_of_a_real_level_is_accepted(home, value):
    """setup_logging uppercases, so both cases genuinely work.

    Listing them in _legal_values would mean twenty entries in an error
    message; the check reads logging's own mapping instead.
    """
    assert settings._why_not_that_level("ui.log_level", value) == ""


@pytest.mark.parametrize("value", ["SHOUT", "verbose", "loud", "9", "",
                                   "NOTSET"])
def test_a_level_the_logger_would_ignore_is_refused(home, value):
    """`ui.log_level = SHOUT` was written, then silently became INFO.

    `getattr(logging, "SHOUT", logging.INFO)` is the fallback that made it
    quiet. NOTSET is refused too: it is a real attribute and it means "ask
    the parent", which is not a level anyone means to choose here.
    """
    why = settings._why_not_that_level("ui.log_level", value)
    assert why, f"{value!r} was accepted as a logging level"
    assert "logging level" in why


@pytest.mark.parametrize("value", ["", "auto", "en", "zh", "ja", "yue", "de"])
def test_a_language_code_is_accepted(home, value):
    assert settings._why_not_that_language("speech.language", value) == ""
    assert settings._why_not_that_language("modes.default.language", value) == ""


@pytest.mark.parametrize("value", ["chinese", "klingon", "zh-CN", "en_US",
                                   "ZH", "z", "中文"])
def test_a_language_name_or_locale_is_refused(home, value):
    """The realistic mistakes, which whisper answers with a 400 minutes later.

    A shape check and not a list of whisper's 99 codes: those are whisper's
    and a copy here would go stale. It lets through invented two-letter codes,
    which the engine itself then names — and catches every one of these.
    """
    why = settings._why_not_that_language("speech.language", value)
    assert why, f"{value!r} was accepted as a language"
    assert "ISO-639" in why


def test_the_language_check_ignores_keys_that_are_not_languages(home):
    """It matches `modes.*.language` by suffix, so it must not catch more."""
    for key in ("ui.language", "speech.backend", "modes.default.inject"):
        assert settings._why_not_that_language(key, "chinese") == "", key


# -- the model key reaches a shell -----------------------------------------


def test_speech_model_must_be_in_the_catalogue(home):
    """It is interpolated into the command `omavoi setup --run` shells out.

    `mode set <m> speech_model` has checked this since it was written and
    config.validate reports a bad one once per load — but `config set
    speech.model` took anything, and setup builds
    `omavoi model pull {key}` out of it for subprocess.call(shell=True).
    """
    bad = settings._why_not_that_speech_model
    assert bad("speech.model", "x; echo pwned")
    assert bad("speech.model", "not-a-model")
    assert bad("speech.model", "llm:qwen3-4b"), "an LLM is not a speech model"
    assert bad("speech.model", "ggml:large-v3") == ""
    # `model rm` and `mode set … speech_model` both take the bare name, so
    # this does too rather than being the one command that does not.
    # retire_cuda_engine rewrites it to ggml:large-v3 on the next load.
    assert bad("speech.model", "large-v3") == ""
    assert bad("hotkey.key", "anything") == "", "it answers for one key only"


def test_the_setup_command_survives_a_hand_edited_config(home):
    """Validation is the real fix; this is the one that holds when someone
    writes the key straight into config.toml."""
    import shlex

    from omavoi import config, setup

    cfg = config.load()
    cfg["speech"]["model"] = "x; echo pwned"
    pull = [s for s in setup.check(cfg).steps if s.key == "model"]
    assert pull, "no model step"
    command = pull[0].command
    assert "; echo pwned" not in command.replace(shlex.quote("x; echo pwned"), ""), command
    # The whole value must sit inside one shell word.
    assert shlex.split(command) == ["omavoi", "model", "pull", "x; echo pwned"], command
