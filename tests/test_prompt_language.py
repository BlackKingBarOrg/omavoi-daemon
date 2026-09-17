"""The prompt a new LLM step starts with follows the interface language.

It is filled in when the step is made, so a console switched to Chinese
afterwards kept every English prompt it had already written.

The rule that makes moving a stored prompt safe: one that still matches a
shipped default — in any of the eight languages — follows the interface
language; one that has been edited is the user's own instructions and is
never rewritten. A single character's difference is enough to make it theirs,
because the cost of guessing wrong is somebody's prompt silently replaced.

The interface language and not the speech language: this is a paragraph
someone reads and edits on the Modes tab, and "in its original language",
inside the prompt itself, is what keeps a Chinese prompt correct over an
English take.
"""
from __future__ import annotations

import pytest

from omavoi import config, i18n


def _prose_prompt():
    return str(config.load()["modes"]["prose"]["steps"][0]["prompt"])


def _set_prose(text):
    cfg = config.load()
    cfg["modes"]["prose"]["steps"][0]["prompt"] = text
    config.write(cfg)


# -- the translations themselves -------------------------------------------


@pytest.mark.parametrize("lang", i18n.LANGUAGES)
def test_every_language_has_a_prompt(home, lang):
    got = config.default_step_prompt(lang)
    assert got.strip(), lang
    if lang != "en":
        assert got != config.DEFAULT_STEP_PROMPT, f"{lang} is untranslated"


@pytest.mark.parametrize("lang", [x for x in i18n.LANGUAGES if x != "en"])
def test_a_translation_keeps_the_load_bearing_shape(home, lang):
    """Without the last sentence the model answers the dictation instead of
    editing it, and the answer is what gets typed. A translation that lost it
    would be worse than no translation."""
    got = config.default_step_prompt(lang)
    assert "\n" in got, "the instruction is two paragraphs in every language"
    # Two sentences of instruction after the newline, as in the English.
    tail = got.split("\n", 1)[1]
    assert len(tail) > 30, f"{lang}: the second half is too short to carry it"


def test_an_unset_language_is_english(home):
    assert config.default_step_prompt("") == config.DEFAULT_STEP_PROMPT
    assert config.default_step_prompt("en") == config.DEFAULT_STEP_PROMPT


def test_an_unknown_language_falls_through_to_english(home):
    assert config.default_step_prompt("xx") == config.DEFAULT_STEP_PROMPT


# -- what counts as still-a-default ----------------------------------------


@pytest.mark.parametrize("lang", ("", *i18n.LANGUAGES))
def test_a_shipped_prompt_in_any_language_is_recognised(home, lang):
    assert config.is_default_step_prompt(config.default_step_prompt(lang))


@pytest.mark.parametrize("text", [
    "",
    "please rewrite this",
    "请把下面这段话改写得更正式一点，只输出结果。",
])
def test_anything_else_is_not(home, text):
    assert not config.is_default_step_prompt(text)


def test_one_character_different_is_the_users(home):
    edited = config.default_step_prompt("zh").replace("。", "．", 1)
    assert not config.is_default_step_prompt(edited)


def test_surrounding_whitespace_does_not_make_it_the_users(home):
    """A text area can leave a trailing newline behind."""
    assert config.is_default_step_prompt("\n" + config.DEFAULT_STEP_PROMPT + "  \n")


# -- following a language change -------------------------------------------


def test_an_untouched_prompt_follows_the_language(home):
    config.write(config.defaults())
    assert _prose_prompt() == config.DEFAULT_STEP_PROMPT

    for lang in ("zh", "ja", "de", "fr", ""):
        config.set_path("ui.language", lang)
        assert _prose_prompt() == config.default_step_prompt(lang), lang


def test_an_edited_prompt_is_never_rewritten(home):
    config.write(config.defaults())
    mine = config.DEFAULT_STEP_PROMPT + "\nAlways use British spelling."
    _set_prose(mine)

    for lang in ("zh", "ja", "de", ""):
        config.set_path("ui.language", lang)
        assert _prose_prompt() == mine, f"it was rewritten under {lang!r}"


def test_a_prompt_written_by_hand_is_never_rewritten(home):
    config.write(config.defaults())
    hand = "请把下面这段话改写得更正式一点，只输出结果。"
    _set_prose(hand)
    for lang in ("en", "fr", "zh"):
        config.set_path("ui.language", lang)
        assert _prose_prompt() == hand, lang


def test_the_move_is_reported_not_silent(home):
    """Like the [llm.*] fold beside it: the config file changed under you."""
    cfg = config.defaults()
    cfg["ui"]["language"] = "zh"
    notes = config.follow_ui_language(cfg)
    assert notes, "a change to the config file must be reported"
    assert "prose" in notes[0] and "zh" in notes[0], notes


def test_nothing_to_do_reports_nothing(home):
    cfg = config.defaults()
    assert config.follow_ui_language(cfg) == [], "already English, already right"
    assert config.follow_ui_language(cfg) == [], "and it is not a ratchet"


def test_a_mode_with_no_steps_is_not_a_problem(home):
    cfg = config.defaults()
    cfg["modes"]["default"]["steps"] = []
    cfg["modes"]["broken"] = {"steps": ["not a dict"]}
    cfg["ui"]["language"] = "de"
    config.follow_ui_language(cfg)  # must not raise


# -- a new step gets the prompt in the right language -----------------------


@pytest.mark.parametrize("lang,other", [("zh", "把这段转写"), ("ja", "この書き起こし")])
def test_a_new_step_starts_in_the_interface_language(home, lang, other):
    import argparse

    from omavoi.commands import modes as modes_cmd

    config.write(config.defaults())
    config.set_path("ui.language", lang)
    code = modes_cmd.cmd_mode(argparse.Namespace(
        action="step", rest=["default", "add", "local"], json=False, force=False))
    assert code == 0
    step = config.load()["modes"]["default"]["steps"][-1]
    assert step["prompt"].startswith(other), step["prompt"][:40]
    assert step["prompt"] == config.default_step_prompt(lang)


def test_a_prompt_given_on_the_command_line_wins(home):
    import argparse

    from omavoi.commands import modes as modes_cmd

    config.write(config.defaults())
    config.set_path("ui.language", "zh")
    modes_cmd.cmd_mode(argparse.Namespace(
        action="step", rest=["default", "add", "local", "just", "do", "this"],
        json=False, force=False))
    assert config.load()["modes"]["default"]["steps"][-1]["prompt"] == "just do this"
