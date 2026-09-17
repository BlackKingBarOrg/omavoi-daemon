"""The console speaks eight languages; the rules behind it spoke two.

Every piece of this was built and tuned against Chinese dictation, and it
shows in the defaults rather than in the code:

  llm.local.model            shipped as qwen3-8b, whose own note said it was
                             "a good default when the source language is
                             Chinese" — and modes.prose has a local LLM step,
                             so that is the model a fresh install ran
  the recommended LLM        the same one, a conditional recommendation used
                             as the unconditional one
  post.fillers_*             lists for zh and en, and nothing for the other
                             six interface languages
  post.hallucinations        seven English entries, five Chinese, one
                             Japanese, one Russian

None of that is wrong for the person who wrote it. It is wrong as a default
for someone who installs it and speaks German.
"""
from __future__ import annotations

import pytest

from omavoi import config, i18n, models
from omavoi.post.rules import drop_hallucinations, strip_fillers

# Chinese and Japanese fillers are bracketed by punctuation; Latin-script
# ones are matched on a word boundary. Thai and Vietnamese are deliberately
# empty — their fillers are overloaded particles and a wrong entry deletes a
# real word — so the key must exist and may be empty.
FILLER_KEYS = {
    "en": "fillers_en", "de": "fillers_de", "fr": "fillers_fr",
    "es": "fillers_es", "zh": "fillers_cjk", "ja": "fillers_ja",
    "th": "fillers_th", "vi": "fillers_vi",
}


def test_every_interface_language_has_a_filler_key():
    assert set(FILLER_KEYS) == set(i18n.LANGUAGES), "a language gained or lost"
    post = config.DEFAULTS["post"]
    for lang, key in FILLER_KEYS.items():
        assert key in post, f"{lang} has no filler list at all"


@pytest.mark.parametrize("lang", ["en", "de", "fr", "es", "zh", "ja"])
def test_the_languages_we_are_confident_about_have_entries(lang):
    """Thai and Vietnamese are excluded on purpose, and the comment in
    config.py says why: their fillers are ordinary words as often as they
    are hesitation."""
    assert config.DEFAULTS["post"][FILLER_KEYS[lang]], lang


def test_the_recommended_llm_does_not_assume_a_language():
    """A recommendation with "if you dictate in Chinese" on it is not the
    unconditional one."""
    recommended = [entry for entry in models.CATALOG
                   if entry.kind == models.LLM and "recommended" in entry.tags]
    assert len(recommended) == 1, [e.key for e in recommended]
    assert recommended[0].languages == "broad multilingual", recommended[0].note


def test_the_shipped_local_llm_is_the_recommended_one():
    """modes.prose has a local LLM step, so this is what a fresh install
    actually runs — it was the Chinese-first model."""
    shipped = config.DEFAULTS["llm"]["local"]["model"]
    recommended = next(entry.key for entry in models.CATALOG
                       if entry.kind == models.LLM and "recommended" in entry.tags)
    assert shipped == recommended, f"ships {shipped}, recommends {recommended}"


def test_the_shipped_llm_fits_beside_the_shipped_speech_model():
    """Both resident at once is the point: swapping per take costs seconds."""
    from omavoi import gpu

    speech = models.spec(config.DEFAULTS["speech"]["model"])
    llm = models.spec(config.DEFAULTS["llm"]["local"]["model"])
    together = gpu.needed_mb(speech.size_mb) + gpu.needed_mb(llm.size_mb)
    # 8 GB is the smallest card worth running this on at all.
    assert together < 8192, f"{together} MiB together"


@pytest.mark.parametrize("phrase,lang", [
    ("Subtitles by the Amara.org community", "en"),
    ("谢谢观看", "zh"),
    ("ご視聴ありがとうございました", "ja"),
    ("Untertitel der Amara.org-Community", "de"),
    ("Sous-titres réalisés par la communauté d'Amara.org", "fr"),
    ("Subtítulos realizados por la comunidad de Amara.org", "es"),
    ("Phụ đề được thực hiện bởi cộng đồng Amara.org", "vi"),
    ("คำบรรยายโดยชุมชน Amara.org", "th"),
])
def test_a_silent_take_is_caught_in_every_interface_language(phrase, lang):
    phrases = config.DEFAULTS["post"]["hallucinations"]
    assert drop_hallucinations(phrase, phrases) == "", lang
    # With punctuation, and inside a longer sentence, which must survive.
    assert drop_hallucinations(phrase + "!", phrases) == "", lang
    kept = f"I said {phrase} out loud on purpose"
    assert drop_hallucinations(kept, phrases) == kept, lang


def test_the_two_that_escaped_on_this_machine_are_covered():
    """Measured, not supposed: over 291 takes on a turbo model these two
    reached the window, because a turbo model's no_speech_prob is 0 and the
    phrase list was the only guard left."""
    phrases = config.DEFAULTS["post"]["hallucinations"]
    for phrase in ("字幕志愿者 李宗盛", "谢谢大家!"):
        assert drop_hallucinations(phrase, phrases) == "", phrase


def test_a_filler_list_holds_nothing_that_is_a_word():
    """The spaced lists are removed wherever they appear as a word, so a real
    word on one would be deleted out of the middle of a sentence."""
    post = config.DEFAULTS["post"]
    spaced = [f for key in ("fillers_en", "fillers_de", "fillers_fr",
                            "fillers_es", "fillers_vi")
              for f in post[key]]
    # A sentence built around each filler must lose only the filler.
    for filler in spaced:
        text = f"the {filler} word stays"
        out = strip_fillers(text, [], spaced)
        assert "word stays" in out, f"{filler!r} ate the sentence: {out!r}"


def test_every_model_says_which_languages_it_is_for():
    """The field was on every entry from the start and empty on all seven
    speech models — and displayed nowhere, for either family.

    It matters most for the thing it would have said: the default speech
    model is a distillation and is not even across languages.
    """
    for entry in models.CATALOG:
        assert entry.languages, f"{entry.key} does not say"


def test_the_payload_carries_it():
    """Populating a field nobody reads would have been the same as leaving
    it empty."""
    import inspect

    from omavoi.commands import catalogue

    src = inspect.getsource(catalogue.cmd_model)
    assert '"languages": spec.languages' in src, "the console cannot see it"
