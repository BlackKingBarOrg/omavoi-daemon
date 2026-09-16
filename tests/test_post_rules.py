"""The text rules, which had no tests at all.

This is the module that makes omavoi more than a transcribe-and-paste script,
and it is 300 lines of pure functions — the easiest thing in the repo to test
and the most expensive to get wrong, because every one of them rewrites what
the user said, into the window the user is looking at, silently.

Two bugs were sitting in it, both of the same shape as everything else this
project keeps finding: a step that ran when it had no business running.

  1. `_SENTENCE_SPLIT` consumed the whitespace it split on, and both callers
     rejoin with "". So every multi-sentence Latin-script take lost the space
     after every sentence: "Hello there. How are you?" came out
     "Hello there.How are you?" — on every mode, because both rules that use
     it are on by default. CJK hid it for as long as it lasted, because 。
     carries its own spacing and there was no space there to lose.

  2. `strip_fillers` tidied up after itself whether or not it had removed
     anything: it stripped leading punctuation and recapitalised the first
     letter unconditionally. With fillers on in all four shipped modes,
     `getUserName` became `GetUserName`, `.gitignore` became `Gitignore` and
     `./configure` became `/configure`.

Both were reported in `omavoi last` as a change by a rule that had done
nothing — "hallucinations" for the first, "fillers" for the second — which is
the part that made them hard to see.
"""
from __future__ import annotations

import pytest

from omavoi import config
from omavoi.post import rules as R

ZH = ["嗯", "呃", "那个", "就是", "然后"]
EN = ["um", "uh", "like", "you know"]


# -- the two bugs ----------------------------------------------------------


@pytest.mark.parametrize("text", [
    "Hello there. How are you?",
    "It broke. I fixed it. Then it broke again.",
    "First point; second point.",
    "Wait! What happened?",
    "Mixed 中文. And English.",
    "3.14159 is pi.",
    "Dr. Smith went to Washington.",
])
def test_sentence_punctuation_keeps_the_space_after_it(text):
    """Neither rule may touch text that has nothing for it to do."""
    assert R.drop_hallucinations(text, ["谢谢观看"]) == text
    assert R.dedupe_sentences(text) == text


@pytest.mark.parametrize("text", [
    "hello world", "getUserName", ".gitignore", "./configure",
    "...and then it broke", "def main():", "git commit -m fix",
    "--no-color", "npm run dev", "i said no",
])
def test_a_take_with_no_filler_in_it_is_returned_unchanged(text):
    assert R.strip_fillers(text, ZH, EN) == text


def test_leading_whitespace_alone_is_not_a_removal():
    """`text != before` after a .strip() is also true of a leading space."""
    assert R.strip_fillers("  leading space kept lower", ZH, EN) == (
        "leading space kept lower")


# -- and the work they are actually for ------------------------------------


@pytest.mark.parametrize("text,want", [
    ("um, hello world", "Hello world"),
    ("Um, hello world", "Hello world"),
    ("uh like you know it broke", "It broke"),
    ("嗯，我想说的是", "我想说的是"),
    ("嗯，那个，我想说", "我想说"),
])
def test_a_filler_at_the_front_goes_and_the_sentence_is_recapitalised(text, want):
    assert R.strip_fillers(text, ZH, EN) == want


def test_a_filler_from_the_middle_does_not_recapitalise():
    """The sentence never lost its first letter, so it keeps its own case."""
    assert R.strip_fillers("it was, um, broken", ZH, EN) == "it was, broken"


def test_a_filler_that_is_a_real_word_survives():
    """那个 is a filler in 那个，我想说 and a demonstrative in 那个函数."""
    assert R.strip_fillers("那个函数有问题", ZH, EN) == "那个函数有问题"


@pytest.mark.parametrize("text,want", [
    ("好的。好的。好的。", "好的。"),
    ("OK. OK. OK.", "OK."),
    ("Yes! Yes!", "Yes!"),
])
def test_the_repeat_loop_still_collapses(text, want):
    assert R.dedupe_sentences(text) == want


def test_a_hallucination_goes_only_when_it_is_the_whole_sentence():
    phrases = ["谢谢观看", "Thanks for watching."]
    assert R.drop_hallucinations("谢谢观看", phrases) == ""
    assert R.drop_hallucinations("谢谢观看。", phrases) == ""
    assert R.drop_hallucinations("真的结束了。谢谢观看。", phrases) == "真的结束了。"
    # Said inside a longer sentence, it is something the speaker meant.
    kept = "我说的是谢谢观看这四个字"
    assert R.drop_hallucinations(kept, phrases) == kept


# -- boundaries, spacing, dictionary, punctuation --------------------------


@pytest.mark.parametrize("text,want", [
    # A pause is a comma, not a full stop: whisper cuts where the speaker
    # breathed, and it writes the full stop itself where there was one.
    ("我今天想说的是\n这个功能很好", "我今天想说的是，这个功能很好"),
    ("one line\ntwo line", "one line, two line"),
    # whisper.cpp writes subtitle-style dashes at the head of a cue.
    ("- subtitle dash\n- another cue", "subtitle dash, another cue"),
    # Already terminated, so nothing is added and the two are set tight.
    ("已经结束了。\n下一句", "已经结束了。下一句"),
    ("trailing\n\n\n", "trailing"),
])
def test_segment_breaks_become_punctuation(text, want):
    assert R.normalise_boundaries(text) == want


def test_newlines_can_be_kept():
    text = "one\ntwo"
    assert R.normalise_boundaries(text, newlines="keep") == text


@pytest.mark.parametrize("text,want", [
    ("用Python写", "用 Python 写"),
    ("GitHub上的repo", "GitHub 上的 repo"),
    ("已经结束", "已经结束"),
    ("all latin", "all latin"),
])
def test_cjk_latin_spacing(text, want):
    assert R.cjk_latin_spacing(text) == want


def test_the_dictionary_takes_the_longest_key_first():
    """Or a prefix wins and the longer rule never fires."""
    d = {"hyper": "HYPER", "hyper land": "Hyprland"}
    out, hits = R.apply_dictionary("i use hyper land", d)
    assert out == "i use Hyprland", hits


def test_an_ascii_dictionary_key_needs_a_word_boundary():
    """Or `github` rewrites the middle of `githubbing`."""
    out, hits = R.apply_dictionary("a githubbing thing", {"github": "GitHub"})
    assert out == "a githubbing thing"
    assert hits == []


def test_a_cjk_dictionary_key_matches_bare():
    r"""\b means nothing next to a CJK character, so the key matches as-is."""
    out, hits = R.apply_dictionary("用派森写的", {"派森": "Python"})
    assert out == "用Python写的"
    assert hits == ["派森→Python×1"]


def test_a_backslash_in_a_replacement_is_a_backslash():
    """re.sub would read it as a group reference."""
    out, _ = R.apply_dictionary("see path", {"path": r"C:\Users"})
    assert out == r"see C:\Users"


@pytest.mark.parametrize("text,want", [
    ("ls -la.", "ls -la"),
    ("really?", "really"),
    ("wait...", "wait"),
    ("no punctuation here", "no punctuation here"),
])
def test_strip_drops_the_trailing_punctuation(text, want):
    assert R.apply_punctuation_policy(text, "strip") == want


def test_keep_leaves_it_alone():
    assert R.apply_punctuation_policy("ls -la.", "keep") == "ls -la."


# -- run() ------------------------------------------------------------------


def _ctx(mode="default"):
    cfg = config.load()
    return cfg, R.Context(rules=dict(cfg["modes"][mode]["rules"]))


@pytest.mark.parametrize("text", [
    "Hello there. How are you?",
    "getUserName",
    ".gitignore",
    "这是第一句。这是第二句。",
    "git commit -m fix",
])
def test_the_whole_pipeline_leaves_clean_text_alone(home, text):
    """And reports no change, because none was made."""
    cfg, ctx = _ctx()
    out = R.run(text, cfg, ctx)
    assert out.text == text, out.changes
    assert out.changes == [], "a rule claimed a change it did not make"
    assert not out.changed


def test_an_empty_take_is_rejected(home):
    cfg, ctx = _ctx()
    assert R.run("   ", cfg, ctx).rejected == "empty"


def test_a_take_that_is_only_a_hallucination_is_rejected(home):
    cfg, ctx = _ctx()
    out = R.run("谢谢观看", cfg, ctx)
    assert out.text == ""
    assert "hallucination" in out.rejected


def test_the_models_own_verdict_outranks_the_string_matching(home):
    """no_speech_prob is the one number that knows whether anything was said."""
    cfg, ctx = _ctx()
    out = R.run("something", cfg, ctx, max_no_speech=0.99)
    assert out.text == ""
    assert "no_speech_prob" in out.rejected


def test_post_processing_can_be_switched_off_entirely(home):
    cfg, ctx = _ctx()
    cfg["post"]["enabled"] = False
    text = "um, hello. hello."
    assert R.run(text, cfg, ctx).text == text
