"""Deterministic cleanup of raw ASR output. No model, no added latency.

Every rule reports whether it changed anything, so `omavoi last` can show
exactly which step turned hyperland into Hyprland — a prompt cannot show you that.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from ..asr.base import Segment

# CJK ideographs, kana, hangul — everything that wants a space beside Latin.
_CJK = r"㐀-䶿一-鿿぀-ゟ゠-ヿ가-힯"
_LATIN = r"A-Za-z0-9"

_CJK_THEN_LATIN = re.compile(rf"([{_CJK}])([{_LATIN}])")
_LATIN_THEN_CJK = re.compile(rf"([{_LATIN}])([{_CJK}])")
# Zero-width, so the whitespace between sentences belongs to the chunk after
# it and "".join puts the text back exactly as it came.
#
# It used to be `(?<=[...])\s*`, which consumed that whitespace — and both
# callers rejoin with "". So every multi-sentence Latin-script take came out
# with the spaces gone:
#
#   "Hello there. How are you?"  ->  "Hello there.How are you?"
#   "Wait! What happened?"       ->  "Wait!What happened?"
#
# on every mode, because both rules that use this run by default. CJK hid it:
# 。 carries its own spacing, so there was no space there to lose. And the
# change was reported as "hallucinations", naming a rule that had dropped
# nothing — which is what made it hard to see in `omavoi last`.
_SENTENCE_SPLIT = re.compile(r"(?<=[。．.！!？?；;])(?!$)")
_TRAILING_PUNCT = re.compile(r"[。．.，,、；;：:！!？?\s]+$")
_WS = re.compile(r"[^\S\n]{2,}")
# Hangul uses word spaces even though it shares CJK/Latin spacing rules above.
_UNSPACED_CHAR = re.compile(r"[㐀-䶿一-鿿぀-ゟ゠-ヿ]")


def normalise_boundaries(text: str, *, newlines: str = "space",
                         add_punctuation: bool = False) -> str:
    """Fold text lines without inventing punctuation or deleting content.

    A subtitle boundary can fall inside a word: it is not evidence of a
    pause. ASR adapters must preserve token spacing when joining segments;
    once that spacing has been stripped, this generic text rule cannot tell
    a split word from two words. Real line breaks follow the selected policy.

    add_punctuation is accepted for compatibility only. Even when True, a
    line break alone cannot justify inserting a comma.
    """
    if newlines == "keep" or "\n" not in text:
        return text

    chunks = [c.strip() for c in text.split("\n") if c.strip()]
    if not chunks:
        return ""

    out = ""
    for chunk in chunks:
        if not out:
            out = chunk
            continue
        tail, head = out[-1], chunk[0]
        # Full-width punctuation carries its own trailing space in the glyph,
        # so anything after it is set tight — including a Latin word.
        # Two CJK chunks also butt up. Everything else reads as two words.
        tight = (tail in "，。、；：！？…"
                 or (_UNSPACED_CHAR.search(tail) and _UNSPACED_CHAR.search(head)))
        out += ("" if tight else " ") + chunk
    return out


@dataclass(slots=True)
class Context:
    """All the rules are allowed to know about where the text is headed.

    `rules` comes from the resolved mode, not from the global config: which
    cleanups run is a per-mode decision, while the word lists behind them
    are shared.
    """

    window_class: str = ""
    window_title: str = ""
    rules: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class PostResult:
    text: str
    raw: str
    changes: list[str] = field(default_factory=list)
    rejected: str = ""  # non-empty means the whole transcript was dropped

    @property
    def changed(self) -> bool:
        return self.text != self.raw

    def as_dict(self) -> dict[str, Any]:
        return {"changes": self.changes, "rejected": self.rejected, "changed": self.changed}


def _norm(text: str) -> str:
    """Fold for comparison only: drop punctuation, spaces, case, width."""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"[\s\W_]+", "", folded, flags=re.UNICODE)


def drop_hallucinations(text: str, phrases: list[str], *, silence_evidence: bool = False) -> str:
    """Match stock phrases only after the caller establishes silence.

    Every phrase can also be genuine dictation, even as a complete sentence.
    Text alone cannot distinguish that from a hallucination.
    """
    if not silence_evidence:
        return text
    targets = {_norm(p) for p in phrases if p.strip()}
    if not text or not targets:
        return text
    # The whole take first, because a phrase with a full stop inside it never
    # matches a sentence chunk: the split happens at that full stop. Every
    # Amara subtitle credit contains "Amara.org", so
    # "Subtitles by the Amara.org community" split into "Subtitles by the
    # Amara." and "org community" and neither was the phrase. Three of the
    # fourteen shipped phrases had never once fired, including the English
    # one — and the credits are the commonest thing whisper writes for
    # silence.
    if _norm(text) in targets:
        return ""
    kept = [s for s in _SENTENCE_SPLIT.split(text) if s.strip() and _norm(s) not in targets]
    return "".join(kept).strip()


def dedupe_sentences(text: str) -> str:
    """Collapse the repeat loops Whisper falls into ("好的。好的。好的。")."""
    out: list[str] = []
    for part in _SENTENCE_SPLIT.split(text):
        if not part.strip():
            continue
        if out and _norm(part) and _norm(part) == _norm(out[-1]):
            continue
        out.append(part)
    return "".join(out)


def strip_fillers(text: str, unspaced: list[str], spaced: list[str]) -> str:
    """Drop hesitation sounds, but only where they stand alone.

    Two lists because two scripts behave differently, not because two
    languages do. `unspaced` is for text written without spaces — Chinese,
    Japanese, Thai — where there is no word boundary to anchor on, so a
    filler goes only when a sentence boundary or a comma brackets it: that
    is what lets 那個 be a filler in 那个，我想说 and a demonstrative in
    那个函数. `spaced` is matched on \b wherever it appears, which is
    stricter about what may go on it: a real word there would be deleted out
    of the middle of a sentence.
    """
    if not text:
        return text

    before = text

    zh = [re.escape(f) for f in unspaced if f.strip()]
    if zh:
        boundary = r"[，,。．.！!？?；;：:\s]"
        pattern = re.compile(rf"(^|(?<={boundary}))\s*(?:{'|'.join(zh)})\s*(?={boundary}|$)")
        for _ in range(3):  # 嗯，嗯，那个， needs more than one pass
            new = pattern.sub("", text)
            if new == text:
                break
            text = new

    en = [re.escape(f) for f in spaced if f.strip()]
    if en:
        text = re.sub(rf"\b(?:{'|'.join(en)})\b[,\s]*", "", text, flags=re.IGNORECASE)

    # Everything below tidies up after a removal, and everything below used to
    # run whether or not one happened — so on a take with no filler in it at
    # all, and every mode has this rule on:
    #
    #   getUserName   -> GetUserName      an identifier, recapitalised
    #   .gitignore    -> Gitignore        the dot gone and then recapitalised
    #   ./configure   -> /configure       the dot gone
    #   def main():   -> Def main():
    #   hello world   -> Hello world
    #
    # and `omavoi last` labelled the change "fillers", which is the part that
    # made it hard to see: the label named a rule that had done nothing.
    # Decided here, before any whitespace tidying: `text != before` after a
    # .strip() is also true of a take that merely arrived with a leading
    # space, and that is not a removal.
    removed = text != before

    if removed:
        # The punctuation the filler was sitting in front of: "Um, hello"
        # leaves ", hello", and "嗯，我" leaves "，我".
        text = re.sub(r"^[，,、。．.；;：:\s]+", "", text)
        # And the comma it was sitting between: 嗯，那个，我 leaves ，，我.
        text = re.sub(r"([，,、])\s*(?=[，,。．.！!？?])", "", text)

    # Whitespace hygiene, which is true of any transcript: whisper does not
    # produce aligned columns, and a leading space is never meant.
    text = _WS.sub(" ", text).strip()

    # Removing a leading "Um, " leaves the sentence starting lowercase — so
    # only when the removal was at the front. "it was, um, broken" lost a
    # filler from the middle and has no business being recapitalised, and a
    # take that simply begins in lower case is how people say identifiers,
    # flags and paths. Comparing the first character is deliberately
    # conservative: "um, umbrella" keeps its lower case, which is the
    # direction to be wrong in.
    head_changed = removed and before.strip()[:1] != text[:1]
    if head_changed and text[:1].islower() and text[:1].isascii():
        text = text[0].upper() + text[1:]
    return text


def apply_dictionary(text: str, dictionary: dict[str, str]) -> tuple[str, list[str]]:
    """Fix jargon the model mis-hears. Longest key first so prefixes don't win."""
    if not text or not dictionary:
        return text, []
    hits: list[str] = []
    for src in sorted(dictionary, key=len, reverse=True):
        dst = dictionary[src]
        if not src:
            continue
        # \b only means anything for ASCII keys; CJK keys match bare.
        ascii_key = re.match(r"^[\w\s]+$", src, re.ASCII) is not None
        pattern = rf"\b{re.escape(src)}\b" if ascii_key else re.escape(src)
        new, n = re.subn(pattern, dst.replace("\\", r"\\"), text, flags=re.IGNORECASE)
        if n:
            hits.append(f"{src}→{dst}×{n}")
            text = new
    return text, hits


def cjk_latin_spacing(text: str) -> str:
    """Insert the space CJK typography wants around Latin runs."""
    return _LATIN_THEN_CJK.sub(r"\1 \2", _CJK_THEN_LATIN.sub(r"\1 \2", text))


def apply_punctuation_policy(text: str, policy: str) -> str:
    """`strip` drops the trailing full stop — terminals don't want one."""
    return _TRAILING_PUNCT.sub("", text) if policy == "strip" else text


def _filter_silent_segments(
    text: str,
    segments: Sequence[Segment],
    post: dict[str, Any],
    rules: dict[str, Any],
    *,
    quiet: bool,
) -> tuple[str, list[str]]:
    """Remove only the text covered by corroborated segment evidence.

    The no-speech probability is not a verdict: confidently decoded text
    must survive it. Both measurements must be present, finite and agree.
    The -1.0 log-probability floor is also used by Transcript.warnings().

    Backends may return a full transcript that differs from their segment
    text. Require exact, ordered text coverage (apart from whitespace) before
    assigning segment scores to any characters. On a mismatch, keep it all.
    """
    spans: list[tuple[int, int, int, Segment]] = []
    cursor = 0
    for index, segment in enumerate(segments, 1):
        part = segment.text.strip()
        if not part:
            continue
        start = text.find(part, cursor)
        if start < 0 or text[cursor:start].strip():
            return text, []
        end = start + len(part)
        spans.append((start, end, index, segment))
        cursor = end
    if text[cursor:].strip():
        return text, []

    thresholds = [float(post.get("no_speech_threshold", 0.8))]
    if quiet:
        thresholds.append(float(post.get("quiet_no_speech_threshold", 0.5)))
    active = [value for value in thresholds if math.isfinite(value) and value > 0]
    if not active:
        return text, []
    threshold = min(active)

    chunks: list[str] = []
    changes: list[str] = []
    cursor = 0
    for start, end, index, segment in spans:
        probability = segment.no_speech_prob
        logprob = segment.avg_logprob
        silent = (
            probability is not None and math.isfinite(probability)
            and threshold <= probability <= 1.0
            and logprob is not None and math.isfinite(logprob) and logprob < -1.0
        )
        chunks.append(text[cursor:start])
        if silent:
            # The phrase list describes a rejected segment; it cannot grant
            # permission to delete speech. Silence evidence applies to this
            # segment only, never the rest of the recording.
            known_phrase = rules.get("hallucinations", True) and drop_hallucinations(
                segment.text.strip(), post.get("hallucinations", []), silence_evidence=True,
            ) != segment.text.strip()
            reason = "hallucination / silence" if known_phrase else "silence"
            changes.append(
                f"segment {index}: {reason}, no_speech_prob={probability:.2f}, avg_logprob={logprob:.2f}"
            )
        else:
            chunks.append(text[start:end])
        cursor = end
    chunks.append(text[cursor:])
    return "".join(chunks).strip(), changes


def run(
    text: str,
    cfg: dict[str, Any],
    ctx: Context | None = None,
    *,
    segments: Sequence[Segment] = (),
    quiet: bool = False,
) -> PostResult:
    raw = text
    post = cfg["post"]
    result = PostResult(text=text.strip(), raw=raw)

    if not post.get("enabled", True):
        return result
    if not result.text:
        result.rejected = "empty"
        return result

    ctx = ctx or Context()
    rules = ctx.rules
    step, removed = _filter_silent_segments(result.text, segments, post, rules, quiet=quiet)
    result.changes.extend(removed)
    if not step:
        result.text = ""
        result.rejected = "all text segments rejected: " + "; ".join(removed)
        return result

    after = normalise_boundaries(
        step,
        newlines=str(post.get("newlines", "space")),
    )
    if after != step:
        result.changes.append("line breaks folded")
        step = after

    if rules.get("hallucinations", True):
        after = dedupe_sentences(step)
        if after != step:
            result.changes.append("deduped")
            step = after

    if rules.get("fillers", True):
        # Gathered by script, not by language: strip_fillers matches the
        # first set between punctuation (no word boundaries to use) and the
        # second on \b. Whatever language someone is dictating in, only the
        # lists that cannot fire on it are inert.
        unspaced = [f for key in ("fillers_cjk", "fillers_ja", "fillers_th")
                    for f in post.get(key, []) or []]
        spaced = [f for key in ("fillers_en", "fillers_de", "fillers_fr",
                                "fillers_es", "fillers_vi")
                  for f in post.get(key, []) or []]
        after = strip_fillers(step, unspaced, spaced)
        if after != step:
            result.changes.append("fillers")
            step = after

    spacing = rules.get("cjk_spacing", True)
    spaced = False
    if spacing:
        # Space *before* the dictionary runs, or "the 派森" would arrive as
        # "the派森" and come out "thePython": the CJK is already gone by the
        # time the spacing rule would have seen it.
        after = cjk_latin_spacing(step)
        if after != step:
            spaced = True
            step = after

    if rules.get("dictionary", True):
        after, hits = apply_dictionary(step, cfg.get("dictionary", {}).get("rules", {}))
        if hits:
            result.changes.append("dictionary: " + ", ".join(hits))
            step = after

    if spacing:
        after = cjk_latin_spacing(step)
        if after != step or spaced:
            result.changes.append("cjk spacing")
            step = after

    policy = str(rules.get("punctuation", "keep"))
    after = apply_punctuation_policy(step, policy)
    if after != step:
        result.changes.append(f"punctuation={policy}")
        step = after

    if str(post.get("newlines", "space")) != "keep":
        step = re.sub(r"\s*\n\s*", " ", step)
    result.text = _WS.sub(" ", step).strip()
    if not result.text and raw.strip():
        result.rejected = "nothing left after post-processing"
    return result
