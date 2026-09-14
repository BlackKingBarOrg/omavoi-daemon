"""Proper nouns, matched by sound rather than by spelling.

A heard->meant rule needs you to know what the model got wrong. For a name
you do not, and for Chinese you never will: 李文渊 comes back as 李文远,
李闻渊 or 里闻鸢 — an open set of homophones. So a name is written once, in
its correct form, and works two ways:

  seeding   the most-used names go into the decoder prompt, which is what
            makes the model produce them in the first place
  matching  everything else is recovered afterwards by comparing sound —
            pinyin for CJK, a consonant skeleton for Latin — so every
            homophone spelling collapses back to the one entry

Matching is the only thing in this program that can damage text which was
already correct, so a new name stays inert until its dry run is accepted.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

_CJK = re.compile(r"[㐀-䶿一-鿿]")
_WORD = re.compile(r"[A-Za-z][A-Za-z'’\-]*")

# Consonant classes that survive being misheard. Order matters: digraphs first.
_PHON = [
    ("PH", "F"), ("GH", "F"), ("CK", "K"), ("CH", "X"), ("SH", "X"),
    ("TH", "0"), ("QU", "KW"), ("WR", "R"), ("KN", "N"), ("PS", "S"),
    ("C", "K"), ("Q", "K"), ("X", "KS"), ("Z", "S"), ("V", "F"),
    ("J", "Y"), ("W", "V"), ("G", "K"),
]
_VOWELS = set("AEIOUY")


def phonetic_key(text: str) -> str:
    """A rough consonant skeleton: enough for 'wee type' to reach 'wtype'."""
    upper = re.sub(r"[^A-Za-z0-9]+", "", text.upper())
    if not upper:
        return ""
    out = upper
    for src, dst in _PHON:
        out = out.replace(src, dst)
    head, rest = out[0], out[1:]
    rest = "".join(ch for ch in rest if ch not in _VOWELS)
    key = head + rest
    # Collapse doubles: "MMM" and "M" sound the same.
    return re.sub(r"(.)\1+", r"\1", key)


def pinyin_key(text: str, *, tones: bool = False) -> str:
    try:
        from pypinyin import Style, lazy_pinyin
    except ImportError:  # pragma: no cover - optional at runtime
        return ""
    style = Style.TONE3 if tones else Style.NORMAL
    return " ".join(lazy_pinyin(text, style=style, errors="ignore"))


def _ratio(a: str, b: str) -> float:
    """Similarity in [0,1]. Cheap Levenshtein over short keys."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


@dataclass(slots=True)
class NameEntry:
    name: str
    group: str = ""
    # pinyin | phonetic | exact — inferred from the script when unset.
    match: str = ""
    seed: bool = True
    # Sound matching stays off until a dry run has been looked at.
    enabled: bool = False
    modes: list[str] = field(default_factory=list)
    hits: int = 0

    @property
    def is_cjk(self) -> bool:
        return bool(_CJK.search(self.name))

    def resolved_match(self) -> str:
        if self.match:
            return self.match
        return "pinyin" if self.is_cjk else "phonetic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "group": self.group, "match": self.resolved_match(),
            "seed": self.seed, "enabled": self.enabled, "modes": self.modes,
        }


@dataclass(slots=True)
class Hit:
    name: str
    found: str
    start: int
    end: int
    score: float


def load(cfg: dict[str, Any]) -> list[NameEntry]:
    out: list[NameEntry] = []
    for raw in cfg.get("dictionary", {}).get("names", []) or []:
        if isinstance(raw, str):
            out.append(NameEntry(name=raw))
        elif isinstance(raw, dict) and raw.get("name"):
            out.append(NameEntry(
                name=str(raw["name"]),
                group=str(raw.get("group", "")),
                match=str(raw.get("match", "")),
                seed=bool(raw.get("seed", True)),
                enabled=bool(raw.get("enabled", False)),
                modes=[str(m) for m in raw.get("modes", []) or []],
            ))
    return out


class NameIndex:
    """Everything the pipeline needs from the names list, built once."""

    def __init__(self, cfg: dict[str, Any], mode: str = "") -> None:
        settings = cfg.get("dictionary", {}).get("names_settings", {})
        self.tones = bool(settings.get("pinyin_require_tones", False))
        self.seed_enabled = bool(settings.get("seed_prompt", True))
        self.budget = int(settings.get("seed_budget_tokens", 224))
        self.min_chars = int(settings.get("min_chars", 2))
        self.match_new = bool(settings.get("match_new_names", False))
        self.threshold = float(settings.get("threshold", 0.86))

        self.entries = [
            e for e in load(cfg)
            if not e.modes or not mode or mode in e.modes
        ]
        self._keys: list[tuple[NameEntry, str, str]] = []
        for entry in self.entries:
            if not (entry.enabled or self.match_new):
                continue
            kind = entry.resolved_match()
            if kind == "pinyin":
                key = pinyin_key(entry.name, tones=self.tones)
            elif kind == "phonetic":
                key = phonetic_key(entry.name)
            else:
                key = entry.name
            if key:
                self._keys.append((entry, kind, key))

    # -- seeding -----------------------------------------------------------

    def seed_text(self) -> str:
        """Names to hand the decoder, most-used first, inside the budget.

        Budgeted in characters rather than real tokens: whisper's prompt cap
        is 224 tokens, and a CJK name costs roughly one token per character,
        so characters are the conservative estimate.
        """
        if not self.seed_enabled:
            return ""
        wanted = [e for e in self.entries if e.seed]
        wanted.sort(key=lambda e: (-e.hits, e.name))
        picked: list[str] = []
        used = 0
        for entry in wanted:
            cost = len(entry.name) + 2
            if used + cost > self.budget:
                break
            picked.append(entry.name)
            used += cost
        return ", ".join(picked)

    # -- matching ----------------------------------------------------------

    def apply(self, text: str) -> tuple[str, list[Hit]]:
        """Replace spans that sound like a name with the name itself."""
        if not text or not self._keys:
            return text, []

        hits: list[Hit] = []
        for entry, kind, key in self._keys:
            if len(entry.name) < self.min_chars:
                continue
            finder = self._find_cjk if kind == "pinyin" else self._find_latin
            for hit in finder(text, entry, key):
                # Already correct: nothing to do, and no hit to report.
                if hit.found != entry.name:
                    hits.append(hit)

        if not hits:
            return text, []

        # Highest score first, then leftmost; drop anything that overlaps a
        # replacement already made, so two names cannot fight over one span.
        hits.sort(key=lambda h: (-h.score, h.start))
        taken: list[tuple[int, int]] = []
        applied: list[Hit] = []
        for hit in hits:
            if any(hit.start < end and start < hit.end for start, end in taken):
                continue
            taken.append((hit.start, hit.end))
            applied.append(hit)

        for hit in sorted(applied, key=lambda h: -h.start):
            text = text[: hit.start] + hit.name + text[hit.end :]
        return text, applied

    def _find_cjk(self, text: str, entry: NameEntry, key: str) -> Iterable[Hit]:
        width = len(entry.name)
        for start in range(max(0, len(text) - width + 1)):
            span = text[start : start + width]
            if not _CJK.search(span):
                continue
            if pinyin_key(span, tones=self.tones) == key:
                yield Hit(entry.name, span, start, start + width, 1.0)

    def _find_latin(self, text: str, entry: NameEntry, key: str) -> Iterable[Hit]:
        words = list(_WORD.finditer(text))
        if not words:
            return
        # A name can arrive split across words ("wee type" for "wtype"), so
        # try every run of up to four words, longest first.
        max_span = min(4, len(words))
        for size in range(max_span, 0, -1):
            for i in range(len(words) - size + 1):
                start, end = words[i].start(), words[i + size - 1].end()
                span = text[start:end]
                score = _ratio(phonetic_key(span), key)
                if score >= self.threshold:
                    yield Hit(entry.name, span, start, end, score)


def dry_run(cfg: dict[str, Any], texts: Iterable[str], *,
            force_all: bool = True) -> list[dict[str, Any]]:
    """What matching would do to text you already have.

    Sound matching is the one feature here that can corrupt a correct
    transcript, so this runs before anything is switched on.
    """
    index = NameIndex(cfg)
    if force_all:
        # Include entries not yet enabled — the point is to decide about them.
        index.match_new = True
        index = NameIndex({**cfg, "dictionary": {
            **cfg.get("dictionary", {}),
            "names_settings": {
                **cfg.get("dictionary", {}).get("names_settings", {}),
                "match_new_names": True,
            },
        }})

    out: list[dict[str, Any]] = []
    for text in texts:
        new, hits = index.apply(text)
        if hits:
            out.append({
                "before": text,
                "after": new,
                "hits": [
                    {"name": h.name, "found": h.found, "score": round(h.score, 3)}
                    for h in hits
                ],
            })
    return out
