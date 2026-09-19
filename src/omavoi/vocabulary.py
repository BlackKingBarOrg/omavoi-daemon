"""One word, its preferred spelling, and optional corrections.

The legacy dictionary remains readable until a safe migration is committed.
Indexes are derived from entries, never persisted as a second word list.
"""
from __future__ import annotations

import copy
import re
import unicodedata
import uuid
from typing import Any

from . import names


class InvalidWord(ValueError):
    def __init__(self, code: str, detail: str = ""):
        self.code = code
        self.detail = detail
        super().__init__(f"{code}: {detail}" if detail else code)


def folded(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold()


def word(text: str, **values: Any) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex, "text": text, "aliases": [], "enabled": True,
        "recognition_hint": True, "normalize_case": True,
        "phonetic": {"enabled": False, "method": "auto"}, "modes": [],
        "updated_at": "", **values,
    }


def modern(cfg: dict) -> bool:
    return cfg.get("dictionary", {}).get("schema_version") == 2


def overlapping(a: dict, b: dict) -> bool:
    return not a["modes"] or not b["modes"] or bool(set(a["modes"]) & set(b["modes"]))


def _clean(value: Any) -> str:
    if not isinstance(value, str):
        raise InvalidWord("invalid_text")
    value = value.strip()
    if not value or len(value) > 200 or any(unicodedata.category(c).startswith("C") for c in value):
        raise InvalidWord("invalid_text", value[:40])
    return value


def validate(entries: list[dict], mode_ids: set[str] | None = None) -> None:
    ids: set[str] = set()
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("id"), str) or not entry["id"]:
            raise InvalidWord("invalid_entry")
        if entry["id"] in ids:
            raise InvalidWord("duplicate_id", entry["id"])
        ids.add(entry["id"])
        entry["text"] = _clean(entry.get("text"))
        for key in ("enabled", "recognition_hint", "normalize_case"):
            if not isinstance(entry.get(key), bool):
                raise InvalidWord("invalid_entry", key)
        if not isinstance(entry.get("aliases"), list) or not isinstance(entry.get("modes"), list):
            raise InvalidWord("invalid_entry")
        if any(not isinstance(m, str) for m in entry["modes"]):
            raise InvalidWord("invalid_mode")
        if mode_ids is not None and set(entry["modes"]) - mode_ids:
            raise InvalidWord("invalid_mode", ", ".join(set(entry["modes"]) - mode_ids))
        p = entry.get("phonetic", {})
        if not isinstance(p, dict) or not isinstance(p.get("enabled"), bool):
            raise InvalidWord("invalid_entry", "phonetic")
        if p.get("method") not in ("auto", "pinyin", "phonetic", "exact"):
            raise InvalidWord("invalid_entry", "method")
        aliases: dict[str, str] = {}
        for alias in entry["aliases"]:
            alias = _clean(alias)
            if alias == entry["text"]:
                raise InvalidWord("already_correct", alias)
            if folded(alias) == folded(entry["text"]) and entry["normalize_case"]:
                continue
            aliases.setdefault(folded(alias), alias)
        entry["aliases"] = list(aliases.values())

    for i, a in enumerate(entries):
        for b in entries[i + 1:]:
            if not overlapping(a, b):
                continue
            if folded(a["text"]) == folded(b["text"]):
                raise InvalidWord("duplicate_word", b["text"])
    # A target must not trigger another entry, even as a substring. This
    # deliberately conservative check makes another application stable.
    for a in entries:
        triggers = list(a["aliases"])
        if a["normalize_case"]:
            triggers.append(a["text"])
        for b in entries:
            if a is b or not overlapping(a, b):
                continue
            if set(map(folded, a["aliases"])) & set(map(folded, b["aliases"])):
                raise InvalidWord("alias_conflict", b["text"])
            for trigger in triggers:
                changes_target = any(b["text"][start:end] != a["text"]
                                     for start, end in spans(b["text"], trigger,
                                                            legacy=a.get("legacy", {}).get("boundaries", False)))
                contains_word = trigger in a["aliases"] and any(spans(trigger, b["text"]))
                if changes_target or contains_word:
                    raise InvalidWord("target_conflict", f"{trigger} / {b['text']}")


def migration(cfg: dict) -> tuple[dict, list[str]]:
    """Plan only. Refuse ambiguous old behavior instead of rewriting it."""
    if modern(cfg):
        data = copy.deepcopy(cfg["dictionary"])
        validate(data["entries"])
        return data, []
    old = cfg.get("dictionary", {})
    entries: dict[str, dict] = {}
    issues: list[str] = []
    for source, target in old.get("rules", {}).items():
        if not target or not source:
            issues.append(f"{source!r} → {target!r}: empty replacement")
            continue
        e = entries.setdefault(target, word(
            target, id=uuid.uuid5(uuid.NAMESPACE_URL, "omavoi:" + target).hex,
            recognition_hint=False, normalize_case=False,
            legacy={"rule_keys": [], "name": False, "boundaries": True},
        ))
        e["legacy"]["rule_keys"].append(source)
        if source.isascii() and target.isascii() and source.lower() == target.lower():
            e["normalize_case"] = True
        elif source != target:
            e["aliases"].append(source)
        else:
            e["normalize_case"] = True
    for old_name in names.load(cfg):
        if old_name.name in entries and old_name.modes:
            issues.append(f"{old_name.name}: corrections and name have different mode scopes")
            continue
        e = entries.setdefault(old_name.name, word(
            old_name.name, id=uuid.uuid5(uuid.NAMESPACE_URL, "omavoi:" + old_name.name).hex,
            normalize_case=False, legacy={"rule_keys": [], "name": True, "boundaries": False},
        ))
        if e["legacy"].get("name") and e.get("legacy_name_loaded"):
            issues.append(f"{old_name.name}: duplicate name")
        e["legacy_name_loaded"] = True
        e["legacy"]["name"] = True
        e["legacy"]["group"] = old_name.group
        e["recognition_hint"] = old_name.seed
        e["phonetic"] = {
            "enabled": old_name.enabled or old.get("names_settings", {}).get("match_new_names", False),
            "method": old_name.resolved_match(),
        }
        e["modes"] = old_name.modes
    result = {"schema_version": 2, "revision": 0, "entries": list(entries.values()),
              "names_settings": copy.deepcopy(old.get("names_settings", {}))}
    for e in result["entries"]:
        e.pop("legacy_name_loaded", None)
    try:
        validate(result["entries"], set(cfg.get("modes", {})))
    except (InvalidWord, TypeError, KeyError) as exc:
        issues.append(str(exc))
    # Old re.IGNORECASE differs from Unicode casefold for non-ASCII text;
    # preserve its matcher for migrated explicit rules.
    return result, issues


def mode_flags(cfg: dict, rules: dict) -> tuple[bool, bool]:
    if "vocabulary" in rules:
        return bool(rules["vocabulary"]), bool(rules["vocabulary"])
    return (bool(rules.get("dictionary", True) and cfg.get("post", {}).get("enabled", True)),
            bool(rules.get("names", True)))


def _inside(char: str) -> bool:
    return bool(char) and (char == "_" or (char.isalnum() and not names._CJK.match(char))
                           or unicodedata.category(char).startswith("M"))


def spans(text: str, needle: str, *, legacy: bool = False):
    if legacy:
        pattern = re.escape(needle)
        if re.match(r"^[\w\s]+$", needle, re.ASCII):
            pattern = rf"\b{pattern}\b"
        for m in re.finditer(pattern, text, re.IGNORECASE):
            yield m.start(), m.end()
        return
    # Normalize one grapheme-like cluster at a time and keep source offsets.
    # Casefold may expand ß and NFC may compose e + combining acute.
    hay, starts, ends = "", [], []
    start = 0
    while start < len(text):
        end = start + 1
        while end < len(text) and unicodedata.category(text[end]).startswith("M"):
            end += 1
        key = folded(text[start:end])
        hay += key
        starts.extend([start] * len(key))
        ends.extend([end] * len(key))
        start = end
    key = folded(needle)
    pos = hay.find(key)
    while pos >= 0:
        last = pos + len(key) - 1
        start, end = starts[pos], ends[last]
        whole_cluster = (pos == 0 or starts[pos - 1] != start) and (
            last + 1 == len(hay) or ends[last + 1] != end)
        left = text[start - 1] if start else ""
        right = text[end] if end < len(text) else ""
        # CJK phrases have no space boundary; symbol-bearing Latin terms do.
        cjk = bool(names._CJK.search(needle)) and not re.search(r"[A-Za-z0-9]", needle)
        if whole_cluster and (cjk or (not _inside(left) and not _inside(right))):
            yield start, end
        pos = hay.find(key, pos + 1)


class Index:
    def __init__(self, cfg: dict, mode: str = "", rules: dict | None = None):
        self.cfg = cfg
        self.rules = rules or {}
        self.correct, self.hint = mode_flags(cfg, self.rules)
        self.entries = [e for e in cfg["dictionary"]["entries"] if e["enabled"]
                        and (not e["modes"] or not mode or mode in e["modes"])]
        name_entries = []
        for e in self.entries:
            name_entries.append({
                "name": e["text"], "seed": e["recognition_hint"] and self.hint,
                "enabled": e["phonetic"]["enabled"] and self.hint,
                "match": "" if e["phonetic"]["method"] == "auto" else e["phonetic"]["method"],
            })
        settings = {**cfg["dictionary"].get("names_settings", {}), "match_new_names": False}
        self.names = names.NameIndex({"dictionary": {"names": name_entries, "names_settings": settings}})

    def seed_text(self) -> str:
        return self.names.seed_text()

    def apply(self, text: str) -> tuple[str, list[dict]]:
        candidates: list[dict] = []
        protected: list[tuple[int, int]] = []
        for e in self.entries:
            active = self.correct or (self.hint and e["phonetic"]["enabled"])
            if not active:
                continue
            for start, end in spans(text, e["text"]):
                if text[start:end] == e["text"]:
                    protected.append((start, end))
            if not self.correct:
                continue
            triggers = [(s, "alias", 0) for s in e["aliases"]]
            if e["normalize_case"]:
                triggers.append((e["text"], "case", 1))
            for source, reason, priority in triggers:
                for start, end in spans(text, source, legacy=e.get("legacy", {}).get("boundaries", False)):
                    if text[start:end] != e["text"]:
                        candidates.append({"id": e["id"], "before": text[start:end], "after": e["text"],
                                           "start": start, "end": end, "reason": reason, "priority": priority})
        # Generate fuzzy candidates independently, so tied names stay unchanged.
        fuzzy: list[dict] = []
        for entry, kind, key in self.names._keys:
            if len(entry.name) < self.names.min_chars or names._too_vague(kind, key):
                continue
            finder = self.names._find_cjk if kind == "pinyin" else self.names._find_latin
            for hit in finder(text, entry, key):
                if hit.found != entry.name:
                    e = next(e for e in self.entries if e["text"] == entry.name)
                    fuzzy.append({"id": e["id"], "before": hit.found, "after": hit.name,
                                  "start": hit.start, "end": hit.end, "score": hit.score,
                                  "reason": "phonetic", "priority": 2})
        for hit in fuzzy:
            tied = any(h["after"] != hit["after"] and h["score"] == hit["score"]
                       and h["start"] < hit["end"] and hit["start"] < h["end"] for h in fuzzy)
            if not tied:
                candidates.append(hit)
        candidates.sort(key=lambda h: (h["priority"], -h.get("score", 1),
                                       -(h["end"] - h["start"]), h["start"]))
        occupied = list(protected)
        applied = []
        for h in candidates:
            if any(h["start"] < end and start < h["end"] for start, end in occupied):
                continue
            occupied.append((h["start"], h["end"]))
            applied.append(h)
        for h in sorted(applied, key=lambda h: -h["start"]):
            text = text[:h["start"]] + h["after"] + text[h["end"]:]
        return text, applied
