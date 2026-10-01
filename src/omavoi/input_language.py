"""Input-language choices and the optional Chinese script preference.

The Whisper language tokens come from openai/whisper's tokenizer.py.
speech_languages.json contains their display names from Babel's CLDR data;
names are bundled so displaying a menu needs no network or model load.

Which language the recognizer listens for and which Chinese characters come
out are two settings, not one. They used to share the picker: Simplified and
Traditional Chinese were entries in the language list, so choosing one also
pinned recognition to Chinese. Someone who speaks Chinese, English and Thai
wants auto-detection *and* Simplified characters, and that combination had
no way to be said — people wrote "only output Simplified Chinese" into the
decoder prompt instead, which is a hint the model is free to ignore.

The script preference is an application setting, never a Whisper token.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib.resources import files

SCRIPTS = {"zh-Hans": "t2s", "zh-Hant": "s2t"}
# Written with Han characters that are not Chinese, so a Simplified or
# Traditional preference must leave them alone. Codes for a mode's own
# language, the names whisper reports for a detected one.
_HAN_ELSEWHERE = frozenset({"ja", "japanese", "ko", "korean"})
_AUTO = {"en": "Auto", "zh": "自动", "th": "อัตโนมัติ", "de": "Automatisch",
         "fr": "Automatique", "es": "Automático", "ja": "自動", "vi": "Tự động"}


@cache
def catalogue() -> tuple[dict, ...]:
    return tuple(json.loads(files("omavoi").joinpath("speech_languages.json").read_text(encoding="utf-8")))


def valid(value: str) -> bool:
    """A recognition language. zh-Hans and zh-Hant are still accepted here,
    because they were languages until the script became its own setting;
    split() turns them into the pair they meant."""
    return value in ("", "auto", *SCRIPTS) or any(row["code"] == value for row in catalogue())


def valid_script(value: str) -> bool:
    """Empty leaves the characters as recognized."""
    return value in ("", *SCRIPTS)


def split(value: str) -> tuple[str, str]:
    """A language value as (recognition language, script).

    zh-Hans used to mean both "listen for Chinese" and "write Simplified",
    so an old value comes apart into the two settings it stood for.
    """
    return ("zh", value) if value in SCRIPTS else (value, "")


def recognition_language(value: str) -> str:
    """An explicit auto overrides a backend's global language default."""
    return "zh" if value in SCRIPTS else value or "auto"


def applies(script: str, language: str, detected: str = "") -> bool:
    """Whether a script preference should touch this take.

    Any take that can carry Chinese characters: one recognized as Chinese,
    and one on auto — a Chinese sentence with English in it is often
    detected as English. Not Japanese or Korean, whose Han characters are
    not Chinese ones to convert, whether the mode names that language or
    auto detected it. `detected` is what the backend reported; an unknown
    one does not block the preference.
    """
    if script not in SCRIPTS:
        return False
    spoken = language if language not in ("", "auto") else str(detected or "")
    return spoken.strip().lower() not in _HAN_ELSEWHERE


def options(model: str, locale: str = "en", current: str = "") -> list[dict[str, str]]:
    from . import models

    locale = locale if locale in _AUTO else "en"
    rows = [{"value": "auto", "label": _AUTO[locale], "description": ""}]
    # Cantonese has a separate token starting with large-v3. Every model in
    # our local catalogue is multilingual; older weights have 99 tokens.
    older = model.removeprefix("ggml:") in {"base", "small", "medium", "large-v1", "large-v2", "whisper-1"}
    english_only = model.removesuffix(".bin").endswith(".en")
    # A model that is not whisper takes the languages its catalogue entry
    # lists: Qwen3-ASR has 30 of these hundred, and offering the rest would
    # name a language it was never trained to hear.
    spec = models.spec(model) if model else None
    only = set(spec.language_codes) if spec is not None and spec.language_codes else None
    for row in catalogue():
        code = row["code"]
        if (older and code == "yue") or (english_only and code != "en"):
            continue
        if only is not None and code not in only:
            continue
        names = row["names"]
        label = names[locale]
        search = list(dict.fromkeys([row["native"], names["en"], names["zh"], code]))
        rows.append({"value": code, "label": label,
                     "description": " · ".join(s for s in search if s != label)})
    preferred = {code: i for i, code in enumerate(("auto", "zh", "en", "th"))}
    rows.sort(key=lambda row: (preferred.get(row["value"], 4), row["label"].casefold()))
    if current and current != "auto" and not any(row["value"] == current for row in rows):
        rows.append({"value": current, "label": current, "description": ""})
    return rows


@cache
def _converter(script: str):
    # Load dictionaries once, only if a script preference is actually used.
    from opencc import OpenCC

    return OpenCC(SCRIPTS[script])


# Protect verbatim structures. Include CJK path characters but stop before
# sentence punctuation so ordinary prose after a URL still gets converted.
_VERBATIM = re.compile(
    r"```[\s\S]*?(?:```|$)|`[^`\n]*`"
    r"|(?:https?://|www\.)[^\s<>\"'，。！？；、]+"
    r"|(?<!\w)(?:[/~]|\.{1,2}/|[\w.-]+/|[A-Za-z]:\\)[^\s<>\"'，。！？；、]+"
    r"|[^\s<>:：，。！？；、]+@[^\s<>:：，。！？；、]+"
)
_KANA = re.compile(r"[\u3040-\u30ff]")
_CLAUSE = re.compile(r"[^\n。！？!?；;，,:：]+")


def convert(text: str, script: str, protected: tuple[str, ...] = ()) -> str:
    if script not in SCRIPTS or not text:
        return text
    spans = [(m.start(), m.end()) for m in _VERBATIM.finditer(text)]
    # Kana provides evidence that a clause is Japanese. Han-only proper names
    # cannot be classified reliably; the user's dictionary protects those.
    spans.extend((m.start(), m.end()) for m in _CLAUSE.finditer(text) if _KANA.search(m.group()))
    for spelling in protected:
        if spelling:
            spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(spelling), text))
    converter = _converter(script)
    out: list[str] = []
    end = 0
    for start, stop in sorted(spans):
        if stop <= end:
            continue
        if start > end:
            out.append(converter.convert(text[end:start]))
        out.append(text[max(start, end):stop])
        end = stop
    out.append(converter.convert(text[end:]))
    return "".join(out)
