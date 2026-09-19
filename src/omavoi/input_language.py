"""Input-language choices and optional Chinese script normalization.

The Whisper language tokens come from openai/whisper's tokenizer.py.
speech_languages.json contains their display names from Babel's CLDR data;
names are bundled so displaying a menu needs no network or model load.
Chinese script choices are application preferences, never Whisper tokens.
"""

from __future__ import annotations

import json
import re
from functools import cache
from importlib.resources import files

SCRIPTS = {"zh-Hans": "t2s", "zh-Hant": "s2t"}
_AUTO = {"en": "Auto", "zh": "自动", "th": "อัตโนมัติ", "de": "Automatisch",
         "fr": "Automatique", "es": "Automático", "ja": "自動", "vi": "Tự động"}
_SCRIPT_NAMES = {
    "zh-Hans": {"en": "Simplified Chinese", "zh": "简体中文", "th": "ภาษาจีนตัวย่อ",
                "de": "Chinesisch (vereinfacht)", "fr": "Chinois simplifié", "es": "Chino simplificado",
                "ja": "簡体字中国語", "vi": "Tiếng Trung giản thể"},
    "zh-Hant": {"en": "Traditional Chinese", "zh": "繁体中文", "th": "ภาษาจีนตัวเต็ม",
                "de": "Chinesisch (traditionell)", "fr": "Chinois traditionnel", "es": "Chino tradicional",
                "ja": "繁体字中国語", "vi": "Tiếng Trung phồn thể"},
}
_LEGACY_CHINESE = {
    "en": "Chinese (no conversion)", "zh": "中文（不转换）", "th": "ภาษาจีน (ไม่แปลงตัวอักษร)",
    "de": "Chinesisch (ohne Umwandlung)", "fr": "Chinois (sans conversion)",
    "es": "Chino (sin conversión)", "ja": "中国語（字体変換なし）", "vi": "Tiếng Trung (không đổi dạng chữ)",
}


@cache
def catalogue() -> tuple[dict, ...]:
    return tuple(json.loads(files("omavoi").joinpath("speech_languages.json").read_text(encoding="utf-8")))


def valid(value: str) -> bool:
    return value in ("", "auto", *SCRIPTS) or any(row["code"] == value for row in catalogue())


def recognition_language(value: str) -> str:
    """An explicit auto overrides a backend's global language default."""
    return "zh" if value in SCRIPTS else value or "auto"


def options(model: str, locale: str = "en", current: str = "") -> list[dict[str, str]]:
    locale = locale if locale in _AUTO else "en"
    rows = [{"value": "auto", "label": _AUTO[locale], "description": ""}]
    # Cantonese has a separate token starting with large-v3. Every model in
    # our local catalogue is multilingual; older weights have 99 tokens.
    older = model.removeprefix("ggml:") in {"base", "small", "medium", "large-v1", "large-v2", "whisper-1"}
    english_only = model.removesuffix(".bin").endswith(".en")
    for row in catalogue():
        code = row["code"]
        if (older and code == "yue") or (english_only and code != "en"):
            continue
        choices = list(SCRIPTS) if code == "zh" else [code]
        # Existing `zh` remains a real choice with its old no-conversion
        # semantics. Opening the picker must never silently opt it into Hans.
        if code == "zh" and current == "zh":
            choices.append("zh")
        for choice in choices:
            names = _SCRIPT_NAMES.get(choice, row["names"])
            label = _LEGACY_CHINESE[locale] if choice == "zh" else names[locale]
            native = names["zh"] if choice in SCRIPTS else row["native"]
            search = list(dict.fromkeys([native, names["en"], names["zh"], choice]))
            rows.append({"value": choice, "label": label,
                         "description": " · ".join(s for s in search if s != label)})
    preferred = {code: i for i, code in enumerate(("auto", "zh-Hans", "zh-Hant", "en", "th"))}
    rows.sort(key=lambda row: (preferred.get(row["value"], 5), row["label"].casefold()))
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


def convert(text: str, language: str, protected: tuple[str, ...] = ()) -> str:
    if language not in SCRIPTS or not text:
        return text
    spans = [(m.start(), m.end()) for m in _VERBATIM.finditer(text)]
    # Kana provides evidence that a clause is Japanese. Han-only proper names
    # cannot be classified reliably; the user's dictionary protects those.
    spans.extend((m.start(), m.end()) for m in _CLAUSE.finditer(text) if _KANA.search(m.group()))
    for spelling in protected:
        if spelling:
            spans.extend((m.start(), m.end()) for m in re.finditer(re.escape(spelling), text))
    converter = _converter(language)
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
