"""The translated strings the daemon itself ships.

Small table, easy to add to and forget one language in — and a gap shows as
English in the middle of a Chinese sentence rather than as an error.
"""
from __future__ import annotations

from omavoi import i18n


def test_every_entry_has_every_language():
    gaps = {en: [lang for lang in i18n.LANGUAGES
                 if lang != "en" and lang not in entry]
            for en, entry in i18n._TABLE.items()}
    gaps = {k: v for k, v in gaps.items() if v}
    assert not gaps, f"{len(gaps)} entries are missing a language: {gaps}"


def test_a_language_outside_the_table_falls_back_to_english():
    text = next(iter(i18n._TABLE))
    assert i18n.t(text, "klingon") == text
    assert i18n.t(text, "") == text
    assert i18n.t("something never translated", "zh") == "something never translated"


def test_languages_is_the_set_the_table_actually_covers():
    covered = {lang for entry in i18n._TABLE.values() for lang in entry}
    assert covered | {"en"} == set(i18n.LANGUAGES)
