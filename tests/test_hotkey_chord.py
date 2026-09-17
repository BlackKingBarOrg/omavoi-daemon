"""A hotkey can be a combination, not only a single key.

It was one evdev code all the way down: `self.code: int`, one `_held` bool,
`event.code == self.code`. Which was fine until the key that does nothing on
its own turned out to be layout-dependent — Right Alt is AltGr on most
non-US keyboards — and the answer to that is usually a chord, because a
modifier plus an ordinary key collides with nothing.

A single key is a chord of one, so there is one code path rather than two.
"""
from __future__ import annotations

import pytest

from omavoi.hotkey import (
    HotkeyListener,
    HotkeyUnavailable,
    canonical_name,
    key_code,
    parse_chord,
)


def _codes(*names):
    from evdev import ecodes

    return [getattr(ecodes, f"KEY_{n}") for n in names]


# -- parsing ---------------------------------------------------------------


@pytest.mark.parametrize("spec,name,parts", [
    ("RIGHTALT", "RIGHTALT", 1),
    ("rightalt", "RIGHTALT", 1),
    ("CTRL+SPACE", "CTRL+SPACE", 2),
    ("SUPER+V", "SUPER+V", 2),
    ("ALT+F9", "ALT+F9", 2),
    ("CTRL+SHIFT+SPACE", "CTRL+SHIFT+SPACE", 3),
])
def test_a_chord_parses(spec, name, parts):
    chord = parse_chord(spec)
    assert chord.name == name
    assert len(chord.parts) == parts


@pytest.mark.parametrize("spec,want", [
    ("shift+ctrl+space", "CTRL+SHIFT+SPACE"),
    ("space+ctrl", "CTRL+SPACE"),
    ("v+super+shift", "SHIFT+SUPER+V"),
])
def test_one_chord_has_one_spelling(spec, want):
    """However it was typed or captured. Otherwise `hotkey check` reports a
    daemon bound to SHIFT+CTRL+SPACE as disagreeing with a config that says
    CTRL+SHIFT+SPACE."""
    assert parse_chord(spec).name == want
    assert canonical_name(spec.upper().split("+")) == want


def test_parts_are_in_the_same_order_as_the_name():
    chord = parse_chord("space+ctrl")
    assert chord.name == "CTRL+SPACE"
    assert set(chord.parts[0]) == set(_codes("LEFTCTRL", "RIGHTCTRL"))
    assert chord.parts[1] == tuple(_codes("SPACE"))


def test_a_bare_modifier_means_either_side():
    """evdev has no "Ctrl", only KEY_LEFTCTRL and KEY_RIGHTCTRL, and someone
    writing CTRL+SPACE wants whichever is under their hand."""
    chord = parse_chord("CTRL+SPACE")
    assert set(chord.parts[0]) == set(_codes("LEFTCTRL", "RIGHTCTRL"))
    assert chord.satisfied_by(set(_codes("LEFTCTRL", "SPACE")))
    assert chord.satisfied_by(set(_codes("RIGHTCTRL", "SPACE")))


def test_a_sided_modifier_stays_on_that_side():
    chord = parse_chord("RIGHTCTRL+SPACE")
    assert not chord.satisfied_by(set(_codes("LEFTCTRL", "SPACE")))
    assert chord.satisfied_by(set(_codes("RIGHTCTRL", "SPACE")))


@pytest.mark.parametrize("spec", ["", "+", "NOTAKEY", "CTRL+NOTAKEY", "CTRL+CTRL"])
def test_a_bad_chord_is_refused(spec):
    with pytest.raises(HotkeyUnavailable):
        parse_chord(spec)


def test_key_code_still_answers_for_a_single_key():
    assert key_code("RIGHTALT") == _codes("RIGHTALT")[0]
    with pytest.raises(HotkeyUnavailable):
        key_code("CTRL+SPACE")


# -- what the chord watches ------------------------------------------------


def test_every_code_of_every_part_is_watched():
    chord = parse_chord("CTRL+SPACE")
    assert chord.codes == frozenset(_codes("LEFTCTRL", "RIGHTCTRL", "SPACE"))


@pytest.mark.parametrize("held,satisfied", [
    (("SPACE",), False),
    (("LEFTCTRL",), False),
    (("LEFTCTRL", "SPACE"), True),
    (("LEFTCTRL", "SPACE", "A"), True),      # other keys down do not matter
    (("LEFTSHIFT", "SPACE"), False),
])
def test_a_chord_is_held_only_when_every_part_is(held, satisfied):
    assert parse_chord("CTRL+SPACE").satisfied_by(set(_codes(*held))) is satisfied


# -- the listener's edges --------------------------------------------------


def _listener(key, mode="push_to_talk"):
    events = []
    listener = HotkeyListener(
        {"hotkey": {"key": key, "mode": mode, "devices": []}},
        on_press=lambda: events.append("press"),
        on_release=lambda: events.append("release"),
        on_toggle=lambda: events.append("toggle"),
    )
    return listener, events


def _press(listener, *names):
    for code in _codes(*names):
        listener._handle(code, 1)


def _release(listener, *names):
    for code in _codes(*names):
        listener._handle(code, 0)


def test_a_single_key_behaves_exactly_as_before(home):
    listener, events = _listener("RIGHTALT")
    _press(listener, "RIGHTALT")
    _release(listener, "RIGHTALT")
    assert events == ["press", "release"]


def test_a_chord_fires_when_the_last_key_goes_down(home):
    listener, events = _listener("CTRL+SPACE")
    _press(listener, "LEFTCTRL")
    assert events == [], "the modifier alone is not the chord"
    _press(listener, "SPACE")
    assert events == ["press"]


def test_releasing_any_part_ends_the_take(home):
    """Releasing Space while still holding Ctrl has to end it: the chord is
    no longer held, and waiting for the modifier would leave a take running
    after the user let go of the key they think of as the button."""
    listener, events = _listener("CTRL+SPACE")
    _press(listener, "LEFTCTRL", "SPACE")
    _release(listener, "SPACE")
    assert events == ["press", "release"]
    _release(listener, "LEFTCTRL")
    assert events == ["press", "release"], "the second release is not another one"


def test_autorepeat_is_not_a_new_press(home):
    listener, events = _listener("CTRL+SPACE")
    _press(listener, "LEFTCTRL", "SPACE")
    for code in _codes("SPACE"):
        listener._handle(code, 2)      # _KEY_HOLD
    assert events == ["press"]


def test_other_keys_pressed_during_a_take_change_nothing(home):
    listener, events = _listener("CTRL+SPACE")
    _press(listener, "LEFTCTRL", "SPACE")
    listener._handle(_codes("A")[0], 1)
    listener._handle(_codes("A")[0], 0)
    assert events == ["press"]


def test_either_side_of_a_bare_modifier_works(home):
    listener, events = _listener("CTRL+SPACE")
    _press(listener, "RIGHTCTRL", "SPACE")
    _release(listener, "SPACE")
    assert events == ["press", "release"]


def test_toggle_fires_on_the_way_in_only(home):
    """A combination has as many release edges as it has keys, and a toggle
    that fired on each would stop the recording it just started."""
    listener, events = _listener("CTRL+SPACE", mode="toggle")
    _press(listener, "LEFTCTRL", "SPACE")
    _release(listener, "SPACE", "LEFTCTRL")
    assert events == ["toggle"]
    _press(listener, "LEFTCTRL", "SPACE")
    assert events == ["toggle", "toggle"]


def test_a_keyboard_going_away_mid_take_ends_it(home):
    """A key cannot be released by a device that is gone."""
    listener, events = _listener("CTRL+SPACE")
    _press(listener, "LEFTCTRL", "SPACE")
    assert listener._held
    # What the read loop does when the device disappears.
    listener._down.clear()
    listener._held = False
    listener._on_release()
    assert events == ["press", "release"]


# -- what ends up in the config file ---------------------------------------


@pytest.mark.parametrize("typed,stored", [
    ("super+v", "SUPER+V"),
    ("shift+ctrl+space", "CTRL+SHIFT+SPACE"),
    ("space+ctrl", "CTRL+SPACE"),
    ("rightalt", "RIGHTALT"),
])
def test_the_config_stores_the_canonical_spelling(home, typed, stored):
    """Both spellings work — the listener canonicalises internally — but
    then `config get` shows what you typed while `hotkey check` and the
    daemon report the canonical one, and the comparison that matters (has
    the daemon picked up the change) is between two spellings of one thing.
    """
    from omavoi import config

    config.write(config.defaults())
    config.set_path("hotkey.key", typed)
    assert config.load()["hotkey"]["key"] == stored


def test_a_key_that_does_not_resolve_is_only_uppercased(home):
    """Canonicalising the spelling needs no evdev — it is the modifier order
    and the case — and resolving the names here would put a 25 ms import on
    every config.load. validate() is what reports that it is not a key."""
    from omavoi import config

    cfg = config.defaults()
    cfg["hotkey"]["key"] = "notakey"
    config.normalise_hotkey(cfg)
    assert cfg["hotkey"]["key"] == "NOTAKEY"
    assert any("not an evdev key" in problem for problem in config.validate(cfg))


def test_nothing_to_do_is_reported_as_nothing(home):
    from omavoi import config

    cfg = config.defaults()
    assert config.normalise_hotkey(cfg) == []
    assert config.normalise_hotkey(cfg) == [], "and it is not a ratchet"


def test_a_combination_passes_the_set_time_check(home):
    from omavoi.commands.keys import _why_not_that_hotkey

    assert _why_not_that_hotkey("hotkey.key", "CTRL+SPACE") == ""
    assert _why_not_that_hotkey("hotkey.key", "SUPER+V") == ""
    assert _why_not_that_hotkey("hotkey.key", "CTRL+NOTAKEY")
    assert _why_not_that_hotkey("hotkey.key", "CTRL+CTRL")
