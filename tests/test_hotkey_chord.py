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


# -- naming a key by the character on it -----------------------------------


def test_a_key_nothing_emits_is_refused(home, monkeypatch):
    """KEY_QUESTION is a real evdev code and no keyboard has it, because ?
    is Shift and some other key. The name resolved, so `config set
    hotkey.key CTRL+QUESTION` was accepted and the daemon bound nothing.
    """
    from omavoi import hotkey
    from omavoi.commands import keys as keys_mod

    monkeypatch.setattr(hotkey, "absent_parts",
                        lambda chord, explicit=None: ["QUESTION"]
                        if "QUESTION" in chord.name else [])
    why = keys_mod._why_not_that_hotkey("hotkey.key", "CTRL+QUESTION")
    assert why and "QUESTION" in why
    assert "hotkey capture" in why, "it must say how to find the right key"
    assert keys_mod._why_not_that_hotkey("hotkey.key", "CTRL+SLASH") == ""


def test_nothing_is_refused_when_nothing_can_be_proven(home, monkeypatch):
    """A shell without the `input` group can open no device, so it knows
    nothing about which keys exist — and refusing there would be the sixth
    time this program answered for the checking process instead of the one
    that runs."""
    from omavoi import hotkey
    from omavoi.commands import keys as keys_mod

    monkeypatch.setattr(hotkey, "absent_parts", lambda chord, explicit=None: [])
    assert keys_mod._why_not_that_hotkey("hotkey.key", "CTRL+QUESTION") == ""


def test_absent_parts_proves_nothing_when_it_can_open_nothing(home, tmp_path):
    """Its whole contract: [] means "cannot tell", never "the key is fine"."""
    from omavoi.hotkey import absent_parts, parse_chord

    missing = tmp_path / "no-such-device"
    assert absent_parts(parse_chord("CTRL+QUESTION"), [str(missing)]) == []


def test_naming_a_key_by_its_character_says_so(home):
    """`CTRL+?` said only "unknown key name '?'", which is no help to
    someone reading the character printed on the key."""
    from omavoi.commands.keys import _why_not_that_hotkey

    why = _why_not_that_hotkey("hotkey.key", "CTRL+?")
    assert "physical key" in why
    assert "every layout" in why, "the answer is layout-dependent, not a table"
    # An ordinary typo gets the ordinary message, without the lecture.
    plain = _why_not_that_hotkey("hotkey.key", "CTRL+NOTAKEY")
    assert "physical key" not in plain


@pytest.mark.parametrize("name,looks", [
    ("?", True), ("+", True), (":", True), ("~", True),
    ("A", False), ("F9", False), ("SLASH", False), ("RIGHTALT", False),
])
def test_what_counts_as_naming_a_character(name, looks):
    from omavoi.commands.keys import _looks_like_a_character

    assert _looks_like_a_character(name) is looks


# -- capture, which is the only way to set this from the console -----------


class _FakeEvent:
    def __init__(self, code, value):
        from evdev import ecodes

        self.type = ecodes.EV_KEY
        self.code = code
        self.value = value


class _FakeDevice:
    """One device that hands out a scripted burst of events."""

    fd = 99

    def __init__(self, bursts):
        from evdev import ecodes

        self._bursts = list(bursts)
        self._caps = {ecodes.EV_KEY: list(range(256))}

    def capabilities(self):
        return self._caps

    def read(self):
        return self._bursts.pop(0) if self._bursts else []

    def close(self):
        pass


def _capture_with(monkeypatch, script, timeout=5.0):
    """Run capture() against a scripted key sequence."""
    import select

    from omavoi import hotkey

    device = _FakeDevice(script)
    monkeypatch.setattr(hotkey, "time", hotkey.time)
    monkeypatch.setattr("evdev.InputDevice", lambda path: device)
    monkeypatch.setattr("evdev.list_devices", lambda: ["/dev/input/event0"])
    monkeypatch.setattr(select, "select",
                        lambda r, w, x, t=None: ([device.fd], [], []))
    return hotkey.capture(timeout)


def test_capture_returns_a_single_key_as_one_name(home, monkeypatch):
    from evdev import ecodes

    got = _capture_with(monkeypatch, [
        [_FakeEvent(ecodes.KEY_RIGHTALT, 1)],
        [_FakeEvent(ecodes.KEY_RIGHTALT, 0)],
    ])
    assert got == "RIGHTALT"


def test_capture_returns_a_held_combination(home, monkeypatch):
    """This is the console's only way to set the hotkey: hold it while the
    button is waiting."""
    from evdev import ecodes

    got = _capture_with(monkeypatch, [
        [_FakeEvent(ecodes.KEY_LEFTCTRL, 1)],
        [_FakeEvent(ecodes.KEY_SLASH, 1)],
        [_FakeEvent(ecodes.KEY_SLASH, 0)],
    ])
    assert got == "CTRL+SLASH"


def test_capture_orders_what_it_caught(home, monkeypatch):
    """Pressed slash-then-ctrl, stored CTRL+SLASH, so the config and the
    daemon agree however the fingers landed."""
    from evdev import ecodes

    got = _capture_with(monkeypatch, [
        [_FakeEvent(ecodes.KEY_SLASH, 1)],
        [_FakeEvent(ecodes.KEY_LEFTCTRL, 1)],
        [_FakeEvent(ecodes.KEY_LEFTCTRL, 0)],
    ])
    assert got == "CTRL+SLASH"


def test_capture_ends_on_the_first_release(home, monkeypatch):
    """Waiting for every key to come up would let a slow finger add one that
    was never meant to be in the chord."""
    from evdev import ecodes

    got = _capture_with(monkeypatch, [
        [_FakeEvent(ecodes.KEY_LEFTCTRL, 1)],
        [_FakeEvent(ecodes.KEY_SLASH, 1)],
        [_FakeEvent(ecodes.KEY_SLASH, 0)],
        [_FakeEvent(ecodes.KEY_LEFTSHIFT, 1)],   # too late to join
    ])
    assert got == "CTRL+SLASH"


def test_capture_ignores_a_mouse_button(home, monkeypatch):
    """A mouse reports its buttons as EV_KEY too, and the first capture this
    program ever did picked up a stray left-click."""
    from evdev import ecodes

    got = _capture_with(monkeypatch, [
        [_FakeEvent(ecodes.BTN_LEFT, 1)],
        [_FakeEvent(ecodes.BTN_LEFT, 0)],
        [_FakeEvent(ecodes.KEY_F9, 1)],
        [_FakeEvent(ecodes.KEY_F9, 0)],
    ])
    assert got == "F9"


def test_a_lone_modifier_captured_stays_on_its_side(home, monkeypatch):
    """The shipped hotkey is RIGHTCTRL, and someone who binds a bare
    modifier picked that one because the other is in constant use for
    shortcuts. Widening it would start a recording on every Ctrl-C."""
    from evdev import ecodes

    got = _capture_with(monkeypatch, [
        [_FakeEvent(ecodes.KEY_RIGHTCTRL, 1)],
        [_FakeEvent(ecodes.KEY_RIGHTCTRL, 0)],
    ])
    assert got == "RIGHTCTRL"


def test_a_modifier_in_a_chord_is_widened_to_either_side(home, monkeypatch):
    """evdev reports the physical key, so this captured LEFTCTRL+SLASH — a
    binding that works with one hand and not the other."""
    from evdev import ecodes

    got = _capture_with(monkeypatch, [
        [_FakeEvent(ecodes.KEY_RIGHTCTRL, 1)],
        [_FakeEvent(ecodes.KEY_SLASH, 1)],
        [_FakeEvent(ecodes.KEY_SLASH, 0)],
    ])
    assert got == "CTRL+SLASH", "right Ctrl and left Ctrl capture the same"
