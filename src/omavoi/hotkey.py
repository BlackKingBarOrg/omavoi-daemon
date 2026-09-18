"""Global push-to-talk key, read straight from evdev.

Why not a Hyprland binding: binding press *and* release on a modifier key
fights itself — the modmask changes the instant the key goes down, which
immediately fires the release binding and yields a 0.0s recording. evdev
sits below xkb, so RIGHTALT is KEY_RIGHTALT no matter how the layout
remaps it (altwin:swap_alt_win included).

The devices are read passively, never grabbed, so the key still reaches
the focused application.
"""

from __future__ import annotations

import logging
import selectors
import threading
import time
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)

_KEY_UP, _KEY_DOWN, _KEY_HOLD = 0, 1, 2


class HotkeyUnavailable(RuntimeError):
    pass


# A bare modifier name means either side. Someone writing CTRL+SPACE wants
# the chord to fire from whichever Ctrl is under their hand, and evdev has no
# such thing as "Ctrl" — only KEY_LEFTCTRL and KEY_RIGHTCTRL. Writing
# RIGHTCTRL+SPACE still pins it to one.
_EITHER_SIDE = {
    "CTRL": ("LEFTCTRL", "RIGHTCTRL"),
    "CONTROL": ("LEFTCTRL", "RIGHTCTRL"),
    "SHIFT": ("LEFTSHIFT", "RIGHTSHIFT"),
    "ALT": ("LEFTALT", "RIGHTALT"),
    "SUPER": ("LEFTMETA", "RIGHTMETA"),
    "META": ("LEFTMETA", "RIGHTMETA"),
    "WIN": ("LEFTMETA", "RIGHTMETA"),
    "CMD": ("LEFTMETA", "RIGHTMETA"),
}

# Modifiers first and in a fixed order, so one chord has one spelling however
# it was typed or captured: CTRL+SHIFT+SPACE, never SHIFT+CTRL+SPACE.
_MODIFIER_ORDER = ("CTRL", "SHIFT", "ALT", "SUPER")
_SIDED = {"LEFTCTRL": "CTRL", "RIGHTCTRL": "CTRL",
          "LEFTSHIFT": "SHIFT", "RIGHTSHIFT": "SHIFT",
          "LEFTALT": "ALT", "RIGHTALT": "ALT",
          "LEFTMETA": "SUPER", "RIGHTMETA": "SUPER"}


class Chord:
    """One key, or several that have to be held together.

    `parts` is one entry per named key, and each entry is every evdev code
    that satisfies it — one code for RIGHTALT, two for a bare CTRL. The
    chord is down when every part has at least one of its codes held, which
    is what lets CTRL+SPACE fire from either Ctrl.

    A single key is a chord of one, so nothing below has two shapes to
    handle. That is the whole reason this type exists rather than a special
    case beside the old `code: int`.
    """

    __slots__ = ("name", "parts")

    def __init__(self, name: str, parts: tuple[tuple[int, ...], ...]) -> None:
        self.name = name
        self.parts = parts

    @property
    def codes(self) -> frozenset[int]:
        """Everything worth watching for, across every part."""
        return frozenset(code for part in self.parts for code in part)

    @property
    def is_combo(self) -> bool:
        return len(self.parts) > 1

    def satisfied_by(self, held: set[int]) -> bool:
        return all(any(code in held for code in part) for part in self.parts)

    def __repr__(self) -> str:
        return f"Chord({self.name!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Chord) and other.name == self.name

    def __hash__(self) -> int:
        return hash(self.name)


def _one_code(name: str) -> int:
    from evdev import ecodes

    for candidate in (name, f"KEY_{name}"):
        code = getattr(ecodes, candidate, None)
        if isinstance(code, int):
            return code
    raise HotkeyUnavailable(
        f"unknown key name {name!r} (try RIGHTALT, F9, CAPSLOCK, "
        f"or a combination like CTRL+SPACE)")


def _generalise(names: list[str]) -> list[str]:
    """A modifier inside a chord means either side; alone it does not.

    evdev reports the physical key, so pressing Ctrl and slash captures
    LEFTCTRL+SLASH — a binding that works with one hand and not the other,
    which nobody pressing "Ctrl and slash" was asking for. Inside a chord
    the modifier is just "the Ctrl key", so it is widened.

    Alone it is not. The shipped hotkey is RIGHTCTRL, and someone who binds
    a bare modifier has picked *that* key precisely because the other one is
    in constant use for shortcuts — widening it would start a recording
    every time they pressed Ctrl-C. Writing RIGHTCTRL+SLASH by hand still
    pins a chord to one side.
    """
    if len(names) < 2:
        return names
    return [_SIDED.get(name, name) for name in names]


def _key_name(code: int) -> str:
    """`KEY_RIGHTALT` -> `RIGHTALT`, or "" for anything that is not a key."""
    from evdev import ecodes

    names = ecodes.KEY.get(code)
    if isinstance(names, (list, tuple)):
        names = next((n for n in names if str(n).startswith("KEY_")), names[0])
    if not names or not str(names).startswith("KEY_"):
        return ""
    return str(names).replace("KEY_", "", 1)


def canonical_name(names: list[str]) -> str:
    """One spelling per chord: modifiers first, in a fixed order."""
    mods, rest = [], []
    for raw in names:
        name = raw.strip().upper()
        family = name if name in _EITHER_SIDE else _SIDED.get(name, "")
        (mods if family else rest).append((family, name))
    mods.sort(key=lambda pair: _MODIFIER_ORDER.index(pair[0]))
    return "+".join(name for _, name in mods + rest)


def parse_chord(spec: str) -> Chord:
    """`RIGHTALT`, `CTRL+SPACE`, `SUPER+SHIFT+V` — one key or several."""
    names = [part.strip().upper() for part in str(spec).split("+") if part.strip()]
    if not names:
        raise HotkeyUnavailable("no key configured")
    if len(names) != len(set(names)):
        raise HotkeyUnavailable(f"{spec!r} names the same key twice")
    # Built in the canonical order, so `parts` and `name` describe the same
    # chord in the same sequence. Nothing indexes parts today; a name that
    # says CTRL+SPACE over parts that start with SPACE is the kind of thing
    # that is true until something does.
    canonical = canonical_name(names)
    parts = tuple(
        tuple(_one_code(side) for side in _EITHER_SIDE[name])
        if name in _EITHER_SIDE else (_one_code(name),)
        for name in canonical.split("+")
    )
    return Chord(canonical, parts)


def key_code(name: str) -> int:
    """The single code for a single key. Raises on a combination.

    Kept for the two places that genuinely want one code and cannot mean a
    chord; everything about the hotkey itself goes through parse_chord.
    """
    chord = parse_chord(name)
    if chord.is_combo or len(chord.parts[0]) != 1:
        raise HotkeyUnavailable(f"{name!r} is a combination, not a single key")
    return chord.parts[0][0]


def explain_missing(chord: Chord, explicit: list[str] | None = None) -> str:
    """Why this chord cannot be read, naming the part that cannot.

    A combination is only as readable as its least readable key, and saying
    "CTRL+SPACE cannot be read" when every keyboard has Ctrl and none has
    been asked about Space is the same shape as the group message this
    function was written to replace.
    """
    for part, name in zip(chord.parts, chord.name.split("+"), strict=True):
        why = _explain_one(part, name, explicit)
        if why:
            return why if not chord.is_combo else f"{name}: {why}"
    return ""


def _explain_one(codes: tuple[int, ...], name: str,
                 explicit: list[str] | None = None) -> str:
    """Why no device can emit this key, in the words of the actual cause.

    find_devices swallows a failed open with a bare `continue`, so three
    unrelated situations produced one message — and that message named the
    input group, which is right in only one of them. A notification blaming
    group membership on a machine whose user is in the group is worse than no
    notification at all.
    """
    import glob
    import grp
    import os

    from evdev import InputDevice, ecodes

    # Not list_devices(): it globs the nodes and then silently drops the ones
    # it cannot open, so with no `input` group it returns an empty list and
    # this function said "there are no input devices at all" about a machine
    # with twenty-five of them. The nodes are counted first, and whether they
    # can be opened is the next question rather than the same one.
    paths = explicit if explicit is not None else sorted(
        glob.glob("/dev/input/event*"))
    if not paths:
        return "there are no input devices at all"

    denied = 0
    opened: list[str] = []
    for path in paths:
        try:
            dev = InputDevice(path)
        except PermissionError:
            denied += 1
            continue
        except OSError:
            continue
        try:
            emits = dev.capabilities().get(ecodes.EV_KEY, [])
            if any(code in emits for code in codes):
                return ""          # it is there after all
            opened.append(dev.name)
        finally:
            dev.close()

    if denied and not opened:
        try:
            entry = grp.getgrnam("input")
        except KeyError:
            return "there is no `input` group on this system"
        user = os.environ.get("USER") or ""
        listed = user in entry.gr_mem
        holds = entry.gr_gid in os.getgroups()
        if listed and not holds:
            return ("you are in the `input` group but this process started "
                    "before that took effect — re-run the plugin's install.sh, "
                    "which starts the daemon through newgrp, or log out and "
                    "back in")
        if not listed:
            return ("you are not in the `input` group: sudo usermod -aG input "
                    "$USER, then re-run the plugin's install.sh or log out and "
                    "back in")
        return f"{denied} input devices exist but none could be opened"

    if opened:
        return (f"{len(opened)} readable devices, none of which emits {name}: "
                + ", ".join(opened[:4]))
    return f"no readable device emits {name}"


def absent_parts(chord: Chord, explicit: list[str] | None = None) -> list[str]:
    """The parts of this chord that no *readable* keyboard emits.

    Deliberately narrower than explain_missing, which answers "why can this
    not be read" and is right to blame the input group. This answers only
    "can we prove this key is not on any keyboard we were able to open", and
    returns nothing when we could open none — because a shell without the
    group can prove nothing, and refusing a perfectly good key there is the
    mistake this program has made in five other places.

    KEY_QUESTION is why this exists. It is a real evdev code, so the name
    resolves and `config set hotkey.key CTRL+QUESTION` was accepted — and no
    keyboard in the world emits it, because ? is Shift and the SLASH key.
    """
    from evdev import InputDevice, ecodes, list_devices

    emits: set[int] = set()
    opened = 0
    for path in explicit or list_devices():
        try:
            dev = InputDevice(path)
        except (OSError, PermissionError):
            continue
        try:
            opened += 1
            emits |= set(dev.capabilities().get(ecodes.EV_KEY, []))
        finally:
            dev.close()
    if not opened:
        return []
    return [name for part, name in zip(chord.parts, chord.name.split("+"), strict=True)
            if not any(code in emits for code in part)]


def find_devices(chord: Chord, explicit: list[str] | None = None) -> list[Any]:
    """Every readable device that can emit any part of this chord.

    Any, not all: a chord is usually one keyboard, but the modifier and the
    key can sit on different devices — a foot pedal, a macro pad — and the
    listener tracks what is held across all of them anyway. Watching only
    devices that carry the whole chord would silently drop that.
    """
    from evdev import InputDevice, ecodes, list_devices

    wanted = chord.codes
    paths = explicit or list_devices()
    found = []
    for path in paths:
        try:
            dev = InputDevice(path)
        except (OSError, PermissionError):
            continue
        if wanted & set(dev.capabilities().get(ecodes.EV_KEY, [])):
            found.append(dev)
        else:
            dev.close()
    return found


def capture(timeout: float = 10.0, explicit: list[str] | None = None) -> str:
    """Wait for a key press and return what was held, or "" on timeout.

    A chord, not a key: it collects what goes down and returns the whole set
    at the moment the first key comes back up. Pressing one key returns one
    name, which is what this did before and what most people will do.

    Reading, never grabbing: the keys you press still reach whatever has
    focus. That is the same choice the listener makes, and it is why the keys
    worth binding are the ones that do nothing on their own.
    """
    import select

    from evdev import InputDevice, ecodes, list_devices

    devices = []
    for path in explicit or list_devices():
        try:
            dev = InputDevice(path)
        except (OSError, PermissionError):
            continue
        if ecodes.EV_KEY in dev.capabilities():
            devices.append(dev)
        else:
            dev.close()
    if not devices:
        raise HotkeyUnavailable(
            "no readable input devices; the `input` group is granted at login "
            "— re-run the plugin's install.sh to start the daemon through "
            "newgrp, or log out and back in"
        )

    by_fd = {dev.fd: dev for dev in devices}
    deadline = time.monotonic() + timeout
    held: list[str] = []          # in the order pressed, deduplicated
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select(list(by_fd), [], [],
                                        max(0.0, deadline - time.monotonic()))
            for fd in ready:
                for event in by_fd[fd].read():
                    if event.type != ecodes.EV_KEY:
                        continue
                    name = _key_name(event.code)
                    # Keyboard keys only. A mouse reports its buttons as
                    # EV_KEY too, so the first capture picked up a stray
                    # left-click — and a button is not something parse_chord
                    # can resolve anyway.
                    if not name:
                        continue
                    if event.value == _KEY_DOWN:
                        if name not in held:
                            held.append(name)
                    elif event.value == _KEY_UP and held:
                        # The first release ends it: everything down at that
                        # moment is the chord. Waiting for all of them to
                        # come up would let a slow finger add a key that was
                        # never meant to be part of it.
                        return canonical_name(_generalise(held))
        return ""
    finally:
        for dev in devices:
            dev.close()


class HotkeyListener:
    """Calls on_press/on_release (push_to_talk) or on_toggle (toggle)."""

    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
        on_toggle: Callable[[], None],
        # Whether the key can be read at all, as it changes. The listener
        # observes it and says nothing about what to do; the daemon owns that,
        # the same way it owns the press callbacks.
        on_availability: Callable[[bool, str], None] | None = None,
    ) -> None:
        hk = cfg["hotkey"]
        self.chord = parse_chord(hk["key"])
        # The canonical spelling, which is what everything downstream reports
        # and compares against — `hotkey check` asks whether the daemon is
        # bound to the configured key, and "ctrl+space" and "CTRL+SPACE" are
        # the same binding.
        self.key_name: str = self.chord.name
        self.mode: str = hk.get("mode", "push_to_talk")
        self.explicit: list[str] = list(hk.get("devices", []) or [])
        self.rescan_seconds = float(hk.get("rescan_seconds", 5.0))

        self._on_press = on_press
        self._on_release = on_release
        self._on_toggle = on_toggle
        self._on_availability = on_availability

        self._running = threading.Event()
        self._thread: threading.Thread | None = None
        self._devices: list[Any] = []
        # Which of the chord's keys are down, and whether the whole chord was
        # satisfied on the last event. One bool was enough for one key; a
        # combination has to know which parts are held to know when the
        # chord stops being held.
        self._down: set[int] = set()
        self._held = False
        # True once every device is gone, so each transition is said once
        # rather than every half second.
        self._blind = False

    @property
    def device_names(self) -> list[str]:
        return [f"{d.path} {d.name}" for d in self._devices]

    def start(self) -> None:
        devices = find_devices(self.chord, self.explicit or None)
        if not devices:
            raise HotkeyUnavailable(
                f"{self.key_name} cannot be read: "
                + (explain_missing(self.chord, self.explicit or None)
                   or "the devices changed while binding; try again")
            )
        self._devices = devices
        log.info("watching %s (%s) on: %s", self.key_name, self.mode, ", ".join(self.device_names))
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="omavoi-hotkey", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        for dev in self._devices:
            try:
                dev.close()
            except Exception:
                pass
        self._devices = []

    def _loop(self) -> None:
        from evdev import ecodes

        sel = selectors.DefaultSelector()
        for dev in self._devices:
            sel.register(dev, selectors.EVENT_READ)
        last_scan = time.monotonic()

        while self._running.is_set():
            try:
                for key, _ in sel.select(timeout=0.5):
                    dev = key.fileobj
                    try:
                        for event in dev.read():  # type: ignore[union-attr]
                            if (event.type == ecodes.EV_KEY
                                    and event.code in self.chord.codes):
                                self._handle(event.code, event.value)
                    except OSError:
                        # Keyboard unplugged or re-enumerated.
                        log.warning("input device went away: %s", getattr(dev, "path", dev))
                        sel.unregister(dev)
                        if dev in self._devices:
                            self._devices.remove(dev)  # type: ignore[arg-type]
                        if self._held:
                            # A key cannot be released by a keyboard that is
                            # gone, so the chord is broken by definition.
                            self._down.clear()
                            self._held = False
                            self._safe(self._on_release)
                        # Losing the last device is a dead hotkey that looks
                        # exactly like a working one: epoll with nothing
                        # registered returns empty on schedule, so the loop
                        # spun here at 2 Hz, silently, for as long as the
                        # keyboard stayed gone. It is the shape of "it just
                        # stopped working" with nothing in the log after the
                        # first line, so now it is announced.
                        if not self._devices and not self._blind:
                            self._blind = True
                            log.error("no input device can be read; %s is dead "
                                      "until one comes back", self.key_name)
                            self._availability(False,
                                               f"{self.key_name} has no keyboard to "
                                               f"read — it was unplugged or "
                                               f"re-enumerated")
            except Exception:
                log.exception("hotkey loop error")
                time.sleep(0.2)

            if time.monotonic() - last_scan > self.rescan_seconds:
                last_scan = time.monotonic()
                self._rescan(sel)

        sel.close()

    def _rescan(self, sel: selectors.BaseSelector) -> None:
        """Pick up a keyboard that was plugged in after we started."""
        known = {d.path for d in self._devices}
        for dev in find_devices(self.chord, self.explicit or None):
            if dev.path in known:
                dev.close()
                continue
            log.info("new input device: %s %s", dev.path, dev.name)
            self._devices.append(dev)
            sel.register(dev, selectors.EVENT_READ)
        if self._blind and self._devices:
            self._blind = False
            log.info("%s is readable again on: %s", self.key_name,
                     ", ".join(d.path for d in self._devices))
            self._availability(True, f"{self.key_name} works again")

    def _availability(self, ok: bool, detail: str) -> None:
        if self._on_availability is None:
            return
        try:
            self._on_availability(ok, detail)
        except Exception:
            log.exception("hotkey availability callback error")

    def _handle(self, code: int, value: int) -> None:
        # _KEY_HOLD (autorepeat) is not a new press and changes nothing about
        # what is down.
        if value == _KEY_DOWN:
            self._down.add(code)
        elif value == _KEY_UP:
            self._down.discard(code)
        else:
            return

        was, now = self._held, self.chord.satisfied_by(self._down)
        if was == now:
            return
        self._held = now

        if self.mode == "toggle":
            # The edge into the chord only: releasing must not toggle a
            # second time, and a combination has as many release edges as it
            # has keys.
            if now:
                self._safe(self._on_toggle)
            return

        self._safe(self._on_press if now else self._on_release)

    @staticmethod
    def _safe(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("hotkey callback error")
