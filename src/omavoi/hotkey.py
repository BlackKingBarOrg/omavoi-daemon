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


def key_code(name: str) -> int:
    from evdev import ecodes

    name = name.strip().upper()
    for candidate in (name, f"KEY_{name}"):
        code = getattr(ecodes, candidate, None)
        if isinstance(code, int):
            return code
    raise HotkeyUnavailable(f"unknown key name {name!r} (try RIGHTALT, F9, CAPSLOCK)")


def explain_missing(code: int, name: str, explicit: list[str] | None = None) -> str:
    """Why no device can emit this key, in the words of the actual cause.

    find_devices swallows a failed open with a bare `continue`, so three
    unrelated situations produced one message — and that message named the
    input group, which is right in only one of them. A notification blaming
    group membership on a machine whose user is in the group is worse than no
    notification at all.
    """
    import grp
    import os

    from evdev import InputDevice, ecodes, list_devices

    paths = explicit if explicit is not None else list_devices()
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
            if code in dev.capabilities().get(ecodes.EV_KEY, []):
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
                    "before that took effect — log out and back in")
        if not listed:
            return ("you are not in the `input` group: sudo usermod -aG input "
                    "$USER, then log out and back in")
        return f"{denied} input devices exist but none could be opened"

    if opened:
        return (f"{len(opened)} readable devices, none of which emits {name}: "
                + ", ".join(opened[:4]))
    return f"no readable device emits {name}"


def find_devices(code: int, explicit: list[str] | None = None) -> list[Any]:
    """Every readable device that can emit this key."""
    from evdev import InputDevice, ecodes, list_devices

    paths = explicit or list_devices()
    found = []
    for path in paths:
        try:
            dev = InputDevice(path)
        except (OSError, PermissionError):
            continue
        if code in dev.capabilities().get(ecodes.EV_KEY, []):
            found.append(dev)
        else:
            dev.close()
    return found


def capture(timeout: float = 10.0, explicit: list[str] | None = None) -> str:
    """Wait for one key press and return its evdev name, or "" on timeout.

    Reading, never grabbing: the key you press still reaches whatever has
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
            "no readable input devices; membership of the `input` group takes "
            "effect at your next login"
        )

    by_fd = {dev.fd: dev for dev in devices}
    deadline = time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            ready, _, _ = select.select(list(by_fd), [], [],
                                        max(0.0, deadline - time.monotonic()))
            for fd in ready:
                for event in by_fd[fd].read():
                    if event.type != ecodes.EV_KEY or event.value != 1:
                        continue
                    # Keyboard keys only. A mouse reports its buttons as
                    # EV_KEY too, so the first capture picked up a stray
                    # left-click — and a button is not something the config's
                    # key_code() can resolve anyway.
                    names = ecodes.KEY.get(event.code)
                    if isinstance(names, (list, tuple)):
                        names = next((n for n in names
                                      if str(n).startswith("KEY_")), names[0])
                    if not names or not str(names).startswith("KEY_"):
                        continue
                    # KEY_RIGHTALT -> RIGHTALT, which is what the config takes.
                    return str(names).replace("KEY_", "", 1)
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
        self.key_name: str = hk["key"]
        self.code = key_code(self.key_name)
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
        self._held = False
        # True once every device is gone, so each transition is said once
        # rather than every half second.
        self._blind = False

    @property
    def device_names(self) -> list[str]:
        return [f"{d.path} {d.name}" for d in self._devices]

    def start(self) -> None:
        devices = find_devices(self.code, self.explicit or None)
        if not devices:
            raise HotkeyUnavailable(
                f"{self.key_name} cannot be read: "
                + (explain_missing(self.code, self.key_name,
                                   self.explicit or None)
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
                            if event.type == ecodes.EV_KEY and event.code == self.code:
                                self._handle(event.value)
                    except OSError:
                        # Keyboard unplugged or re-enumerated.
                        log.warning("input device went away: %s", getattr(dev, "path", dev))
                        sel.unregister(dev)
                        if dev in self._devices:
                            self._devices.remove(dev)  # type: ignore[arg-type]
                        if self._held:
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
        for dev in find_devices(self.code, self.explicit or None):
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

    def _handle(self, value: int) -> None:
        if self.mode == "toggle":
            if value == _KEY_DOWN:
                self._safe(self._on_toggle)
            return

        if value == _KEY_DOWN and not self._held:
            self._held = True
            self._safe(self._on_press)
        elif value == _KEY_UP and self._held:
            self._held = False
            self._safe(self._on_release)
        # _KEY_HOLD (autorepeat) is ignored — it is not a new press.

    @staticmethod
    def _safe(fn: Callable[[], None]) -> None:
        try:
            fn()
        except Exception:
            log.exception("hotkey callback error")
