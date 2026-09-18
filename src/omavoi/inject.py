"""Getting text into the focused window.

Three paths, because none of them works everywhere:

  wtype      — virtual-keyboard protocol. Correct for native Wayland apps.
               X11 clients never receive the keymap it installs, so they
               decode its keycodes against the system layout and a sentence
               arrives as "1234567890-=".
  clipboard  — wl-copy plus a synthetic paste shortcut. Fine between Wayland
               apps. Useless for XWayland wherever the compositor's X11
               clipboard bridge is not working, and then it fails silently:
               the copy succeeds, the keystroke lands, and the X client finds
               an empty selection.
  xdotool    — XTEST, the way X11 has always injected input, using the X
               server's own keymap. Only reaches X11 clients.

The keymap is the whole story for X11. wtype's synthetic map does not reach
those clients, so both the text it types and the Ctrl+V it sends arrive as
different keys — which is why pasting into an X11 app appeared to do nothing
while the clipboard held the text all along. XTEST uses the server's own map
and gets both right.

X11 clients are typed into with XTEST, character by character. Sending the
payload by clipboard and only the keystroke by XTEST would be faster, and it
was tried: it pastes nothing, because the compositor's Wayland-to-X11
clipboard bridge does not carry the selection either. Copying by hand and
pasting works only because that copy already happened on the X11 side.

So the slow route is the one that works. It costs about 12 ms a character,
and a focus change part-way through will scatter the rest of the text into
whatever window took over.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .window import Window

log = logging.getLogger(__name__)


@dataclass(slots=True)
class InjectResult:
    ok: bool
    method: str
    seconds: float
    error: str = ""
    fell_back: bool = False
    # How the paste keystroke was sent, when one was. Without this in the
    # record, "route=clipboard ok=True" says nothing about which of the two
    # halves failed.
    paste_via: str = ""
    chars: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok, "method": self.method,
            "seconds": round(self.seconds, 3),
            "error": self.error, "fell_back": self.fell_back,
            "paste_via": self.paste_via, "chars": self.chars,
        }


# The names Xlib and xkb spell with capitals. Anything else that is more than
# one character is capitalised, which covers Tab, Escape, Home, Up, Down.
_KEYSYMS = {"RETURN": "Return", "ENTER": "Return", "KP_ENTER": "KP_Enter",
            "SPACE": "space", "TAB": "Tab", "ESCAPE": "Escape", "ESC": "Escape",
            "BACKSPACE": "BackSpace", "DELETE": "Delete"}

class Injector:
    def __init__(self, cfg: dict[str, Any]) -> None:
        inject = cfg["inject"]
        self.method: str = inject.get("method", "auto")
        self.clipboard_classes = [c.lower() for c in inject.get("clipboard_classes", [])]
        self.restore_after = float(inject.get("restore_clipboard_after", 4.0))
        self.wtype_delay_ms = int(inject.get("wtype_delay_ms", 0))
        self.default_paste_key: str = inject.get("paste_key", "CTRL+V")
        # What a typed newline becomes. Return, unless the mode says otherwise
        # -- a chat window wants SHIFT+RETURN, or the first line gets sent.
        self.default_newline_key: str = inject.get("newline_key", "RETURN")
        self.paste_method: str = inject.get("paste_method", "") or ""
        self.xdotool_delay_ms = int(inject.get("xdotool_delay_ms", 12))
        # 150, not the 60 this used to say: XWayland mirrors the Wayland
        # selection lazily. The number lives in config.DEFAULTS with the
        # reason beside it, and a second, different one here taught
        # anyone reading this file the wrong answer.
        self.paste_settle_ms = int(inject.get("paste_settle_ms", 150))

    # -- routing -----------------------------------------------------------

    def choose(self, win: Window, profile: dict[str, Any]) -> str:
        override = str(profile.get("inject", "") or "")

        # An X11 client cannot be reached by wtype or by the clipboard, so a
        # mode asking for either is asking for silence. Overriding the route
        # per mode is legitimate; overriding it into one that cannot arrive is
        # a mistake, and it stayed invisible for hours because the override
        # short-circuited the detection below before it ever ran.
        if win.xwayland and override in ("wtype", "clipboard"):
            if shutil.which("xdotool"):
                log.warning(
                    "mode asks for %s but %s is an X11 window, which cannot "
                    "receive it — using xdotool", override, win.cls or "?")
                return "xdotool"

        if override in ("wtype", "clipboard", "xdotool"):
            return override
        if self.method in ("wtype", "clipboard", "xdotool"):
            return self.method

        # XWayland clients never see the keymap wtype installs for its virtual
        # keyboard, so they decode its keycodes against the system layout and
        # type digits instead of your words. Detecting the client is better
        # than listing them: it covers every X11 app without anyone keeping a
        # list up to date.
        if win.xwayland:
            # Never wtype here: its keymap is exactly what X11 clients cannot
            # read. Never the clipboard either: the bridge does not carry it.
            # XTEST typing is slow and it is the only thing that arrives.
            if shutil.which("xdotool"):
                return "xdotool"
            return "clipboard"

        haystack = f"{win.cls} {win.title}".lower()
        if any(c and c in haystack for c in self.clipboard_classes):
            return "clipboard"
        return "wtype"

    def _paste_via(self, win: Window, profile: dict[str, Any]) -> str:
        explicit = str(profile.get("paste_method", "") or "")
        if explicit:
            return explicit
        if win.xwayland:
            return "xdotool"
        return self.paste_method or "shortcut"

    def inject(self, text: str, win: Window, profile: dict[str, Any] | None = None) -> InjectResult:
        profile = profile or {}
        if not text:
            return InjectResult(True, "noop", 0.0)

        method = self.choose(win, profile)
        started = time.monotonic()
        try:
            via = self._deliver(method, text, profile, win)
            return InjectResult(True, method, time.monotonic() - started,
                                paste_via=via, chars=len(text))
        except Exception as exc:
            log.warning("%s injection failed: %s", method, exc)
            # One retry on a route that could plausibly work instead. A
            # dropped transcript is worse than a clipboard we had to touch.
            other = self._fallback_for(method, win)
            if other is None:
                return InjectResult(False, method, time.monotonic() - started, error=str(exc))
            try:
                via = self._deliver(other, text, profile, win)
                return InjectResult(True, other, time.monotonic() - started,
                                    error=str(exc), fell_back=True,
                                    paste_via=via, chars=len(text))
            except Exception as exc2:
                return InjectResult(False, method, time.monotonic() - started,
                                    error=f"{exc} / {exc2}")

    def _deliver(self, method: str, text: str, profile: dict[str, Any],
                 win: Window) -> str:
        if method == "wtype":
            self._wtype(text, profile)
            return ""
        if method == "xdotool":
            self._xdotool(text, profile)
            return "xtest-type"
        via = self._paste_via(win, profile)
        self._clipboard(text, profile, via)
        return via

    def _fallback_for(self, method: str, win: Window) -> str | None:
        """The next thing worth trying, or None when nothing else can work."""
        if win.xwayland:
            # wtype is not a fallback here: its keymap is the original bug.
            if method == "clipboard":
                return "xdotool" if shutil.which("xdotool") else None
            return "clipboard"
        return "clipboard" if method == "wtype" else "wtype"

    def _newline_key(self, profile: dict[str, Any]) -> str:
        return str(profile.get("newline_key") or self.default_newline_key)

    def _typed_segments(self, text: str, profile: dict[str, Any],
                        type_one: Callable[[str], None], method: str) -> None:
        """Type `text`, sending each newline as the mode's newline key.

        `xdotool type` and wtype both turn a literal newline into Return,
        which in an editor is a newline and in a chat window is the send
        key -- so a two-line take posted its first line and left the second
        sitting in the box. Typing the lines one at a time and sending
        SHIFT+RETURN (or whatever the mode names) between them is what a
        person does to get a second line into a chat window.
        """
        if "\n" not in text:
            type_one(text)
            return
        key = self._newline_key(profile)
        lines = text.split("\n")
        for i, line in enumerate(lines):
            if line:
                type_one(line)
            if i < len(lines) - 1:
                self._send_paste(key, method)

    def _xdotool(self, text: str, profile: dict[str, Any] | None = None) -> None:
        """XTEST, for X11 clients. No clipboard, no custom keymap."""
        if shutil.which("xdotool") is None:
            raise RuntimeError("xdotool is not installed")
        self._typed_segments(text, profile or {}, self._xdotool_type, "xdotool")

    def _xdotool_type(self, text: str) -> None:
        env = dict(os.environ)
        env.setdefault("DISPLAY", ":0")
        proc = subprocess.run(
            ["xdotool", "type", "--clearmodifiers",
             "--delay", str(self.xdotool_delay_ms), "--", text],
            capture_output=True, timeout=60, env=env,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()
                               or "xdotool exited non-zero")

    # -- backends ----------------------------------------------------------

    def _wtype(self, text: str, profile: dict[str, Any] | None = None) -> None:
        if shutil.which("wtype") is None:
            raise RuntimeError("wtype is not installed")
        self._typed_segments(text, profile or {}, self._wtype_type, "wtype")

    def _wtype_type(self, text: str) -> None:
        # Text goes in on stdin, not argv: no escaping, no ARG_MAX limit,
        # and text starting with '-' can't be read as a flag.
        argv = ["wtype"]
        if self.wtype_delay_ms:
            argv += ["-d", str(self.wtype_delay_ms)]
        argv.append("-")
        proc = subprocess.run(argv, input=text.encode(), capture_output=True, timeout=30, check=False)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip() or "wtype exited non-zero")

    def _clipboard(self, text: str, profile: dict[str, Any],
                   paste_via: str = "shortcut") -> None:
        if shutil.which("wl-copy") is None:
            raise RuntimeError("wl-clipboard is not installed")

        saved = self._read_clipboard() if self.restore_after > 0 else None
        subprocess.run(["wl-copy", "--"], input=text.encode(), check=True, timeout=5)
        # Give the compositor a moment to publish the new selection before
        # the paste keystroke asks for it.
        time.sleep(self.paste_settle_ms / 1000.0)
        self._send_paste(str(profile.get("paste_key", self.default_paste_key)), paste_via)

        if saved is not None:
            threading.Timer(self.restore_after, self._restore_clipboard, args=(saved,)).start()

    @staticmethod
    def _read_clipboard() -> bytes | None:
        if shutil.which("wl-paste") is None:
            return None
        try:
            proc = subprocess.run(
                ["wl-paste", "--no-newline"], capture_output=True, timeout=2,
                check=False,
            )
            return proc.stdout if proc.returncode == 0 else b""
        except (subprocess.SubprocessError, OSError):
            return None

    @staticmethod
    def _restore_clipboard(saved: bytes) -> None:
        try:
            if saved:
                subprocess.run(["wl-copy", "--"], input=saved, timeout=5, check=False)
            else:
                subprocess.run(["wl-copy", "--clear"], timeout=5, check=False)
        except (subprocess.SubprocessError, OSError) as exc:
            log.debug("could not restore the clipboard: %s", exc)

    @staticmethod
    def _send_paste(combo: str, method: str = "shortcut") -> None:
        parts = [p.strip().upper() for p in combo.split("+") if p.strip()]
        if not parts:
            raise ValueError(f"invalid paste shortcut {combo!r}")
        key, mods = parts[-1], parts[:-1]
        # X keysyms are case-sensitive and multi-letter: "return" is not a
        # key, "Return" is. A single letter is lowered below; a name is
        # written the way Xlib spells it.
        key = _KEYSYMS.get(key, key if len(key) == 1 else key.capitalize())

        if method == "xdotool":
            if shutil.which("xdotool") is None:
                raise RuntimeError("xdotool is not installed")
            combo_x = "+".join([*(m.lower() for m in mods), key.lower() if len(key) == 1 else key])
            env = dict(os.environ)
            env.setdefault("DISPLAY", ":0")
            proc = subprocess.run(["xdotool", "key", "--clearmodifiers", combo_x],
                                  capture_output=True, timeout=10, env=env, check=False)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()
                                   or "xdotool key failed")
            return

        if method == "wtype":
            if shutil.which("wtype") is None:
                raise RuntimeError("wtype is not installed")
            argv: list[str] = []
            for mod in mods:
                argv += ["-M", mod.lower()]
            argv += ["-k", key.lower() if len(key) == 1 else key]
            for mod in reversed(mods):
                argv += ["-m", mod.lower()]
            proc = subprocess.run(["wtype", *argv], capture_output=True, timeout=5, check=False)
            if proc.returncode != 0:
                raise RuntimeError(proc.stderr.decode("utf-8", "replace").strip()
                                   or "wtype paste failed")
            return

        # Hyprland 0.56+ takes a Lua expression, not a bare dispatcher name.
        lua = (
            "hl.dsp.send_shortcut({{ mods = {mods!r}, key = {key!r}, window = 'activewindow' }})"
        ).format(mods=" ".join(mods), key=key).replace("'", '"')
        proc = subprocess.run(
            ["hyprctl", "dispatch", lua], capture_output=True, timeout=5,
            check=False,
        )
        out = proc.stdout.decode("utf-8", "replace").strip()
        if proc.returncode != 0 or out != "ok":
            raise RuntimeError(f"hyprctl send_shortcut failed: {out or proc.stderr.decode()[:120]}")
