"""What window is focused — the context every downstream rule keys off."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Window:
    cls: str = ""
    title: str = ""
    address: str = ""
    pid: int = 0
    # X11 clients running under XWayland. They do not receive the custom
    # keymap wtype installs, so synthetic keystrokes land as whatever those
    # keycodes mean in the system layout — usually digits.
    xwayland: bool = False

    @property
    def known(self) -> bool:
        return bool(self.cls or self.title)

    def as_dict(self) -> dict[str, Any]:
        return {"class": self.cls, "title": self.title, "address": self.address,
                "pid": self.pid, "xwayland": self.xwayland}


def active_window() -> Window:
    if shutil.which("hyprctl") is None:
        return Window()
    try:
        out = subprocess.run(
            ["hyprctl", "-j", "activewindow"],
            capture_output=True, timeout=1.0, check=False,
        )
        data = json.loads(out.stdout or b"{}")
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError) as exc:
        log.debug("activewindow query failed: %s", exc)
        return Window()
    if not isinstance(data, dict):
        return Window()
    return Window(
        cls=str(data.get("class", "") or ""),
        title=str(data.get("title", "") or ""),
        address=str(data.get("address", "") or ""),
        pid=int(data.get("pid", 0) or 0),
        xwayland=bool(data.get("xwayland", False)),
    )


# Unreferenced on purpose: window matching is switched off in the console
# (ModesView.showWindowMatch) but whole in the daemon, and this is the part
# that picks a mode from the focused window. A dead-code scan will find it
# every time, so this is here to say it is waiting rather than abandoned.
def match_profile(win: Window, profiles: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Longest matching key wins, so "org.wezfurlong.wezterm" beats "wez"."""
    haystack = f"{win.cls} {win.title}".lower()
    best_key = ""
    best: dict[str, Any] = {}
    for key, value in profiles.items():
        if not isinstance(value, dict):
            continue
        if key.lower() in haystack and len(key) > len(best_key):
            best_key, best = key, value
    return best_key, best
