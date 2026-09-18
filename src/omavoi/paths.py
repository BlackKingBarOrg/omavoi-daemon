"""XDG paths, in one place — and the two that create them privately.

secrets.toml is created 0600 before anything is written to it, and
config.toml is deliberately safe to share. Between those two stood the
state directory, at 0755, holding history.jsonl and omavoi.log at 0644 —
every sentence ever dictated, readable by every account on the machine. The
log has lines like `typed in 0.59s [default]: <what you said>` and the
history has the text, the raw transcript and the path to the audio.

So `private_dir` and `private_file` here, used wherever those are written.
They also tighten what they find: chmod-on-create fixes nothing for an
install that has been running for a week.
"""

from __future__ import annotations

import contextlib
import os
import stat
from pathlib import Path


def _xdg(var: str, default: str) -> Path:
    value = os.environ.get(var)
    return Path(value) if value else Path.home() / default


def private_dir(path: Path) -> Path:
    """Make sure this directory exists and only this user can read it."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    # mkdir's mode is ignored for a directory that already exists, and umask
    # can widen a new one, so the mode is asserted rather than requested.
    with contextlib.suppress(OSError):
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            path.chmod(0o700)
    return path


def private_file(path: Path) -> Path:
    """Tighten a file to 0600 if it is looser. Silent if it does not exist."""
    with contextlib.suppress(OSError):
        if stat.S_IMODE(path.stat().st_mode) & 0o077:
            path.chmod(0o600)
    return path


def open_private(path: Path, mode: str = "a"):
    """Open for writing, creating at 0600 rather than at whatever umask says.

    Created private rather than created and then chmodded, which leaves a
    window where it is readable — the same reason secrets.py does this.
    """
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if "a" in mode else os.O_TRUNC)
    fd = os.open(path, flags, 0o600)
    return os.fdopen(fd, mode, encoding="utf-8")


def config_dir() -> Path:
    return _xdg("XDG_CONFIG_HOME", ".config") / "omavoi"


def data_dir() -> Path:
    return _xdg("XDG_DATA_HOME", ".local/share") / "omavoi"


def state_dir() -> Path:
    return _xdg("XDG_STATE_HOME", ".local/state") / "omavoi"


def cache_dir() -> Path:
    return _xdg("XDG_CACHE_HOME", ".cache") / "omavoi"


def runtime_dir() -> Path:
    value = os.environ.get("XDG_RUNTIME_DIR")
    return Path(value) if value else Path(f"/run/user/{os.getuid()}")


def config_file() -> Path:
    return config_dir() / "config.toml"


def secrets_file() -> Path:
    return config_dir() / "secrets.toml"


def socket_file() -> Path:
    return runtime_dir() / "omavoi.sock"


def history_file() -> Path:
    return state_dir() / "history.jsonl"


def history_lock() -> Path:
    """What a rewrite of the history holds while it replaces the file.

    Separate from the history itself because the rewrite renames over it, and
    a lock taken on the file follows the inode rather than the name.
    """
    return state_dir() / "history.lock"


def log_file() -> Path:
    return state_dir() / "omavoi.log"


def recordings_dir() -> Path:
    return cache_dir() / "recordings"
