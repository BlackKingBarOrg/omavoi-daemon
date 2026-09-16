"""The two things every entry point needs before it can do anything.

They lived in cli.py, which the command modules cannot import back without
a cycle — cli.py imports them.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
import wave
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from .. import paths

log = logging.getLogger("omavoi")


def setup_logging(level: str = "INFO", to_file: bool = False) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if to_file:
        # The log has lines like `typed in 0.59s [default]: <what you said>`,
        # so it is as private as the history beside it. FileHandler opens the
        # file itself, hence tighten-after rather than create-private; the
        # directory is 0700, which closes the window from the outside.
        paths.private_dir(paths.state_dir())
        log_path = paths.log_file()
        paths.private_file(log_path)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
        paths.private_file(log_path)
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
        force=True,
    )


def load_wav(path: Path, want_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Read an audio file to float32 mono at `want_rate`, via ffmpeg if needed."""
    # numpy costs 30 ms of the CLI's 76 ms floor and only this function
    # needs it, so every command that never touches audio was paying for
    # it. The annotations are strings under `from __future__`, so the
    # signature does not need it at import time either.
    import numpy as np

    # Checked here, because otherwise the first thing to notice is ffmpeg, and
    # what it says is `Error opening input: No such file or directory` under a
    # line of its own diagnostics — an answer about ffmpeg to a question about
    # a filename.
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    if path.is_dir():
        raise SystemExit(f"{path} is a directory")

    try:
        with wave.open(str(path), "rb") as wav:
            if wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == want_rate:
                raw = wav.readframes(wav.getnframes())
                return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0, want_rate
    except (wave.Error, OSError):
        pass

    if shutil.which("ffmpeg") is None:
        raise SystemExit(f"{path} is not 16 kHz mono WAV and ffmpeg is not installed")
    proc = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error",
         "-i", str(path), "-f", "s16le", "-ac", "1", "-ar", str(want_rate), "-"],
        capture_output=True, check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(f"ffmpeg failed: {proc.stderr.decode('utf-8', 'replace')[:300]}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0, want_rate
