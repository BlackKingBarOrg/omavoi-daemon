"""Every dictation, on disk, with the numbers behind it.

Opacity is a fixable problem: if each take records
its own audio levels, per-segment confidences, what the raw model said, what
post-processing changed it to, and where it was injected, then a bad result
is something you can look at instead of something you re-say and hope.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
import wave
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from statistics import median
from typing import Any

from . import paths

log = logging.getLogger(__name__)


def _write_wav(path: Path, samples: Any, rate: int) -> None:
    """A recording of someone's voice, so 0600 like everything beside it."""
    import numpy as np

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    # After, not before: the wave module opens the file itself, so there is
    # no fd to hand it. The window is one write long and the directory is
    # 0700, which closes it from the outside.
    paths.private_file(path)


def entry_id(entry: dict[str, Any]) -> str:
    """The id a take is addressed by, derived when the line has not got one.

    `record` has written one since the beginning, but this file is appended to
    and never migrated, so it can be older than the code reading it -- and a
    take with no id is one the console can show and not delete. The derivation
    is `record`'s own, so a line written before this and one written after
    answer to the same string.
    """
    if entry.get("id"):
        return str(entry["id"])
    return f"{int(float(entry.get('ts') or 0) * 1000):x}"


@contextmanager
def _locked() -> Iterator[None]:
    """Hold the history lock across a read-everything, write-a-replacement.

    An append needs nothing. The trim after it and `remove` below are both a
    full rewrite, and they run in different processes: the daemon trims as a
    take finishes, while the console deletes the take you right-clicked. Left
    unsynchronised, a take that finishes between the read and the rename is
    not in the file afterwards.

    The lock is its own file rather than the history itself because the
    rewrite ends in a rename: flock follows the inode, so two processes that
    lock "the history file" either side of a replacement hold two unrelated
    locks and neither waits for the other.

    Not reentrant -- flock conflicts with a second open file description even
    in the same process -- so nothing under it takes it again.
    """
    path = paths.history_lock()
    paths.private_dir(path.parent)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        # Closing releases it, and so does dying while holding it -- which a
        # lock represented by the file's existence would not.
        os.close(fd)


class History:
    def __init__(self, cfg: dict[str, Any]) -> None:
        hist = cfg["history"]
        self.enabled = bool(hist.get("enabled", True))
        self.keep = int(hist.get("keep", 500))
        self.keep_audio = int(hist.get("keep_audio", 20))
        self.path = paths.history_file()
        self.audio_dir = paths.recordings_dir()

    def record(self, entry: dict[str, Any], samples: Any = None,
               rate: int = 16000) -> dict[str, Any]:
        if not self.enabled:
            return entry

        entry.setdefault("ts", time.time())
        entry.setdefault("id", f"{int(entry['ts'] * 1000):x}")

        if samples is not None and samples.size and self.keep_audio > 0:
            paths.private_dir(self.audio_dir)
            wav_path = self.audio_dir / f"{entry['id']}.wav"
            try:
                _write_wav(wav_path, samples, rate)
                entry["wav"] = str(wav_path)
            except OSError as exc:
                log.debug("could not store the recording: %s", exc)

        # Every sentence ever dictated lives in this file. It was created at
        # whatever umask said, which on Omarchy is 0644.
        paths.private_dir(self.path.parent)
        paths.private_file(self.path)
        # Taken once around both: the append does not need it, the trim does,
        # and a delete landing between the two would be rewriting a file that
        # is about to be replaced by a rewrite of the version before it.
        with _locked():
            with paths.open_private(self.path, "a") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            self._trim()
        return entry

    def _trim(self) -> None:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) > self.keep:
            kept = lines[-self.keep :]
            self._write_lines(kept)
            lines = kept

        if self.keep_audio <= 0 or not self.audio_dir.is_dir():
            return
        paths.private_dir(self.audio_dir)
        # Keep WAVs only for the most recent takes.
        live = set()
        for line in lines[-self.keep_audio :]:
            try:
                wav = json.loads(line).get("wav")
            except json.JSONDecodeError:
                continue
            if wav:
                live.add(os.path.basename(wav))
        for path in self.audio_dir.glob("*.wav"):
            if path.name not in live:
                try:
                    path.unlink()
                except OSError:
                    pass
            else:
                # Recordings written before this was set keep the mode they
                # were created with, and only the new ones pass through
                # _write_wav. This pass already visits every one of them.
                paths.private_file(path)

    def _write_lines(self, lines: list[str]) -> None:
        """Replace the file with these lines, atomically and privately.

        The replacement carries its own mode, so it has to be created private
        or a rewrite undoes the tightening `record` does on the way past.
        """
        tmp = self.path.with_suffix(".jsonl.tmp")
        with paths.open_private(tmp, "w") as fh:
            for line in lines:
                fh.write(line + "\n")
        tmp.replace(self.path)

    def _drop_audio(self, entries: list[dict[str, Any]]) -> int:
        """The recordings those takes name, where they are still there.

        `_trim` would sweep them at the next take anyway -- it keeps only the
        WAVs the last `keep_audio` entries name -- but "delete this" should
        not leave your voice on the disk until the next time you dictate.
        """
        gone = 0
        for entry in entries:
            wav = entry.get("wav")
            if not wav:
                continue
            path = Path(str(wav))
            # Only inside the recordings directory. The path comes out of a
            # file this program rewrites, and unlinking whatever it says is a
            # far larger promise than the one being kept here.
            if path.parent != self.audio_dir:
                continue
            try:
                path.unlink()
                gone += 1
            except OSError:
                pass
        return gone

    def remove(self, ids: Iterable[str]) -> dict[str, Any]:
        """Delete these takes, and the recordings they point at.

        By id rather than by position: the console lists the last forty takes
        newest first and a take can finish while you are reading them, so an
        index names a different row by the time it arrives.
        """
        wanted = {str(i) for i in ids if str(i)}
        if not wanted:
            return {"removed": 0, "audio": 0, "missing": []}

        kept: list[str] = []
        removed: list[dict[str, Any]] = []
        with _locked():
            try:
                lines = self.path.read_text(encoding="utf-8").splitlines()
            except OSError:
                return {"removed": 0, "audio": 0, "missing": sorted(wanted)}
            for line in lines:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    # Unaddressable, so undeletable: keeping it is the only
                    # answer that cannot delete the wrong take.
                    kept.append(line)
                    continue
                if isinstance(entry, dict) and entry_id(entry) in wanted:
                    removed.append(entry)
                else:
                    kept.append(line)
            if removed:
                self._write_lines(kept)

        return {"removed": len(removed), "audio": self._drop_audio(removed),
                "missing": sorted(wanted - {entry_id(e) for e in removed})}

    def clear(self) -> dict[str, Any]:
        """Every take, and every recording, gone.

        The whole recordings directory rather than the files the entries name:
        a take whose audio `_trim` already swept still names it, and audio
        whose entry was trimmed away is named by nothing at all.
        """
        with _locked():
            try:
                lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines()
                         if ln.strip()]
            except OSError:
                lines = []
            if lines:
                self._write_lines([])

        audio = 0
        if self.audio_dir.is_dir():
            for path in self.audio_dir.glob("*.wav"):
                try:
                    path.unlink()
                    audio += 1
                except OSError:
                    pass
        return {"removed": len(lines), "audio": audio}

    def entries(self, limit: int = 20) -> list[dict[str, Any]]:
        return list(self.iter_entries())[-limit:]

    def iter_entries(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            return
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                # Filled in rather than left absent: every reader addresses a
                # take by this, and one written before ids existed is
                # otherwise a row that cannot be deleted.
                entry["id"] = entry_id(entry)
                yield entry

    def last(self) -> dict[str, Any] | None:
        entries = self.entries(1)
        return entries[-1] if entries else None

    def stats(self) -> dict[str, Any]:
        """Aggregate view — how often takes come back empty, how fast, how loud."""
        entries = list(self.iter_entries())
        if not entries:
            return {"count": 0}
        rtfs, rms, empties, injects = [], [], 0, {}
        for e in entries:
            asr = e.get("asr", {})
            if asr.get("rtf"):
                rtfs.append(asr["rtf"])
            audio = e.get("audio", {})
            if audio.get("rms_dbfs") is not None:
                rms.append(audio["rms_dbfs"])
            if not (e.get("text") or "").strip():
                empties += 1
            method = e.get("inject", {}).get("method", "-")
            injects[method] = injects.get(method, 0) + 1
        return {
            "count": len(entries),
            "empty": empties,
            "empty_rate": round(empties / len(entries), 3),
            # statistics.median, not numpy's: these are lists of Python
            # floats, both take the mean of the middle two on an even count,
            # and one of them is in the stdlib. numpy is still imported by
            # _write_wav, where there is an actual array to convert.
            "median_rtf": round(median(rtfs), 4) if rtfs else None,
            "median_rms_dbfs": round(median(rms), 1) if rms else None,
            "inject_methods": injects,
        }
