"""Every dictation, on disk, with the numbers behind it.

Opacity is a fixable problem: if each take records
its own audio levels, per-segment confidences, what the raw model said, what
post-processing changed it to, and where it was injected, then a bad result
is something you can look at instead of something you re-say and hope.
"""

from __future__ import annotations

import json
import logging
import os
import time
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from . import paths

log = logging.getLogger(__name__)


def _write_wav(path: Path, samples: np.ndarray, rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())


class History:
    def __init__(self, cfg: dict[str, Any]) -> None:
        hist = cfg["history"]
        self.enabled = bool(hist.get("enabled", True))
        self.keep = int(hist.get("keep", 500))
        self.keep_audio = int(hist.get("keep_audio", 20))
        self.path = paths.history_file()
        self.audio_dir = paths.recordings_dir()

    def record(self, entry: dict[str, Any], samples: np.ndarray | None = None,
               rate: int = 16000) -> dict[str, Any]:
        if not self.enabled:
            return entry

        entry.setdefault("ts", time.time())
        entry.setdefault("id", f"{int(entry['ts'] * 1000):x}")

        if samples is not None and samples.size and self.keep_audio > 0:
            self.audio_dir.mkdir(parents=True, exist_ok=True)
            wav_path = self.audio_dir / f"{entry['id']}.wav"
            try:
                _write_wav(wav_path, samples, rate)
                entry["wav"] = str(wav_path)
            except OSError as exc:
                log.debug("could not store the recording: %s", exc)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
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
            tmp = self.path.with_suffix(".jsonl.tmp")
            tmp.write_text("\n".join(kept) + "\n", encoding="utf-8")
            tmp.replace(self.path)
            lines = kept

        if self.keep_audio <= 0 or not self.audio_dir.is_dir():
            return
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
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

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
            "median_rtf": round(float(np.median(rtfs)), 4) if rtfs else None,
            "median_rms_dbfs": round(float(np.median(rms)), 1) if rms else None,
            "inject_methods": injects,
        }
