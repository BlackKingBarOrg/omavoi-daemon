"""Always-on ring capture with pre-roll.

The reason this is a ring buffer and not "spawn pw-record when the key goes
down": PipeWire needs 100-300 ms to negotiate and start a stream, and people
start talking the instant they press the key. Start-on-press throws that
speech away — it is the single biggest cause of dropped leading words.

So pw-record runs continuously into a small circular buffer, and a capture
simply slices out [keydown - preroll, keyup + tail]. The first syllable is
already in memory before the key is even registered.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .childlog import ChildLog

log = logging.getLogger(__name__)

_CHUNK_FRAMES = 1024  # ~64 ms at 16 kHz


class AudioUnavailable(RuntimeError):
    pass


@dataclass(slots=True)
class Capture:
    """One slice of the ring, with the numbers needed to diagnose a bad take."""

    samples: np.ndarray
    rate: int
    preroll_seconds: float
    tail_seconds: float
    truncated: bool  # the ring wrapped before we could read it — audio was lost

    @property
    def seconds(self) -> float:
        return self.samples.size / self.rate

    @property
    def peak(self) -> float:
        return float(np.abs(self.samples).max()) if self.samples.size else 0.0

    @property
    def rms_dbfs(self) -> float:
        if not self.samples.size:
            return -120.0
        rms = float(np.sqrt(np.mean(np.square(self.samples))))
        return 20.0 * np.log10(max(rms, 1e-9))

    @property
    def peak_dbfs(self) -> float:
        return 20.0 * np.log10(max(self.peak, 1e-9))


class RingCapture:
    """Continuously records into a circular buffer; hands out slices on demand."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        audio = cfg["audio"]
        self.rate = int(audio["rate"])
        self.target = str(audio.get("target", ""))
        self.preroll = float(audio["preroll_seconds"])
        self.tail = float(audio["tail_seconds"])
        self.max_seconds = float(audio["max_seconds"])

        # The ring must outlast the longest utterance plus its pre-roll.
        ring_seconds = self.max_seconds + self.preroll + self.tail + 2.0
        self._size = int(ring_seconds * self.rate)
        self._ring = np.zeros(self._size, dtype=np.int16)
        self._written = 0  # absolute frame count, never wraps

        self._proc: subprocess.Popen[bytes] | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._running = threading.Event()
        self._first_audio = threading.Event()
        self._restarts = 0

    # -- lifecycle ---------------------------------------------------------

    def _argv(self) -> list[str]:
        argv = [
            "pw-record",
            f"--rate={self.rate}",
            "--channels=1",
            "--format=s16",
            "--container=raw",
            "--latency=20ms",
            "--media-role=Communication",
            "-P", "node.name=omavoi",
            "-P", "media.class=Stream/Input/Audio",
        ]
        if self.target:
            argv.append(f"--target={self.target}")
        argv.append("-")
        return argv

    def start(self) -> None:
        if shutil.which("pw-record") is None:
            raise AudioUnavailable("pw-record not found — install pipewire-tools")
        self._running.set()
        self._thread = threading.Thread(target=self._supervise, name="omavoi-audio", daemon=True)
        self._thread.start()
        if not self._first_audio.wait(timeout=5.0):
            raise AudioUnavailable("no audio arrived within 5s — check the input device")

    def stop(self) -> None:
        self._running.clear()
        self._kill()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _kill(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=1.0)

    def _supervise(self) -> None:
        """Keep pw-record alive; a USB mic that re-enumerates shouldn't kill us."""
        backoff = 0.2
        while self._running.is_set():
            try:
                self._pump_once()
            except Exception:
                log.exception("audio pump crashed")
            if not self._running.is_set():
                break
            self._restarts += 1
            log.warning("pw-record exited, restarting in %.1fs (restart #%d)", backoff, self._restarts)
            time.sleep(backoff)
            backoff = min(backoff * 2, 5.0)

    def _pump_once(self) -> None:
        self._proc = subprocess.Popen(
            self._argv(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0
        )
        # stdout is the audio and is read below. stderr was read only in the
        # finally — which is to say, only once this loop had already ended.
        # A child that fills an unread pipe blocks in write(), and a blocked
        # pw-record stops producing audio, so the read below never returns and
        # the finally never runs: a microphone that dies quietly and cannot
        # recover. It writes almost nothing here in normal use (two bytes in
        # ten seconds, measured), so this is a small pipe and a long wait
        # rather than a likely one — but the failure has no way out.
        errlog = ChildLog(self._proc, keep=80, stream=self._proc.stderr)
        stream = self._proc.stdout
        assert stream is not None
        nbytes = _CHUNK_FRAMES * 2
        try:
            while self._running.is_set():
                chunk = stream.read(nbytes)
                if not chunk:
                    break
                if len(chunk) % 2:
                    chunk = chunk[:-1]
                self._append(np.frombuffer(chunk, dtype="<i2"))
                self._first_audio.set()
        finally:
            self._kill()
            if self._proc is not None:
                err = errlog.text().decode("utf-8", "replace").strip()
                if err:
                    log.debug("pw-record: %s", err)
                if self._proc.stderr is not None:
                    self._proc.stderr.close()
                if self._proc.stdout is not None:
                    self._proc.stdout.close()

    def _append(self, frames: np.ndarray) -> None:
        n = frames.size
        if n == 0:
            return
        with self._lock:
            pos = self._written % self._size
            end = pos + n
            if end <= self._size:
                self._ring[pos:end] = frames
            else:
                split = self._size - pos
                self._ring[pos:] = frames[:split]
                self._ring[: end - self._size] = frames[split:]
            self._written += n

    # -- capture -----------------------------------------------------------

    @property
    def healthy(self) -> bool:
        return self._first_audio.is_set() and (
            self._proc is not None and self._proc.poll() is None
        )

    def mark(self) -> int:
        """Return the ring position to treat as 'now', minus the pre-roll."""
        with self._lock:
            return max(0, self._written - int(self.preroll * self.rate))

    def take(self, start: int) -> Capture:
        """Slice from `start` to now, after waiting out the tail padding."""
        if self.tail > 0:
            time.sleep(self.tail)
        with self._lock:
            end = self._written
            oldest = max(0, end - self._size)
            truncated = start < oldest
            begin = max(start, oldest)
            n = end - begin
            if n <= 0:
                return Capture(np.zeros(0, np.float32), self.rate, self.preroll, self.tail, truncated)
            out = np.empty(n, dtype=np.int16)
            pos = begin % self._size
            if pos + n <= self._size:
                out[:] = self._ring[pos : pos + n]
            else:
                split = self._size - pos
                out[:split] = self._ring[pos:]
                out[split:] = self._ring[: n - split]

        samples = out.astype(np.float32) / 32768.0
        return Capture(samples, self.rate, self.preroll, self.tail, truncated)

    def level(self, seconds: float = 0.25) -> float:
        """Current peak level — for the OSD meter and `omavoi doctor`."""
        n = int(seconds * self.rate)
        with self._lock:
            end = self._written
            begin = max(0, end - n, end - self._size)
            if end <= begin:
                return 0.0
            pos = begin % self._size
            count = end - begin
            if pos + count <= self._size:
                window = self._ring[pos : pos + count]
            else:
                window = np.concatenate((self._ring[pos:], self._ring[: count - (self._size - pos)]))
        return float(np.abs(window).max()) / 32768.0
