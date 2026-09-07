"""Keep a child process's output moving, and keep its last words.

A server started with `stdout=PIPE` that nobody reads runs normally until the
pipe buffer fills, and then blocks in `write()` — forever. The process stays
alive, the kernel still completes TCP handshakes on its listening socket, and
every request goes unanswered. Nothing distinguishes it from a slow model, so
the only cure anyone finds is restarting the daemon by hand.

That is not a hypothetical. Both local servers here were started that way, and
whisper.cpp wedged after roughly a day of use, twice, before the cause was
found. Forcing the pipe down to 4 KiB reproduces it on the seventh take:
2077 bytes of startup banner plus 330 bytes per inference is 4057, and the
next write has nowhere to go.

So the output is drained continuously by a thread that only ever blocks in
`read()`. The last lines are retained because the two callers that used to
read the pipe did so to explain a startup failure — usually a missing ggml
compute backend — and that message is worth more than the bytes it costs.
"""

from __future__ import annotations

import subprocess
import threading
from collections import deque


class ChildLog:
    """Drains a child's stdout so the child can never block writing to it."""

    def __init__(self, proc: subprocess.Popen[bytes], *, keep: int = 400) -> None:
        self._lines: deque[bytes] = deque(maxlen=keep)
        self._lock = threading.Lock()
        self._proc = proc
        self._thread: threading.Thread | None = None
        if proc.stdout is not None:
            self._thread = threading.Thread(
                target=self._pump, name="omavoi-childlog", daemon=True
            )
            self._thread.start()

    def _pump(self) -> None:
        stream = self._proc.stdout
        if stream is None:
            return
        try:
            # Iterating the stream blocks in read(), which is the point: the
            # only thing waiting is this thread, never the child.
            for line in stream:
                with self._lock:
                    self._lines.append(line)
        except (OSError, ValueError):
            # The stream was closed under us — close() does exactly that.
            pass

    def text(self, *, timeout: float = 1.0) -> bytes:
        """What the child has said, most recent `keep` lines.

        Waits briefly first: a caller asking this has just seen the process
        exit, and the pump may still be a line behind.
        """
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout)
        with self._lock:
            return b"".join(self._lines)
