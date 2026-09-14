"""A child that writes more than its pipe holds must not be able to block.

This is the failure that cost the most: whisper.cpp was started with stdout
piped and nothing reading it, so after enough takes the 64 KiB buffer filled
and the server blocked in write() — alive, still accepting connections,
answering nothing, curable only by killing it. pw-record had the same shape on
stderr. The test uses a small pipe so it takes a second rather than a day.
"""
from __future__ import annotations

import fcntl
import subprocess
import sys
import time

from omavoi.childlog import ChildLog

F_SETPIPE_SZ = 1031
CHATTY = (
    "import sys\n"
    "for i in range(4000):\n"
    "    sys.stdout.write('line %d ' % i + 'x' * 60 + '\\n')\n"
    "    sys.stdout.flush()\n"
    "sys.stdout.write('DONE\\n'); sys.stdout.flush()\n"
)


def _spawn():
    proc = subprocess.Popen([sys.executable, "-c", CHATTY],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    fcntl.fcntl(proc.stdout.fileno(), F_SETPIPE_SZ, 4096)
    return proc


def test_without_a_drain_the_child_blocks():
    """The bug itself, so the fix below is measured against something."""
    proc = _spawn()
    time.sleep(1.5)
    assert proc.poll() is None, "the child should still be blocked on write()"
    proc.kill()
    proc.wait()


def test_a_drained_child_runs_to_completion():
    proc = _spawn()
    log = ChildLog(proc)
    assert proc.wait(timeout=20) == 0, "a drained child must not block"
    assert b"DONE" in log.text()


def test_the_tail_is_kept_and_bounded():
    """The two callers that used to read the pipe did so to explain a startup
    failure, so the last lines have to survive — but not all of them."""
    proc = _spawn()
    log = ChildLog(proc, keep=50)
    proc.wait(timeout=20)
    kept = log.text()
    assert b"DONE" in kept, "the most recent lines are the useful ones"
    assert b"line 0 " not in kept, "keep= has to bound what is held"
    assert len(kept.splitlines()) <= 50


def test_it_can_drain_a_stream_that_is_not_stdout():
    """pw-record's stdout is the audio; stderr is the one nobody reads."""
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import sys\n"
         "for i in range(4000): sys.stderr.write('e' * 70 + '\\n')\n"
         "sys.stderr.flush()\n"
         "sys.stdout.write('audio'); sys.stdout.flush()\n"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    fcntl.fcntl(proc.stderr.fileno(), F_SETPIPE_SZ, 4096)
    log = ChildLog(proc, stream=proc.stderr)
    assert proc.wait(timeout=20) == 0
    assert log.text().count(b"\n") > 0
