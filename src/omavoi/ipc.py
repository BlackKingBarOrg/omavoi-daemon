"""The socket, and the two calls that go across it.

These were the top of daemon.py, which is the right place to read them and
the wrong place to import them from. Asking a running daemon for its status
is a client's job: `omavoi config show`, `hotkey check`, `doctor` and the
console's every refresh all do it, and none of them records audio.

But importing `omavoi.daemon` to reach `ping` loads the whole resident
side — audio, pipeline, history, the hotkey listener — and through audio it
loaded numpy, which is 22 ms on its own. Every command paid that, including
the ones that only print a config value.

So the client half lives here, needing a socket path and json. daemon.py
imports these two names back, so it is still true that the daemon owns the
protocol; it simply no longer has to be loaded to speak it.
"""

from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from . import paths


def ping(sock_path: Path | None = None, timeout: float = 1.0) -> dict[str, Any] | None:
    """Ask a running daemon for its status. None if nothing is listening."""
    sock_path = sock_path or paths.socket_file()
    if not sock_path.exists():
        return None
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(timeout)
            client.connect(str(sock_path))
            client.sendall(json.dumps({"cmd": "status"}).encode() + b"\n")
            data = client.makefile("rb").readline()
        return json.loads(data) if data else None
    except (OSError, json.JSONDecodeError):
        return None


def request(payload: dict[str, Any], timeout: float = 30.0) -> dict[str, Any]:
    sock_path = paths.socket_file()
    if not sock_path.exists():
        raise ConnectionError("the omavoi daemon is not running (start it with `omavoi daemon`)")
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
        client.settimeout(timeout)
        client.connect(str(sock_path))
        client.sendall(json.dumps(payload).encode() + b"\n")
        line = client.makefile("rb").readline()
    if not line:
        raise ConnectionError("the daemon did not answer")
    return json.loads(line)
