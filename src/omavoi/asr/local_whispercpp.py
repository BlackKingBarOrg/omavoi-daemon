"""whisper.cpp backend — the portable GPU path.

The only local engine there is. whisper.cpp runs on Vulkan, which covers
NVIDIA, AMD and Intel with one build and falls back to CPU where there is
no GPU; on Arch that is `pacman -S whisper-cpp ggml-vulkan`, because ggml
loads its compute backends as plugins.

The model has to stay resident or push-to-talk latency is dominated by
loading 3 GB of weights, so we run whisper.cpp's own HTTP server as a
managed child process and talk to it over localhost.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
import subprocess
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotations only, and they are strings here. Importing it for
    # real cost 25-34 ms to every command that merely reaches this
    # module: `setup --json` loads it to find the server binary and
    # `model list --json` loads api_whisper for its PROVIDERS dict.
    import numpy as np


from .. import models, notify, secrets
from ..childlog import ChildLog
from ..models import GGML
from .api_whisper import encode_wav
from .base import NotReady, Segment, Transcript

log = logging.getLogger(__name__)

# Distros disagree on the name; upstream renamed `server` to `whisper-server`.
_SERVER_NAMES = ("whisper-server", "whisper.cpp-server", "whisper-cpp-server", "whisper-cpp-http")


def _explain(output: bytes) -> str:
    """Pull the cause out of a whisper.cpp crash dump.

    On a failed assert the tail of the output is a gdb backtrace, which says
    nothing useful; the one line that matters is several screens earlier.
    """
    text = output.decode("utf-8", "replace")
    lines = [
        line for line in text.splitlines()
        if line.strip() and not line.startswith("#") and "in ??" not in line
        and "Thread debugging" not in line and "debuginfod" not in line.lower()
    ]
    for line in lines:
        if "GGML_ASSERT(device)" in line:
            return (line + "\n  No ggml compute backend could be resolved. The GPU "
                    "plugin alone is not enough:\n"
                    "  sudo pacman -S --needed ggml-cpu")
    assertion = [line for line in lines if "GGML_ASSERT" in line or "error" in line.lower()]
    return "\n".join(assertion[-3:] or lines[-8:])


def _ggml_key(model: str) -> str:
    """Accept `large-v3` or `ggml:large-v3`; whisper.cpp only runs ggml."""
    model = (model or "large-v3").strip()
    return model if model.startswith("ggml:") else f"ggml:{model}"


def find_server() -> str | None:
    for name in _SERVER_NAMES:
        found = shutil.which(name)
        if found:
            return found
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class WhisperCppBackend:
    name = "local-whispercpp"

    def __init__(self, cfg: dict[str, Any]) -> None:
        speech = cfg["speech"]
        opts = speech.get("local_whispercpp", {})
        self.model_key: str = _ggml_key(speech.get("model", ""))
        self.binary: str = opts.get("binary", "")
        self.port: int = int(opts.get("port", 0))
        self.threads: int = int(opts.get("threads", 0))
        self.use_gpu: bool = bool(opts.get("gpu", True))
        # ggml picks its backend plugin from here; leave empty for the default.
        self.backend_path: str = opts.get("ggml_backend_path", "")
        self.startup_timeout: float = float(opts.get("startup_timeout", 120.0))
        # The shortest a take is allowed to wait before the server counts as
        # wedged. Long enough for a cold cache, short enough that finding out
        # is not itself the problem.
        self.min_timeout: float = float(opts.get("min_timeout", 20.0))
        self.beam_size = int(opts.get("beam_size", 5))

        self.default_language: str = speech.get("language", "") or ""
        self.default_prompt: str = ""

        self._proc: subprocess.Popen[bytes] | None = None
        self._log: ChildLog | None = None
        self._client: Any = None
        self._url = ""
        self._model_path = ""

    # -- lifecycle ---------------------------------------------------------

    def load(self) -> None:
        import httpx

        binary = self.binary or find_server()
        if not binary:
            raise NotReady(
                "no whisper.cpp server binary found. On Arch/Omarchy: "
                "sudo pacman -S --needed whisper-cpp ggml-cpu ggml-vulkan"
            )
        path = models.local_path(self.model_key)
        if path is None:
            raise NotReady(
                f"{self.model_key} is not downloaded. Run: omavoi model pull {self.model_key}"
            )
        self._model_path = str(path)

        port = self.port or _free_port()
        argv = [binary, "--model", self._model_path, "--host", "127.0.0.1", "--port", str(port)]
        if self.threads:
            argv += ["--threads", str(self.threads)]
        if not self.use_gpu:
            argv.append("--no-gpu")

        env = dict(os.environ)
        if self.backend_path:
            env["GGML_BACKEND_PATH"] = self.backend_path

        log.info("starting whisper.cpp server: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
        )
        # Drained from here on. Left unread, the pipe fills and the server
        # blocks in write() while still accepting connections — alive, and
        # answering nothing.
        self._log = ChildLog(self._proc)
        self._url = f"http://127.0.0.1:{port}"
        # No pooling. whisper-server answers `Keep-Alive: timeout=5, max=100`,
        # so it drops an idle connection after five seconds — and every hang
        # seen so far followed minutes of silence, never a take ten seconds
        # after the last one. A request posted into a socket the server has
        # already let go is written in full and then waits for a reply that
        # cannot come. Over loopback a fresh connection costs nothing worth
        # measuring, so the reuse that creates the race simply goes away.
        self._client = httpx.Client(
            timeout=self.startup_timeout,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self._wait_ready()

    def _post(self, body: bytes, data: dict[str, str], deadline: float) -> Any:
        if self._client is None:
            raise RuntimeError("whisper.cpp server is not running")
        return self._client.post(
            f"{self._url}/inference", data=data,
            files={"file": ("audio.wav", body, "audio/wav")},
            timeout=deadline,
        )

    def use(self, model_key: str) -> None:
        """Point this backend at different weights.

        Measured at 3.7 s from spawn to a usable transcription for large-v3,
        so this is worth doing when the mode changes rather than when the take
        arrives. Only one server runs: swapping costs a reload, not VRAM.
        """
        key = str(model_key or "").strip()
        if not key or key == self.model_key:
            return
        path = models.local_path(key)
        if path is None:
            raise NotReady(
                f"{key} is not downloaded. Run: omavoi model pull {key}"
            )
        spec = models.spec(key)
        if spec is not None and spec.fmt != GGML:
            raise NotReady(
                f"{key} is a {spec.fmt} model and this engine runs {GGML}; "
                f"a mode cannot change the engine, only the weights"
            )
        log.info("switching speech model: %s -> %s", self.model_key, key)
        self.close()
        self.model_key = key
        self.load()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                out = self._log.text() if self._log is not None else b""
                # Usually a missing ggml compute backend, which no amount of
                # restarting will conjure.
                raise NotReady(
                    "whisper.cpp server exited on startup:\n" + _explain(out)
                )
            try:
                # Any answer at all means the HTTP listener is up.
                self._client.get(f"{self._url}/", timeout=1.0)
                log.info("whisper.cpp server ready at %s", self._url)
                return
            except Exception:
                time.sleep(0.25)
        raise RuntimeError(f"whisper.cpp server was not ready within {self.startup_timeout:.0f}s")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        proc = self._proc
        self._proc = None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2.0)

    def state(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine": "whisper.cpp",
            "model": self.model_key,
            "device": "gpu" if self.use_gpu else "cpu",
            # The child process is the model: if it is up, the weights are
            # resident and a take costs a decode, not a load.
            "live": self._proc is not None and self._proc.poll() is None,
            "url": self._url,
            "pid": self._proc.pid if self._proc is not None else 0,
        }

    def describe(self) -> str:
        st = self.state()
        state = "running" if st["live"] else "not started"
        return f"whisper.cpp {st['model']} [{st['device']}] {st['url']} ({state})"

    # -- inference ---------------------------------------------------------

    def transcribe(
        self,
        samples: np.ndarray,
        rate: int,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        import httpx

        if self._client is None:
            raise RuntimeError("whisper.cpp server is not running")

        lang = language if language is not None else self.default_language
        seeded = prompt if prompt is not None else self.default_prompt

        data: dict[str, str] = {
            "response_format": "verbose_json",
            "temperature": "0.0",
            "beam_size": str(self.beam_size),
        }
        # whisper.cpp defaults its language to English rather than detecting
        # it, unlike every other whisper binding. Left implicit, Chinese comes
        # back as invented English — same weights, same audio, different
        # answer from the CUDA backend. So say "auto" out loud.
        data["language"] = lang or "auto"
        if seeded:
            data["prompt"] = seeded

        body = encode_wav(samples, rate)
        # A flat 120s meant a two-second clip waited two minutes to find out
        # the server had stopped answering — and every press during that
        # window was refused as "still transcribing the previous take". Large
        # models run well above realtime on a GPU, so four times the audio
        # plus a floor is generous and still fails fast.
        seconds = len(samples) / float(rate or 16000)
        deadline = max(self.min_timeout, seconds * 4.0 + 10.0)

        started = time.monotonic()
        try:
            response = self._post(body, data, deadline)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # Wedged, not slow: it accepts connections and never finishes. A
            # 29-hour-old server did exactly this, and because nothing here
            # could tell the difference the key was dead until the daemon was
            # restarted by hand. Rebuild it and try the take once more rather
            # than throwing away what the user just said.
            log.warning("whisper.cpp stopped answering (%s); restarting it", exc)
            notify.send("Omavoi", "the speech server stopped answering — "
                                  "restarting it and retrying this take",
                        urgency="normal")
            try:
                self.close()
                self.load()
            except Exception as restart_exc:
                raise RuntimeError(
                    f"whisper.cpp stopped answering and could not be "
                    f"restarted: {restart_exc}"
                ) from exc
            try:
                response = self._post(body, data, deadline)
            except (httpx.TimeoutException, httpx.TransportError) as second:
                raise RuntimeError(
                    "whisper.cpp stopped answering twice, once after a "
                    f"restart: {second}"
                ) from second
            log.info("whisper.cpp answered after a restart")
        elapsed = time.monotonic() - started

        if response.status_code >= 400:
            raise RuntimeError(f"whisper.cpp returned {response.status_code}: "
                               + secrets.scrub(response.text[:200]))

        payload = response.json()
        segments = [
            Segment(
                start=_seconds(seg, "start", "t0"),
                end=_seconds(seg, "end", "t1"),
                text=str(seg.get("text", "")),
                avg_logprob=float(seg.get("avg_logprob", 0.0) or 0.0),
                no_speech_prob=float(seg.get("no_speech_prob", 0.0) or 0.0),
                compression_ratio=float(seg.get("compression_ratio", 0.0) or 0.0),
                temperature=float(seg.get("temperature", 0.0) or 0.0),
            )
            for seg in payload.get("segments", []) or []
        ]
        text = "\n".join(s.text.strip() for s in segments if s.text.strip())
        if not text:
            text = str(payload.get("text", "")).strip()
        return Transcript(
            text=text,
            segments=segments,
            language=str(payload.get("language", "") or lang),
            audio_seconds=samples.size / rate,
            decode_seconds=elapsed,
            model=self.model_key,
            device="whisper.cpp/" + ("gpu" if self.use_gpu else "cpu"),
        )


def _seconds(seg: dict[str, Any], *keys: str) -> float:
    """whisper.cpp emits either OpenAI-style seconds or its own centisecond t0/t1."""
    for key in keys:
        if key in seg and seg[key] is not None:
            value = float(seg[key])
            return value / 100.0 if key in ("t0", "t1") else value
    return 0.0
