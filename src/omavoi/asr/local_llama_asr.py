"""Qwen3-ASR, run by a llama.cpp server omavoi starts and owns.

The second local speech engine, and the first that is not whisper. llama.cpp's
multimodal library knows Qwen3-ASR, and its server answers an OpenAI-shaped
/v1/audio/transcriptions, on the same ggml backends — Vulkan included — that
whisper.cpp already brought in. So this needs `llama-cpp` and nothing else,
which the local LLM needs anyway.

Measured on 2026-10-01 on an Intel Arc B390 (Vulkan), over twenty recorded
takes with a median of 9.3 s: Qwen3-ASR-1.7B averaged 0.80 s a take against
3.38 s for ggml:large-v3-turbo, and their transcripts differed in 2.9% of the
characters. Its published Mandarin error rates are well ahead of whisper's —
AISHELL-1 CER 1.48 against large-v3's 5.14 — and the Open ASR Leaderboard has
its English at 4.31 WER against turbo's 6.36. Those are the vendor's and the
leaderboard's figures; ours had no reference transcripts to score against.

Two things it does differently, both dealt with here rather than downstream:

- The reply is text and nothing else. There are no segments, so the silence
  filter gets no no_speech_prob or avg_logprob and deletes nothing, which is
  what it should do without evidence.
- Every reply begins "language <Name><asr_text>". That is the detected
  language. On silence or noise what follows is either nothing or the model's
  own instruction, "Transcribe audio to text." — three of five silent or
  noise-only clips came back with it — and both come back as an empty take.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import time
import unicodedata
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np

from .. import models, notify, secrets
from ..childlog import ChildLog
from .api_whisper import encode_wav
from .base import NotReady, Transcript

log = logging.getLogger(__name__)

NAME = "local-llama-asr"
_PREFIX = re.compile(r"\s*language\s+(\S+?)\s*<asr_text>", re.IGNORECASE)
# The instruction llama-server gives the model, which is what it writes when
# there is nothing to write. Compared folded: it has come back with a full
# stop and might not always.
INSTRUCTION = "Transcribe audio to text."
_ECHO = "transcribeaudiototext"
_CONTEXT = " Context: "


def _fold(text: str) -> str:
    return re.sub(r"[\s\W_]+", "", unicodedata.normalize("NFKC", text).casefold())


def instruction(context: str) -> str:
    """What the model is told, with the recognition hint after it.

    llama-server puts a request's `prompt` where its own instruction goes, so
    a hint sent on its own would replace "transcribe this" with a list of
    names. Kept, and followed by the hint, it is context and nothing more.

    Not the system turn, where Qwen3-ASR's own toolkit puts context. Measured
    on 2026-10-02 over twenty recorded takes: in the system turn the same
    dictionary words changed four transcripts besides the names -- "UI"
    came back as "U I", "Turbo V3" as "Turbo V 三". After the instruction,
    seventeen came back identical and three differed by a comma; "Cloud
    Code" became "Claude Code" and "Cloud海库" "Cloud Haiku" once those words
    were in the dictionary, and nothing else moved; six silent or noise-only
    clips all came back empty, one of which had echoed the bare instruction;
    and the median take took 0.09 s longer. A mode's hint written as a
    sentence, ahead of the names, brought one "U I" back in twenty.
    """
    context = flatten(context)
    return INSTRUCTION + _CONTEXT + context if context else INSTRUCTION


def flatten(context: str) -> str:
    """The hint on one line, its lines kept apart.

    The pipeline puts the mode's hint and the dictionary's names on lines of
    their own. Run together with a space, the last name of one ran into the
    first of the next, and the names after the join stopped helping: "Claude
    Code" came back as "Cloud Code" with them after a newline, and right with
    them after "; ".
    """
    lines = (" ".join(line.split()) for line in str(context or "").splitlines())
    return "; ".join(line for line in lines if line)


def _echoes(body: str, context: str) -> bool:
    """Whether the reply is the hint read back rather than anything said.

    A clip with nothing in it has come back as the bare instruction, so it
    may come back as the hint too. Only the hint whole, or a list of three
    or more of its words and nothing else: one word of it on its own is
    what someone saying "Hyprland" dictated.
    """
    folded = _fold(body)
    if not folded or not context:
        return False
    if folded in (_fold(context), _fold(instruction(context))) or body.lower().startswith("context:"):
        return True
    words = {_fold(w) for w in re.split(r"[,，、;\n]+", context) if _fold(w)}
    said = [_fold(w) for w in re.split(r"[,，、;\n]+", body) if _fold(w)]
    return len(said) >= 3 and all(w in words for w in said)


def parse(text: str) -> tuple[str, str]:
    """The transcript in a reply, and the language it names ("" if none)."""
    body = str(text or "")
    language = ""
    match = _PREFIX.match(body)
    if match:
        language = match.group(1).lower()
        body = body[match.end():]
    if language == "none":
        language = ""
    body = body.strip()
    if _fold(body) == _ECHO:
        body = ""
    return body, language


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class LlamaAsrBackend:
    name = NAME

    def __init__(self, cfg: dict[str, Any]) -> None:
        speech = cfg["speech"]
        opts = speech.get("local_llama_asr", {}) or {}
        self.model_key = str(speech.get("model", "") or "")
        self.binary = str(opts.get("binary", "") or "")
        self.port = int(opts.get("port", 0) or 0)
        self.gpu_layers = int(opts.get("n_gpu_layers", 99))
        # Room for the longest take audio.max_seconds allows: the audio is a
        # few tokens a second of context before a word is written.
        self.ctx_size = int(opts.get("ctx_size", 8192))
        self.startup_timeout = float(opts.get("startup_timeout", 120.0))
        self.min_timeout = float(opts.get("min_timeout", 20.0))
        self.default_language: str = speech.get("language", "") or ""
        self.default_prompt: str = ""
        self._proc: subprocess.Popen[bytes] | None = None
        self._log: ChildLog | None = None
        self._client: Any = None
        self._url = ""

    # -- lifecycle ---------------------------------------------------------

    def _weights(self, key: str) -> tuple[str, list[str]]:
        """The model file and its companions for `key`, or why there are none."""
        spec = models.spec(key)
        if spec is None or spec.kind != models.SPEECH:
            raise NotReady(f"{key or 'no model'} is not a speech model in the catalogue")
        if spec.backend != NAME:
            raise NotReady(f"{key} runs on {spec.backend}, not on llama.cpp; "
                           f"a mode cannot change the engine, only the weights")
        path = models.local_path(key)
        if path is None:
            raise NotReady(f"{key} is not downloaded. Run: omavoi model pull {key}")
        return str(path), [str(path.parent / name) for name in spec.extra_files]

    def load(self) -> None:
        import httpx

        from ..llm.llama_local import find_server

        binary = self.binary or find_server()
        if not binary:
            raise NotReady("llama-server is not installed. "
                           "On Arch/Omarchy: sudo pacman -S --needed llama-cpp")
        model, companions = self._weights(self.model_key)
        port = self.port or _free_port()
        argv = [binary, "--model", model, "--host", "127.0.0.1", "--port", str(port),
                "--ctx-size", str(self.ctx_size), "--parallel", "1",
                "--n-gpu-layers", str(self.gpu_layers), "--alias", self.model_key]
        for path in companions:
            argv += ["--mmproj", path]
        log.info("starting llama-server for speech: %s", " ".join(argv))
        self._proc = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=dict(os.environ)
        )
        # Drained from here on, for the reason whisper's is: a full pipe
        # blocks the server in write() while it still accepts connections.
        self._log = ChildLog(self._proc)
        self._url = f"http://127.0.0.1:{port}"
        self._client = httpx.Client(
            timeout=self.startup_timeout,
            limits=httpx.Limits(max_keepalive_connections=0),
        )
        self._wait_ready()

    def _wait_ready(self) -> None:
        from ..llm.llama_local import _explain

        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                out = self._log.text() if self._log is not None else b""
                raise NotReady("llama-server exited on startup:\n" + _explain(out))
            try:
                if self._client.get(f"{self._url}/health", timeout=1.0).status_code == 200:
                    log.info("speech llama-server ready at %s", self._url)
                    return
            except Exception:
                pass
            time.sleep(0.25)
        raise RuntimeError(f"llama-server was not ready within {self.startup_timeout:.0f}s")

    def use(self, model_key: str) -> None:
        """Point this backend at different weights, if they are Qwen3-ASR's."""
        key = str(model_key or "").strip()
        if not key or key == self.model_key:
            return
        # Checked before the running server is stopped, as in whisper's.
        self._weights(key)
        log.info("switching speech model: %s -> %s", self.model_key, key)
        self.close()
        self.model_key = key
        self.load()

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
        proc, self._proc = self._proc, None
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
            "engine": "llama.cpp",
            "model": self.model_key,
            "device": "gpu" if self.gpu_layers > 0 else "cpu",
            "live": self._proc is not None and self._proc.poll() is None,
            "url": self._url,
            "pid": self._proc.pid if self._proc is not None else 0,
        }

    def describe(self) -> str:
        st = self.state()
        state = "running" if st["live"] else "not started"
        return f"llama.cpp {st['model']} [{st['device']}] {st['url']} ({state})"

    # -- inference ---------------------------------------------------------

    def _post(self, body: bytes, data: dict[str, str], deadline: float) -> Any:
        if self._client is None:
            raise RuntimeError("the speech llama-server is not running")
        return self._client.post(
            f"{self._url}/v1/audio/transcriptions", data=data,
            files={"file": ("audio.wav", body, "audio/wav")}, timeout=deadline,
        )

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
            raise RuntimeError("the speech llama-server is not running")

        lang = language if language is not None else self.default_language
        data: dict[str, str] = {"model": self.model_key, "response_format": "json"}
        if lang and lang != "auto":
            data["language"] = lang
        # The mode's recognition hint and the dictionary's names, after the
        # instruction rather than in place of it. See instruction().
        context = flatten(prompt or "")
        if context:
            data["prompt"] = instruction(context)

        body = encode_wav(samples, rate)
        seconds = len(samples) / float(rate or 16000)
        deadline = max(self.min_timeout, seconds * 4.0 + 10.0)
        started = time.monotonic()
        try:
            response = self._post(body, data, deadline)
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            # Wedged rather than slow, as whisper-server once was: rebuild
            # it and try this take once more instead of losing it.
            log.warning("the speech llama-server stopped answering (%s); restarting it", exc)
            notify.send("Omavoi", "the speech server stopped answering — "
                                  "restarting it and retrying this take",
                        urgency="normal")
            try:
                self.close()
                self.load()
            except Exception as restart_exc:
                raise RuntimeError(f"the speech llama-server stopped answering and "
                                   f"could not be restarted: {restart_exc}") from exc
            response = self._post(body, data, deadline)
        elapsed = time.monotonic() - started

        if response.status_code >= 400:
            raise RuntimeError(f"llama-server returned {response.status_code}: "
                               + secrets.scrub(response.text[:200]))

        text, detected = parse(response.json().get("text", ""))
        if _echoes(text, context):
            log.info("the speech model read back its recognition hint; nothing was said")
            text = ""
        return Transcript(
            text=text,
            segments=[],
            language=detected or (lang if lang not in ("", "auto") else ""),
            audio_seconds=seconds,
            decode_seconds=elapsed,
            model=self.model_key,
            device="llama.cpp/" + ("gpu" if self.gpu_layers > 0 else "cpu"),
        )
