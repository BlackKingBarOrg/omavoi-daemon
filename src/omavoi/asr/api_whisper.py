"""Remote ASR over the OpenAI-compatible /audio/transcriptions endpoint.

One backend covers OpenAI, Groq, SiliconFlow, DeepInfra, and any local
whisper.cpp / vLLM server that speaks the same shape — they differ only in
base_url and model name.
"""

from __future__ import annotations

import io
import logging
import time
import wave
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotations only, and they are strings here. Importing it for
    # real cost 25-34 ms to every command that merely reaches this
    # module: `setup --json` loads it to find the server binary and
    # `model list --json` loads api_whisper for its PROVIDERS dict.
    import numpy as np


from .. import secrets
from .base import NotReady, Segment, Transcript

log = logging.getLogger(__name__)

# Sensible base_urls so config is one word, not a URL you have to look up.
PROVIDERS: dict[str, dict[str, str]] = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "whisper-1",
        "key_env": "OPENAI_API_KEY",
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "whisper-large-v3",
        "key_env": "GROQ_API_KEY",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "model": "FunAudioLLM/SenseVoiceSmall",
        "key_env": "SILICONFLOW_API_KEY",
    },
    "deepinfra": {
        "base_url": "https://api.deepinfra.com/v1/openai",
        "model": "openai/whisper-large-v3",
        "key_env": "DEEPINFRA_API_KEY",
    },
    "local": {
        # whisper.cpp's server, vLLM, LocalAI — anything OpenAI-shaped.
        "base_url": "http://127.0.0.1:8080/v1",
        "model": "whisper-1",
        "key_env": "",
    },
}


def encode_wav(samples: np.ndarray, rate: int) -> bytes:
    """float32 [-1,1] -> 16-bit PCM WAV, in memory."""
    import numpy as np

    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buf.getvalue()


class ApiWhisperBackend:
    name = "api"

    def __init__(self, cfg: dict[str, Any]) -> None:
        api = cfg["speech"].get("api", {})
        provider = str(api.get("provider", "openai"))
        preset = PROVIDERS.get(provider, {})

        self.provider = provider
        self.base_url = str(api.get("base_url") or preset.get("base_url", "")).rstrip("/")
        self.model_name = str(api.get("model") or preset.get("model", ""))
        self.key_env = str(api.get("key_env") or preset.get("key_env", ""))
        self.key_name = str(api.get("key_name") or provider)
        self.timeout = float(api.get("timeout", 30.0))
        self.response_format = str(api.get("response_format", "verbose_json"))
        self.extra_body: dict[str, Any] = dict(api.get("extra_body", {}))

        self.default_language: str = cfg["speech"].get("language", "") or ""
        self.default_prompt: str = ""
        self._key = ""
        self._client: Any = None

    def load(self) -> None:
        import httpx

        if not self.base_url:
            raise NotReady(f"unknown provider {self.provider!r} and no base_url set")
        self._key = secrets.resolve(self.key_env, self.key_name)
        if self.key_env and not self._key:
            raise NotReady(
                f"no API key: set {self.key_env}, or add "
                f'{self.key_name} = "..." to ~/.config/omavoi/secrets.toml'
            )
        self._client = httpx.Client(timeout=self.timeout)
        log.info("api backend ready: %s %s", self.provider, self.model_name)

    def use(self, model_key: str) -> None:
        # Nothing is resident, so this is only a name in the next request.
        key = str(model_key or "").strip()
        if key and key != self.model_name:
            log.info("switching speech model: %s -> %s", self.model_name, key)
            self.model_name = key

    def state(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine": f"api:{self.provider}",
            "model": self.model_name,
            "device": "remote",
            # Nothing is resident to be warm or cold; having a usable client
            # is the whole of being ready.
            "live": self._client is not None,
            "url": self.base_url,
            "pid": 0,
        }

    def describe(self) -> str:
        st = self.state()
        return (
            f"{st['engine']} {st['model']} @ {st['url']} "
            f"key={secrets.redact(self._key)}"
        )

    def transcribe(
        self,
        samples: np.ndarray,
        rate: int,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        if self._client is None:
            raise RuntimeError("api backend is not initialised")

        lang = language if language is not None else self.default_language
        seeded = prompt if prompt is not None else self.default_prompt

        data: dict[str, str] = {"model": self.model_name, "response_format": self.response_format}
        if lang:
            data["language"] = lang
        if seeded:
            data["prompt"] = seeded
        for key, value in self.extra_body.items():
            data[key] = str(value)

        headers = {"Authorization": f"Bearer {self._key}"} if self._key else {}
        files = {"file": ("audio.wav", encode_wav(samples, rate), "audio/wav")}

        started = time.monotonic()
        response = self._client.post(
            f"{self.base_url}/audio/transcriptions", data=data, files=files, headers=headers
        )
        elapsed = time.monotonic() - started

        if response.status_code >= 400:
            # Never echo the body wholesale — some providers reflect the key.
            raise RuntimeError(
                f"{self.provider} returned {response.status_code}: {response.text[:200]}"
            )

        payload = response.json()
        text = str(payload.get("text", "")).strip()
        segments = [
            Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                text=str(s.get("text", "")),
                avg_logprob=float(s.get("avg_logprob", 0.0) or 0.0),
                no_speech_prob=float(s.get("no_speech_prob", 0.0) or 0.0),
                compression_ratio=float(s.get("compression_ratio", 0.0) or 0.0),
                temperature=float(s.get("temperature", 0.0) or 0.0),
            )
            for s in payload.get("segments", []) or []
        ]
        return Transcript(
            text=text,
            segments=segments,
            language=str(payload.get("language", "") or ""),
            audio_seconds=samples.size / rate,
            decode_seconds=elapsed,
            model=self.model_name,
            device=f"api/{self.provider}",
            extra={"http_status": response.status_code},
        )
