"""Local Whisper via faster-whisper / CTranslate2."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Annotations only, and they are strings here. Importing it for
    # real cost 25-34 ms to every command that merely reaches this
    # module: `setup --json` loads it to find the server binary and
    # `model list --json` loads api_whisper for its PROVIDERS dict.
    import numpy as np


from .. import cudaenv, models
from .base import NotReady, Segment, Transcript

log = logging.getLogger(__name__)


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        cudaenv.preload()
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda"
    except Exception as exc:  # pragma: no cover - host dependent
        log.warning("CUDA probe failed, falling back to CPU: %s", exc)
    return "cpu"


def _resolve_compute_type(requested: str, device: str) -> str:
    if requested != "auto":
        return requested
    return "float16" if device == "cuda" else "int8"


class LocalWhisperBackend:
    name = "local-whisper"

    def __init__(self, cfg: dict[str, Any]) -> None:
        speech = cfg["speech"]
        local = speech.get("local_whisper", {})
        self.model_id: str = speech["model"]
        self.device = _resolve_device(local.get("device", "auto"))
        self.compute_type = _resolve_compute_type(local.get("compute_type", "auto"), self.device)
        self.beam_size = int(local.get("beam_size", 5))
        self.cpu_threads = int(local.get("cpu_threads", 0))
        # Whisper's own VAD is off by default: it silently discards quiet
        # speech, which shows up as dropped words. Endpointing is our job.
        self.vad_filter = bool(local.get("vad_filter", False))
        self.temperature_fallback = bool(local.get("temperature_fallback", True))
        self.default_language: str = speech.get("language", "") or ""
        self.default_prompt: str = ""
        self._model: Any = None

    def load(self) -> None:
        cudaenv.preload()
        try:
            from faster_whisper import WhisperModel
        except ModuleNotFoundError as exc:
            raise NotReady(
                "the CUDA engine needs faster-whisper, which is not installed. "
                "Run: uv tool install --reinstall omavoi[cuda] — or pick "
                "Local / Vulkan, which needs no extra"
            ) from exc

        source = models.resolve_for_load(self.model_id)
        kwargs: dict[str, Any] = {
            "device": self.device,
            "compute_type": self.compute_type,
            "download_root": str(models.model_root()),
        }
        if self.cpu_threads:
            kwargs["cpu_threads"] = self.cpu_threads

        started = time.monotonic()
        try:
            self._model = WhisperModel(source, **kwargs)
        except Exception as exc:
            if self.device != "cuda":
                raise
            # A cuDNN/driver mismatch lands here. 24 CPU cores is slow but
            # usable, so degrade rather than leave the user with no dictation.
            log.error("CUDA load failed (%s), retrying on CPU", exc)
            self.device = "cpu"
            self.compute_type = _resolve_compute_type("auto", "cpu")
            kwargs.update(device=self.device, compute_type=self.compute_type)
            self._model = WhisperModel(source, **kwargs)

        log.info(
            "loaded %s on %s/%s in %.1fs",
            self.model_id, self.device, self.compute_type, time.monotonic() - started,
        )

    def use(self, model_key: str) -> None:
        key = str(model_key or "").strip()
        if not key or key == self.model_id:
            return
        log.info("switching speech model: %s -> %s", self.model_id, key)
        self._model = None
        self.model_id = key
        self.load()

    def state(self) -> dict[str, Any]:
        return {
            "backend": self.name,
            "engine": "faster-whisper",
            "model": self.model_id,
            "device": f"{self.device}/{self.compute_type}",
            "live": self._model is not None,
            "url": "",
            "pid": 0,
        }

    def describe(self) -> str:
        st = self.state()
        state = "loaded" if st["live"] else "not loaded"
        return f"{st['model']} [{st['device']}] ({state})"

    def transcribe(
        self,
        samples: np.ndarray,
        rate: int,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript:
        if self._model is None:
            raise RuntimeError("model is not loaded")
        if rate != 16000:
            raise ValueError(f"whisper needs 16 kHz audio, got {rate}")

        lang = language if language is not None else self.default_language
        seeded = prompt if prompt is not None else self.default_prompt

        started = time.monotonic()
        raw_segments, info = self._model.transcribe(
            samples,
            language=lang or None,
            beam_size=self.beam_size,
            initial_prompt=seeded or None,
            vad_filter=self.vad_filter,
            # Each utterance is independent; carrying context across
            # push-to-talk takes is how repeat-loops get started.
            condition_on_previous_text=False,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if self.temperature_fallback else 0.0,
            word_timestamps=False,
        )

        segments = [
            Segment(
                start=float(s.start),
                end=float(s.end),
                text=s.text,
                avg_logprob=float(s.avg_logprob),
                no_speech_prob=float(s.no_speech_prob),
                compression_ratio=float(s.compression_ratio),
                temperature=float(getattr(s, "temperature", 0.0) or 0.0),
            )
            for s in raw_segments
        ]
        return Transcript(
            text="\n".join(s.text.strip() for s in segments if s.text.strip()),
            segments=segments,
            language=getattr(info, "language", "") or "",
            language_probability=float(getattr(info, "language_probability", 0.0) or 0.0),
            audio_seconds=samples.size / rate,
            decode_seconds=time.monotonic() - started,
            model=self.model_id,
            device=f"{self.device}/{self.compute_type}",
        )
