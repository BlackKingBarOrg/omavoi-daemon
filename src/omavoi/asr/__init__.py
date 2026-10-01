"""ASR backend registry."""

from __future__ import annotations

from typing import Any

from .base import Backend, NotReady, Segment, Transcript

__all__ = [
    "BACKENDS",
    "LOCAL",
    "Backend",
    "NotReady",
    "Segment",
    "Transcript",
    "build",
    "engine_for",
    "why_unavailable",
]


def why_unavailable(name: str) -> str:
    """Why this backend cannot run here, or "" if it can.

    Checked before a command writes the backend into the config, not after.
    Writing one the daemon cannot load is how `speech.backend = api` once left
    a daemon failing on every start for most of a day while the old process
    kept serving.
    """
    if name == "local-whispercpp":
        from .local_whispercpp import find_server

        if find_server() is None:
            return ("whisper.cpp is not installed. "
                    "Run: sudo pacman -S --needed whisper-cpp ggml ggml-vulkan")
        return ""
    if name == "local-llama-asr":
        from ..llm.llama_local import find_server as find_llama

        if find_llama() is None:
            return ("llama.cpp is not installed. "
                    "Run: sudo pacman -S --needed llama-cpp")
        return ""
    if name == "api":
        from .. import secrets

        return "" if secrets.resolve("OPENAI_API_KEY", "openai") else (
            "the API backend needs a key: set OPENAI_API_KEY, or put it in "
            "~/.config/omavoi/secrets.toml")
    return ""

# name -> (aliases, one-line description)
BACKENDS: dict[str, tuple[tuple[str, ...], str]] = {
    "local-whispercpp": (
        # `local` and `whisper` used to name the CUDA engine; they point at
        # the one local engine there is now, so a config or a habit that
        # still says either lands somewhere that works.
        ("local", "whispercpp", "whisper.cpp", "cpp", "vulkan", "whisper"),
        "whisper.cpp. Vulkan covers NVIDIA, AMD and Intel, and it runs on CPU too.",
    ),
    "local-llama-asr": (
        ("llama-asr", "qwen3-asr"),
        "llama.cpp running Qwen3-ASR, on the same Vulkan backend as whisper.cpp.",
    ),
    "api": (
        ("openai", "remote"),
        "Any OpenAI-compatible endpoint: OpenAI, Groq, SiliconFlow, a local server.",
    ),
}

_ALIASES = {alias: name for name, (aliases, _) in BACKENDS.items() for alias in (name, *aliases)}


# The engines that run weights on this machine. Which of them a take uses is
# the weights' business: speech.backend says local or remote.
LOCAL: tuple[str, ...] = ("local-whispercpp", "local-llama-asr")


def canonical(name: str) -> str | None:
    return _ALIASES.get(str(name).strip().lower())


def engine_for(cfg: dict[str, Any]) -> str | None:
    """The engine that runs this config's speech model.

    The backend it names, unless that is a local engine and the model belongs
    to the other one. `model use` writes both together, but the console's
    "on this computer" card writes only the backend, and whisper-server handed
    Qwen3-ASR's weights would fail to load them and take dictation down with it.
    """
    name = canonical(cfg["speech"]["backend"])
    if name in LOCAL:
        from .. import models

        spec = models.spec(str(cfg["speech"].get("model", "") or ""))
        if spec is not None and spec.kind == models.SPEECH and canonical(spec.backend) in LOCAL:
            return canonical(spec.backend)
    return name


def build(cfg: dict[str, Any]) -> Backend:
    name = engine_for(cfg)
    if name == "local-whispercpp":
        from .local_whispercpp import WhisperCppBackend

        return WhisperCppBackend(cfg)
    if name == "local-llama-asr":
        from .local_llama_asr import LlamaAsrBackend

        return LlamaAsrBackend(cfg)
    if name == "api":
        from .api_whisper import ApiWhisperBackend

        return ApiWhisperBackend(cfg)
    raise ValueError(
        f"unknown speech.backend: {cfg['speech']['backend']!r} (one of: {', '.join(BACKENDS)})"
    )
