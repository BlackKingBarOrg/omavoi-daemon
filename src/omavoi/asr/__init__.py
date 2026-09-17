"""ASR backend registry."""

from __future__ import annotations

from typing import Any

from .base import Backend, NotReady, Segment, Transcript

__all__ = [
    "BACKENDS",
    "Backend",
    "NotReady",
    "Segment",
    "Transcript",
    "build",
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
                    "Run: sudo pacman -S --needed whisper-cpp ggml-cpu ggml-vulkan")
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
    "api": (
        ("openai", "remote"),
        "Any OpenAI-compatible endpoint: OpenAI, Groq, SiliconFlow, a local server.",
    ),
}

_ALIASES = {alias: name for name, (aliases, _) in BACKENDS.items() for alias in (name, *aliases)}


def canonical(name: str) -> str | None:
    return _ALIASES.get(str(name).strip().lower())


def build(cfg: dict[str, Any]) -> Backend:
    name = canonical(cfg["speech"]["backend"])
    if name == "local-whispercpp":
        from .local_whispercpp import WhisperCppBackend

        return WhisperCppBackend(cfg)
    if name == "api":
        from .api_whisper import ApiWhisperBackend

        return ApiWhisperBackend(cfg)
    raise ValueError(
        f"unknown speech.backend: {cfg['speech']['backend']!r} (one of: {', '.join(BACKENDS)})"
    )
