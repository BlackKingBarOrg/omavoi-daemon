"""LLM backends, referenced from a mode's chain by name."""

from __future__ import annotations

import logging
from typing import Any

from .base import LlmBackend, LlmResult

log = logging.getLogger(__name__)

__all__ = ["LlmBackend", "LlmResult", "Registry", "BACKENDS", "ON_MACHINE"]

BACKENDS: dict[str, str] = {
    "llama-cpp": "A local llama.cpp / ollama / vLLM server speaking the OpenAI API.",
    "openai": "OpenAI, or anything else OpenAI-compatible.",
    "anthropic": "The Claude API.",
    "llama-local": "A llama.cpp server omavoi starts and owns, with its models "
                   "in the same catalogue as the speech ones.",
    "agent-cli": "Whichever coding agent Omarchy is set to — `omarchy default "
                 "agent` picks from pi, opencode, claude, codex, grok, gemini, "
                 "copilot and others. Already logged in, so no key; costs "
                 "several seconds of process startup per take.",
}


# Backends whose weights run here, so using them sends nothing off the
# machine. Kept beside BACKENDS because the UI badges every LLM with this and
# a new backend added below but not listed here would badge itself "remote".
# claude-cli is deliberately absent: the process is local, the inference is not.
ON_MACHINE: frozenset[str] = frozenset({
    "llama-cpp", "llama.cpp", "llamacpp", "llama-local", "ollama", "vllm",
})


def _build_one(name: str, cfg: dict[str, Any]) -> LlmBackend:
    backend = str(cfg.get("backend", "openai")).strip().lower()
    if backend in ("llama-local", "llama.cpp", "llamacpp"):
        from .llama_local import LlamaLocalBackend

        return LlamaLocalBackend(name, cfg)
    if backend in ("agent-cli", "agent", "claude-cli", "claude-code", "claude"):
        from .agent_cli import AgentCliBackend

        # The claude-* names are kept so configs written before this stay
        # valid; they pin the agent rather than following Omarchy's default.
        if backend in ("claude-cli", "claude-code", "claude") and not cfg.get("agent"):
            cfg = dict(cfg) | {"agent": "claude"}
        return AgentCliBackend(name, cfg)
    if backend == "anthropic":
        from .anthropic import AnthropicBackend

        return AnthropicBackend(name, cfg)
    if backend in ("llama-cpp", "llama.cpp", "openai", "openai-compatible", "ollama", "vllm"):
        from .openai_compat import OpenAiCompatBackend

        return OpenAiCompatBackend(name, cfg)
    raise ValueError(f"unknown llm.{name}.backend: {backend!r} (one of: {', '.join(BACKENDS)})")


class Registry:
    """Lazily builds the LLM clients a mode's chain actually names."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        self._defs: dict[str, Any] = dict(cfg.get("llm", {}))
        self._built: dict[str, LlmBackend] = {}
        self._why: dict[str, str] = {}

    @staticmethod
    def _entry_of(key: str) -> str:
        """The config entry a built key belongs to.

        Keys are `name` or `name@model`. Everything that walks _built has to
        split them, or a reload drops every instance and a sweep unloads the
        one still in use.
        """
        return key.split("@", 1)[0]

    def __contains__(self, name: str) -> bool:
        return name in self._defs

    def names(self) -> list[str]:
        return sorted(self._defs)

    def close(self) -> None:
        """Stop anything this registry started. Called when the daemon exits."""
        for backend in self._built.values():
            stop = getattr(backend, "close", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    log.debug("could not stop %s", getattr(backend, "name", backend))

    def update(self, cfg: dict[str, Any]) -> None:
        """Adopt a new config without restarting what did not change.

        A managed llama-server takes seconds to come up and holds gigabytes of
        VRAM. Rebuilding the registry on every reload — and the console reloads
        after every click — orphaned the running server: its process kept the
        memory, the new registry reported it cold, and the next take started a
        second one. Only entries whose own definition changed are dropped.
        """
        defs = dict(cfg.get("llm", {}))
        for key, built in list(self._built.items()):
            name = self._entry_of(key)
            if name in defs and defs[name] == self._defs.get(name):
                continue
            stop = getattr(built, "close", None)
            if callable(stop):
                try:
                    stop()
                except Exception:
                    log.debug("could not stop %s", name)
            del self._built[key]
            log.info("llm %r changed or went away — dropped its running instance", key)
        self._defs = defs
        self._why.clear()

    def retain(self, needed: set[str]) -> list[str]:
        """Release local models nothing is asking for any more.

        Only local backends hold anything worth reclaiming: a remote one is a
        key and an HTTP client. An empty `needed` is a real state — a
        speech-only mode asks for no LLM at all — and not a reason to skip.

        Returns the names it unloaded, for the caller's log line.
        """
        freed: list[str] = []
        for key, backend in self._built.items():
            if self._entry_of(key) in needed:
                continue
            try:
                st = backend.state()
            except Exception:
                continue
            if st.get("remote") or not st.get("live"):
                continue
            stop = getattr(backend, "close", None)
            if not callable(stop):
                continue
            try:
                stop()
            except Exception:
                log.debug("could not stop %s", key)
                continue
            # The object stays: close() resets it to cold, and keeping it
            # means the next take that names it restarts rather than rebuilds.
            freed.append(key)
        return freed

    def states(self) -> list[dict[str, Any]]:
        """What every configured LLM is doing right now.

        Constructing a backend is side-effect-free — no server starts, no
        request goes out — so an entry no take has reached yet can still say
        what it would be. The throwaway is not cached, which keeps `live`
        meaning "running now" rather than "has been asked about".
        """
        out: list[dict[str, Any]] = []
        # Every live instance first: a step running its own weights is a
        # separate server and would otherwise be invisible.
        seen: set[str] = set()
        for key, backend in self._built.items():
            st = dict(backend.state())
            st["name"] = key
            st["problem"] = st.get("problem") or self._why.get(key, "")
            out.append(st)
            seen.add(self._entry_of(key))
        for name in self.names():
            if name in seen:
                continue
            backend = self._built.get(name)
            if backend is None:
                try:
                    backend = _build_one(name, self._defs[name])
                except Exception as exc:
                    defn = self._defs[name]
                    out.append({
                        "name": name,
                        "backend": str(defn.get("backend", "")),
                        "engine": "",
                        "model": str(defn.get("model", "")),
                        "remote": True,
                        "live": False,
                        "url": "",
                        "pid": 0,
                        "problem": str(exc),
                    })
                    continue
            st = dict(backend.state())
            st["problem"] = st.get("problem") or self._why.get(name, "")
            out.append(st)
        return out

    def why(self, name: str) -> str:
        """Why `get` returned None, for the warning the user actually sees."""
        return self._why.get(name, "not configured")

    def get(self, name: str, model: str = "") -> LlmBackend | None:
        """The backend for this entry, running `model` if one is asked for.

        Keyed by both, because a step may name the local configuration and its
        own weights: two modes then share one configuration and get one server
        each. Without the model in the key they would fight over a single
        backend and swap weights on alternate takes.
        """
        key = f"{name}@{model}" if model else name
        if key in self._built:
            return self._built[key]
        cfg = self._defs.get(name)
        if cfg is None:
            log.warning("no [llm.%s] defined", name)
            self._why[key] = f"no [llm.{name}] in the config"
            return None
        if model:
            cfg = dict(cfg) | {"model": model}
        try:
            backend = _build_one(name, cfg)
        except Exception as exc:
            # A backend name the running daemon does not know reads exactly
            # like a missing entry unless it says so — and after an upgrade
            # that adds one, the daemon needs restarting, not reloading.
            log.error("could not build llm %r: %s", name, exc)
            self._why[key] = str(exc)
            return None
        self._built[key] = backend
        return backend
