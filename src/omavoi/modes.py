"""Modes: a named chain of models, chosen by the window you are typing into.

A mode is resolved once per take. It decides what the speech model is told,
which rules run, which LLM passes follow, and how the text is injected —
so everything downstream reads from the resolved Mode rather than reaching
back into the config.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from .window import Window

log = logging.getLogger(__name__)


@dataclass(slots=True)
class Step:
    """One LLM pass. `llm` names an [llm.<name>] entry."""

    llm: str
    # Which weights, when the entry is the local one. Empty uses whatever the
    # entry is configured with. This is how two modes share the local slot and
    # still run different models: three configurations, not three per model.
    model: str = ""
    prompt: str = ""
    # Fall through to the previous stage's text rather than failing the take.
    timeout: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {"llm": self.llm, "model": self.model,
                "prompt": self.prompt, "timeout": self.timeout}


@dataclass(slots=True)
class Mode:
    name: str
    language: str = ""
    # Empty means the global speech.model. A mode that names its own gets the
    # weights swapped when it becomes the mode in use.
    speech_model: str = ""
    prompt: str = ""
    inject: str = "auto"
    paste_key: str = ""
    rules: dict[str, Any] = field(default_factory=dict)
    steps: list[Step] = field(default_factory=list)
    matched_on: str = ""

    @property
    def joiner(self) -> str:
        """What replaces a newline before injection; "keep" leaves them."""
        return str(self.rules.get("joiner", " "))

    def chain(self) -> str:
        """`large-v3 -> local -> haiku`, for status lines and logs."""
        return " -> ".join(["speech", *(s.llm for s in self.steps)])

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "matched_on": self.matched_on,
            "language": self.language,
            "speech_model": self.speech_model,
            "inject": self.inject,
            "rules": self.rules,
            "steps": [s.as_dict() for s in self.steps],
        }


def _build(name: str, raw: dict[str, Any], base: dict[str, Any], matched_on: str) -> Mode:
    """Materialise a mode, inheriting anything it does not state from default."""
    def pick(key: str, fallback: Any) -> Any:
        value = raw.get(key)
        if value in (None, ""):
            value = base.get(key, fallback)
        return fallback if value in (None, "") else value

    steps = [
        Step(
            llm=str(step.get("llm", "")),
            model=str(step.get("model", "") or ""),
            prompt=str(step.get("prompt", "")),
            timeout=float(step.get("timeout", 0.0) or 0.0),
        )
        for step in (raw.get("steps") or [])
        if step.get("llm")
    ]
    return Mode(
        name=name,
        language=str(raw.get("language", base.get("language", "")) or ""),
        # Not inherited from default: a mode either names its own weights or
        # takes whatever is loaded. Inheriting would make every mode a switch.
        speech_model=str(raw.get("speech_model", "") or ""),
        prompt=str(raw.get("prompt", base.get("prompt", "")) or ""),
        inject=str(pick("inject", "auto")),
        paste_key=str(raw.get("paste_key", base.get("paste_key", "")) or ""),
        rules=dict(base.get("rules") or {}) | dict(raw.get("rules") or {}),
        steps=steps,
        matched_on=matched_on,
    )


def resolve(cfg: dict[str, Any], window: Window | None = None,
            forced: str = "") -> Mode:
    """Pick the mode for this take.

    A forced name always wins. After that it depends on switching.by_window:
    off (the default) every take uses switching.mode, on the longest `match`
    substring found in the window class or title wins, so
    "org.wezfurlong.wezterm" beats a bare "wez".
    """
    modes: dict[str, Any] = cfg.get("modes", {})
    base: dict[str, Any] = modes.get("default", {})
    switching: dict[str, Any] = cfg.get("switching", {})

    if forced:
        raw = modes.get(forced)
        if raw is None:
            log.warning("forced mode %r does not exist, using default", forced)
            return _build("default", base, {}, "forced-missing")
        return _build(forced, raw, base, "forced")

    if not switching.get("by_window", False):
        name = str(switching.get("mode", "default"))
        raw = modes.get(name)
        if raw is None:
            log.warning("switching.mode=%r does not exist, using default", name)
            return _build("default", base, {}, "fixed")
        return _build(name, raw, base, "fixed")

    haystack = ""
    if window is not None:
        haystack = f"{window.cls} {window.title}".lower()

    best_name, best_token = "", ""
    if haystack:
        for name, raw in modes.items():
            if name == "default" or not isinstance(raw, dict):
                continue
            for token in raw.get("match") or []:
                token = str(token).lower()
                if token and token in haystack and len(token) > len(best_token):
                    best_name, best_token = name, token

    if best_name:
        return _build(best_name, modes[best_name], base, best_token)
    return _build("default", base, {}, "")


def names(cfg: dict[str, Any]) -> list[str]:
    return sorted(cfg.get("modes", {}))
