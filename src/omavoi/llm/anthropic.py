"""Anthropic Messages API."""

from __future__ import annotations

import logging
import time
from typing import Any

from .. import secrets
from .base import LlmResult

log = logging.getLogger(__name__)

_API_VERSION = "2023-06-01"


class AnthropicBackend:
    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.backend = "anthropic"
        self.base_url = str(cfg.get("base_url") or "https://api.anthropic.com").rstrip("/")
        self.model = str(cfg.get("model", "claude-haiku-4-5-20251001"))
        self.key_env = str(cfg.get("key_env", "") or "ANTHROPIC_API_KEY")
        self.key_name = str(cfg.get("key_name", "") or "anthropic")
        self.timeout = float(cfg.get("timeout", 20.0))
        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.2))
        self._client: Any = None

    def state(self) -> dict[str, Any]:
        key = secrets.resolve(self.key_env, self.key_name)
        return {
            "name": self.name,
            "backend": self.backend,
            "engine": "anthropic",
            "model": self.model,
            "remote": True,
            # Nothing is resident. Having a key is the whole of being usable,
            # so that is what `live` has to mean here.
            "live": bool(key),
            "url": self.base_url,
            "pid": 0,
            "problem": "" if key else f"no key: set {self.key_env}",
        }

    def describe(self) -> str:
        key = secrets.resolve(self.key_env, self.key_name)
        return f"{self.name}: {self.model} @ anthropic key={secrets.redact(key)}"

    def _ensure(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def complete(self, system: str, text: str, *, timeout: float = 0.0) -> LlmResult:
        started = time.monotonic()
        try:
            key = secrets.resolve(self.key_env, self.key_name)
            if not key:
                return LlmResult(
                    "", self.model, self.backend, 0.0,
                    error=f"no API key: set {self.key_env} or add {self.key_name} to secrets.toml",
                )
            body: dict[str, Any] = {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "messages": [{"role": "user", "content": text}],
            }
            if system.strip():
                body["system"] = system

            response = self._ensure().post(
                f"{self.base_url}/v1/messages",
                json=body,
                headers={
                    "x-api-key": key,
                    "anthropic-version": _API_VERSION,
                    "content-type": "application/json",
                },
                timeout=timeout or self.timeout,
            )
            if response.status_code >= 400:
                return LlmResult("", self.model, self.backend, time.monotonic() - started,
                                 error=f"HTTP {response.status_code}: "
                                       + secrets.scrub(response.text[:160], key))
            payload = response.json()
            parts = [
                block.get("text", "")
                for block in payload.get("content", [])
                if block.get("type") == "text"
            ]
            return LlmResult("".join(parts).strip(), self.model, self.backend,
                             time.monotonic() - started)
        except Exception as exc:
            return LlmResult("", self.model, self.backend, time.monotonic() - started,
                             error=f"{type(exc).__name__}: {exc}")
