"""Any OpenAI-compatible /chat/completions endpoint.

One backend covers llama.cpp's server, ollama, vLLM, LM Studio and OpenAI
itself — they differ only in base_url, model name and whether a key is needed.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from .. import secrets
from .base import LlmResult

log = logging.getLogger(__name__)


class OpenAiCompatBackend:
    def __init__(self, name: str, cfg: dict[str, Any]) -> None:
        self.name = name
        self.backend = str(cfg.get("backend", "openai"))
        self.base_url = str(cfg.get("base_url") or "https://api.openai.com/v1").rstrip("/")
        self.model = str(cfg.get("model", ""))
        self.key_env = str(cfg.get("key_env", ""))
        self.key_name = str(cfg.get("key_name", "") or name)
        self.timeout = float(cfg.get("timeout", 20.0))
        self.max_tokens = int(cfg.get("max_tokens", 1024))
        self.temperature = float(cfg.get("temperature", 0.2))
        self._client: Any = None

    def state(self) -> dict[str, Any]:
        key = secrets.resolve(self.key_env, self.key_name) if self.key_env else ""
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0].split(":", 1)[0]
        local = host in ("127.0.0.1", "localhost", "::1", "0.0.0.0")
        return {
            "name": self.name,
            "backend": self.backend,
            "engine": "openai-compatible",
            "model": self.model,
            "remote": not local,
            "live": bool(key) or not self.key_env,
            "url": self.base_url,
            "pid": 0,
            "problem": "" if (key or not self.key_env) else f"no key: set {self.key_env}",
        }

    def describe(self) -> str:
        return f"{self.name}: {self.model} @ {self.base_url}"

    def _ensure(self) -> Any:
        if self._client is None:
            import httpx

            self._client = httpx.Client(timeout=self.timeout)
        return self._client

    def complete(self, system: str, text: str, *, timeout: float = 0.0) -> LlmResult:
        started = time.monotonic()
        try:
            client = self._ensure()
            key = secrets.resolve(self.key_env, self.key_name)
            headers = {"Authorization": f"Bearer {key}"} if key else {}
            messages = []
            if system.strip():
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": text})

            response = client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "stream": False,
                },
                headers=headers,
                timeout=timeout or self.timeout,
            )
            if response.status_code >= 400:
                # Never echo the body wholesale: some servers reflect the key.
                return LlmResult("", self.model, self.backend, time.monotonic() - started,
                                 error=f"HTTP {response.status_code}: "
                                       + secrets.scrub(response.text[:160], key))
            payload = response.json()
            # `content` is null on more endpoints than it looks: a reasoning
            # model that puts its answer in reasoning_content, a response cut
            # off by max_tokens, a refusal, a tool call. `str(None)` is the
            # four-character string "None", which is non-empty, so LlmResult
            # called it ok and the pipeline typed the word None into whatever
            # window was in front. A non-string content is no content.
            raw = (payload.get("choices") or [{}])[0].get("message", {}).get("content")
            content = raw.strip() if isinstance(raw, str) else ""
            if not content:
                # Said, not swallowed: the caller keeps the text it had, and
                # without this the take looks like the step simply chose to
                # change nothing.
                finish = (payload.get("choices") or [{}])[0].get("finish_reason", "")
                return LlmResult("", self.model, self.backend,
                                 time.monotonic() - started,
                                 error="the endpoint returned no text"
                                       + (f" (finish_reason={finish})" if finish else ""))
            return LlmResult(content, self.model, self.backend,
                             time.monotonic() - started)
        except Exception as exc:
            return LlmResult("", self.model, self.backend, time.monotonic() - started,
                             error=f"{type(exc).__name__}: {exc}")
