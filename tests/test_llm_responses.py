"""What each LLM backend does with a response, which is what gets typed.

A completion step rewrites the transcript before it reaches the window, so a
parse that mistakes "no answer" for an answer types the mistake. Two of the
three backends did: `str(payload["choices"][0]["message"]["content"])` on a
null content is the four-character string "None", which is non-empty, so
LlmResult.ok was True and the pipeline injected the word None.

`content` is null on more endpoints than it looks — a reasoning model with
its answer in reasoning_content, a response cut off by max_tokens, a
refusal, a tool call. Anthropic's backend was already right, because it
filters content blocks by type and joins, and an absent content joins to "".

No network: the httpx client is replaced with one that returns a canned
response.
"""
from __future__ import annotations

import json

import pytest


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        if isinstance(self._payload, str):
            raise ValueError("not json")
        return self._payload

    @property
    def text(self):
        return self._payload if isinstance(self._payload, str) else json.dumps(self._payload)


class FakeClient:
    def __init__(self, response):
        self._response = response
        self.sent = None

    def post(self, url, **kw):
        self.sent = {"url": url, **kw}
        return self._response


def _openai(monkeypatch, payload, status=200):
    from omavoi.llm.openai_compat import OpenAiCompatBackend

    cfg = {"backend": "openai", "base_url": "https://example.invalid/v1",
           "model": "test-model", "key_env": "", "key_name": "nope",
           "max_tokens": 100, "temperature": 0.0, "timeout": 5.0}
    b = OpenAiCompatBackend("api", cfg)
    client = FakeClient(FakeResponse(payload, status))
    monkeypatch.setattr(b, "_ensure", lambda: client)
    return b, client


def _msg(content, finish="stop"):
    return {"choices": [{"message": {"content": content}, "finish_reason": finish}]}


# -- the bug ----------------------------------------------------------------


@pytest.mark.parametrize("content", [None, 123, [], {}, ["a", "b"]])
def test_a_non_string_content_is_no_content(monkeypatch, content):
    """And never the word "None" typed into the user's window."""
    b, _ = _openai(monkeypatch, _msg(content))
    r = b.complete("sys", "hello")
    assert r.text == "", f"{content!r} produced {r.text!r}"
    assert not r.ok, "an unusable response must not be ok"
    assert "None" not in r.text


def test_an_empty_string_content_is_reported_not_swallowed(monkeypatch):
    """The caller keeps the text it had; without an error the take looks like
    the step simply chose to change nothing."""
    b, _ = _openai(monkeypatch, _msg("   ", finish="length"))
    r = b.complete("sys", "hello")
    assert r.text == ""
    assert r.error, "it must say why"
    assert "length" in r.error, "and name the finish_reason"


def test_a_real_answer_comes_through(monkeypatch):
    b, _ = _openai(monkeypatch, _msg("  the rewritten text  "))
    r = b.complete("sys", "hello")
    assert r.text == "the rewritten text"
    assert r.ok
    assert r.error == ""


# -- the request ------------------------------------------------------------


def test_the_system_prompt_is_sent_only_when_there_is_one(monkeypatch):
    b, client = _openai(monkeypatch, _msg("x"))
    b.complete("", "hello")
    roles = [m["role"] for m in client.sent["json"]["messages"]]
    assert roles == ["user"], "an empty system prompt must not be sent"

    b, client = _openai(monkeypatch, _msg("x"))
    b.complete("be brief", "hello")
    roles = [m["role"] for m in client.sent["json"]["messages"]]
    assert roles == ["system", "user"]


def test_streaming_is_off(monkeypatch):
    """The parse reads one whole payload; a stream would arrive as chunks."""
    b, client = _openai(monkeypatch, _msg("x"))
    b.complete("", "hello")
    assert client.sent["json"]["stream"] is False


# -- errors -----------------------------------------------------------------


def test_an_http_error_does_not_echo_the_whole_body(monkeypatch):
    """Some servers reflect the key back in an error."""
    long_body = "x" * 4000
    b, _ = _openai(monkeypatch, long_body, status=401)
    r = b.complete("", "hello")
    assert not r.ok
    assert "401" in r.error
    assert len(r.error) < 300, "the body must be truncated"


def test_a_body_that_is_not_json_is_an_error_not_a_crash(monkeypatch):
    b, _ = _openai(monkeypatch, "<html>gateway timeout</html>")
    r = b.complete("", "hello")
    assert not r.ok
    assert r.error


def test_no_choices_at_all_is_an_error(monkeypatch):
    b, _ = _openai(monkeypatch, {"choices": []})
    r = b.complete("", "hello")
    assert r.text == ""
    assert not r.ok


# -- anthropic, which was already right -------------------------------------


def _anthropic(monkeypatch, payload, status=200):
    from omavoi.llm.anthropic import AnthropicBackend

    cfg = {"backend": "anthropic", "model": "claude-x", "key_env": "",
           "key_name": "nope", "max_tokens": 100, "temperature": 0.0,
           "timeout": 5.0}
    b = AnthropicBackend("claude", cfg)
    client = FakeClient(FakeResponse(payload, status))
    monkeypatch.setattr(b, "_ensure", lambda: client)
    monkeypatch.setattr("omavoi.secrets.resolve", lambda *a, **k: "sk-test")
    return b, client


def test_anthropic_joins_only_the_text_blocks(monkeypatch):
    b, _ = _anthropic(monkeypatch, {"content": [
        {"type": "thinking", "thinking": "hmm"},
        {"type": "text", "text": "the answer"},
        {"type": "text", "text": " continued"},
    ]})
    r = b.complete("sys", "hello")
    assert r.text == "the answer continued"
    assert "hmm" not in r.text, "a thinking block is not the answer"


@pytest.mark.parametrize("payload", [{"content": None}, {"content": []},
                                     {"content": [{"type": "tool_use"}]}, {}])
def test_anthropic_with_no_text_block_is_not_ok(monkeypatch, payload):
    b, _ = _anthropic(monkeypatch, payload)
    assert not b.complete("sys", "hello").ok
