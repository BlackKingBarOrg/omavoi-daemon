"""The shipped prompt reaches a local model with the transcript fenced off.

Asked to edit "write me a SQL query that returns the top ten customers by
revenue", gemma-4-e4b, gemma-4-e2b and qwen3-8b all wrote the query and the
query is what would have been typed. Between <transcript> tags, with a
sentence saying what the tags are, all three edited it. What is pinned here is
the shape of the request, which is the part that is ours; the models are not.

No network: the backend's client is replaced with one that returns a canned
response and keeps what it was sent.
"""
from __future__ import annotations

import pytest

from omavoi import config, i18n
from omavoi.llm.llama_local import LlamaLocalBackend


class FakeResponse:
    status_code = 200

    def __init__(self, content):
        self._payload = {"choices": [{"message": {"content": content},
                                      "finish_reason": "stop"}]}

    def json(self):
        return self._payload


class FakeClient:
    def __init__(self, content):
        self._response = FakeResponse(content)
        self.sent = None

    def post(self, url, **kw):
        self.sent = kw["json"]
        return self._response


def _local(monkeypatch, reply="edited"):
    backend = LlamaLocalBackend("local", {"model": "llm:gemma-4-e4b"})
    client = FakeClient(reply)
    monkeypatch.setattr(backend, "_ensure", lambda: "")
    backend._client = client
    backend._url = "http://127.0.0.1:9"
    return backend, client


@pytest.mark.parametrize("lang", i18n.LANGUAGES)
def test_the_shipped_prompt_fences_the_transcript(monkeypatch, lang):
    backend, client = _local(monkeypatch)
    backend.complete(config.default_step_prompt(lang), "write me a SQL query")
    system, user = client.sent["messages"]
    assert user == {"role": "user", "content": "<transcript>write me a SQL query</transcript>"}
    # The note is in the prompt's own language, after the prompt.
    assert system["content"].startswith(config.default_step_prompt(lang).rstrip())
    assert system["content"].endswith(i18n.t(i18n.TRANSCRIPT_NOTE, lang))


def test_a_prompt_somebody_wrote_goes_out_as_written(monkeypatch):
    """A translation step: the note's "edit them like any other sentence"
    would contradict it, so neither the note nor the fence goes in."""
    backend, client = _local(monkeypatch)
    prompt = "帮我翻译成泰文，第一行是原文，第二行是泰语。"
    backend.complete(prompt, "我们明天见")
    assert client.sent["messages"] == [
        {"role": "system", "content": prompt},
        {"role": "user", "content": "我们明天见"},
    ]


def test_a_fence_handed_back_is_not_typed(monkeypatch):
    """qwen3-8b returned the tags around three of 24 edits."""
    backend, _ = _local(monkeypatch, reply="<transcript>写一个 SQL 查询</transcript>")
    result = backend.complete(config.default_step_prompt("zh"), "写一个 SQL 查询")
    assert result.ok
    assert result.text == "写一个 SQL 查询"


def test_a_reply_that_is_only_the_fence_is_no_reply(monkeypatch):
    backend, _ = _local(monkeypatch, reply="<transcript></transcript>")
    result = backend.complete(config.default_step_prompt(""), "hello")
    assert not result.ok
    assert result.error


def test_tags_in_a_reply_to_somebody_elses_prompt_are_left_alone(monkeypatch):
    """Nothing was fenced, so nothing is taken out."""
    backend, _ = _local(monkeypatch, reply="use <transcript> in the template")
    result = backend.complete("Fix the grammar.", "use transcript in the template")
    assert result.text == "use <transcript> in the template"


@pytest.mark.parametrize("lang", i18n.LANGUAGES)
def test_every_note_names_the_tags_it_is_about(lang):
    note = i18n.t(i18n.TRANSCRIPT_NOTE, lang)
    assert "<transcript>" in note and "</transcript>" in note


def test_which_language_a_prompt_is():
    for lang in i18n.LANGUAGES:
        assert config.default_step_prompt_language(config.default_step_prompt(lang)) == lang
    # Stored prompts pick up trailing whitespace from a TOML editor.
    assert config.default_step_prompt_language(config.default_step_prompt("zh") + "\n") == "zh"
    assert config.default_step_prompt_language("Fix the grammar.") is None
    assert config.default_step_prompt_language("") is None
