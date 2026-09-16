"""What is on disk, and who can read it.

secrets.toml is created 0600 before anything is written into it, and
config.toml is deliberately safe to share. Between those two stood the state
directory at 0755, holding:

  history.jsonl   every take's text, its raw transcript and the path to its
                  audio — 0644
  omavoi.log      lines like `typed in 0.59s [default]: <what you said>` —
                  0644
  audio/*.wav     the recordings themselves

On a laptop with one account that is nothing. On a shared machine it is the
transcript of everything the user has ever dictated, readable by everyone
with a login. The stance was already taken one directory over; this is the
directory it was not applied to.

The tests also cover the other direction: an error body about to be written
into that file can contain a credential, because some providers reflect the
key back in the error that says it is wrong.
"""
from __future__ import annotations

import stat

import pytest

from omavoi import config, history, paths, secrets


def _mode(path):
    return stat.S_IMODE(path.stat().st_mode)


def test_a_recorded_take_leaves_nothing_group_or_world_readable(home):
    cfg = config.load()
    h = history.History(cfg)
    h.record({"text": "what I said", "raw_text": "what i said"})

    hist = paths.history_file()
    assert hist.exists()
    assert _mode(hist) & 0o077 == 0, f"history is {oct(_mode(hist))}"
    assert _mode(hist.parent) & 0o077 == 0, f"state dir is {oct(_mode(hist.parent))}"


def test_an_existing_loose_history_is_tightened(home):
    """chmod-on-create fixes nothing for an install running for a week."""
    hist = paths.history_file()
    hist.parent.mkdir(parents=True, exist_ok=True)
    hist.write_text('{"text": "old take"}\n', encoding="utf-8")
    hist.chmod(0o644)
    hist.parent.chmod(0o755)
    assert _mode(hist) == 0o644

    history.History(config.load()).record({"text": "new take"})
    assert _mode(hist) & 0o077 == 0, "an existing file must be tightened too"
    assert _mode(hist.parent) & 0o077 == 0
    assert "old take" in hist.read_text(), "and its contents kept"


def test_the_trim_does_not_undo_it(home):
    """The replacement file carries its own mode."""
    cfg = config.load()
    cfg["history"]["keep"] = 3
    h = history.History(cfg)
    for i in range(8):
        h.record({"text": f"take {i}"})
    hist = paths.history_file()
    assert len(hist.read_text().splitlines()) == 3, "the trim must have run"
    assert _mode(hist) & 0o077 == 0, f"after a trim it is {oct(_mode(hist))}"


def test_the_log_is_private(home):
    from omavoi.commands._support import setup_logging

    setup_logging("INFO", to_file=True)
    log = paths.log_file()
    assert log.exists()
    assert _mode(log) & 0o077 == 0, f"the log is {oct(_mode(log))}"


def test_private_dir_tightens_and_creates(home, tmp_path):
    loose = tmp_path / "loose"
    loose.mkdir(mode=0o755)
    paths.private_dir(loose)
    assert _mode(loose) & 0o077 == 0

    fresh = tmp_path / "a" / "b" / "c"
    paths.private_dir(fresh)
    assert fresh.is_dir()
    assert _mode(fresh) & 0o077 == 0


def test_private_file_is_silent_about_a_file_that_is_not_there(home, tmp_path):
    paths.private_file(tmp_path / "nope")  # must not raise


def test_open_private_creates_at_0600(home, tmp_path):
    target = tmp_path / "new.txt"
    with paths.open_private(target, "w") as fh:
        fh.write("x")
    assert _mode(target) & 0o077 == 0
    # Created private rather than created and chmodded, which leaves a window
    # where it is readable.
    assert target.read_text() == "x"


# -- credentials in an error body ------------------------------------------


@pytest.mark.parametrize("body,secret", [
    ("Incorrect API key provided: sk-proj-abc123DEF456ghi789xyz. Find yours at…",
     "sk-proj-abc123DEF456ghi789xyz"),
    ("auth failed for gsk_0123456789abcdefghijkl", "gsk_0123456789abcdefghijkl"),
    ("bad token xai-aaaaaaaaaaaaaaaaaaaa", "xai-aaaaaaaaaaaaaaaaaaaa"),
    ("key AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz01 rejected",
     "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz01"),
    ("Authorization: Bearer abcdefghijklmnopqrstuvwxyz",
     "Bearer abcdefghijklmnopqrstuvwxyz"),
])
def test_a_reflected_key_does_not_reach_the_history(body, secret):
    """It goes into the take's warnings, history.jsonl and a notification.

    Truncating the body to 160 characters was the guard, and it does not
    work: "Incorrect API key provided: sk-…" puts the key at the front.
    """
    out = secrets.scrub(body)
    assert secret not in out, out


def test_the_key_this_process_holds_is_scrubbed_whatever_it_looks_like():
    """The only reliable half: a provider with a shape we have never seen."""
    out = secrets.scrub("rejected: 9f3c2b1a8e7d6c5b4a39", "9f3c2b1a8e7d6c5b4a39")
    assert "9f3c2b1a8e7d6c5b4a39" not in out
    assert "9f3c" in out, "a redaction should still be recognisable"


@pytest.mark.parametrize("body", [
    "model llm:qwen3-4b-instruct-q4_k_m not found",
    "unable to access https://api.openai.com/v1/audio/transcriptions",
    '{"error":{"message":"Invalid Authentication","code":"invalid_api_key"}}',
    "ggml:large-v3-turbo-q5_0 failed to load",
    "",
])
def test_scrub_leaves_alone_what_is_not_a_credential(body):
    """A general high-entropy hunt flags model ids and teaches people to
    ignore the redactions."""
    assert secrets.scrub(body) == body


def test_the_scrub_is_actually_reached_by_a_backend(monkeypatch):
    """The function existing is not the same as it being called."""
    import json as jsonlib

    from omavoi.llm.openai_compat import OpenAiCompatBackend

    leaked = ('{"error":{"message":"Incorrect API key provided: '
              'sk-proj-VERYSECRET1234567890abc"}}')

    class Response:
        status_code = 401
        text = leaked
        def json(self): return jsonlib.loads(leaked)

    backend = OpenAiCompatBackend("api", {
        "backend": "openai", "base_url": "https://x/v1", "model": "m",
        "key_env": "", "key_name": "n", "max_tokens": 10,
        "temperature": 0.0, "timeout": 5.0})
    monkeypatch.setattr(backend, "_ensure",
                        lambda: type("C", (), {"post": lambda *a, **k: Response()})())
    result = backend.complete("", "hello")
    assert "VERYSECRET" not in result.error, result.error
    assert "401" in result.error, "and it still says what went wrong"


def test_recordings_written_before_this_are_tightened_too(home):
    """Only new WAVs pass through _write_wav; the trim visits all of them."""
    cfg = config.load()
    cfg["history"]["keep_audio"] = 10
    h = history.History(cfg)
    rec = paths.recordings_dir()
    rec.mkdir(parents=True, exist_ok=True)
    rec.chmod(0o755)
    old = rec / "deadbeef.wav"
    old.write_bytes(b"RIFF" + b"\0" * 40)
    old.chmod(0o644)

    # A take that names it, so the trim keeps rather than deletes it.
    h.record({"text": "t", "id": "deadbeef", "wav": str(old)})
    h._trim()

    assert old.exists(), "it was named by a kept take"
    assert _mode(old) & 0o077 == 0, f"the recording is {oct(_mode(old))}"
    assert _mode(rec) & 0o077 == 0, f"the recordings dir is {oct(_mode(rec))}"
