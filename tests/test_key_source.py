"""Which key is in use, not merely that one exists.

`secrets.resolve` checks the environment first and the file second, so a
variable left in a shell profile silently beats the key just pasted into the
console — and the console said "a key is stored", which was true either way.
"""
from __future__ import annotations

from omavoi import secrets


def test_nothing_stored_has_no_source(home):
    assert secrets.source_of("MY_KEY", "mine") == ""


def test_the_file_is_reported_as_the_file(home):
    secrets.store("mine", "sk-from-the-file")
    assert secrets.source_of("MY_KEY", "mine") == "file"


def test_the_environment_wins_and_says_so(home, monkeypatch):
    secrets.store("mine", "sk-from-the-file")
    monkeypatch.setenv("MY_KEY", "sk-from-the-environment")
    assert secrets.source_of("MY_KEY", "mine") == "env"
    assert secrets.resolve("MY_KEY", "mine") == "sk-from-the-environment", \
        "the file won, which would make the warning wrong"


def test_the_payload_carries_it(home):
    """What the console reads, for both families."""
    import argparse
    import io
    import json
    from contextlib import redirect_stdout

    from omavoi.commands import catalogue
    buf = io.StringIO()
    args = argparse.Namespace(action="list", rest=[], json=True, force=False)
    with redirect_stdout(buf):
        assert catalogue.cmd_model(args) == 0
    payload = json.loads(buf.getvalue())
    for entry in payload["llm"]:
        assert "key_source" in entry, f"{entry['name']} does not say"
    assert "key_source" in payload["speech_api"]
