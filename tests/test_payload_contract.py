"""What `omavoi model list --json` promises the console.

Two asymmetries hid here for a long time. `fits` and `needed_mb` were computed
only for LLMs, so a speech model that would not load reported fits:true and
the speech table had no warning to show. And `active` compared every entry
against the *speech* model, so no llm entry could ever be active however many
configurations pointed at it — which the UI then worked around with a second,
different way of asking.

The key name is here too: ApiWhisperBackend reads its key as `key_name or
provider`, and the console wrote "speech-api", so a key saved from the UI went
somewhere nothing would ever look.
"""
from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stdout

from omavoi import cli, config


def payload(home) -> dict:
    args = argparse.Namespace(action="list", rest=[], json=True, force=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert cli.cmd_model(args) == 0
    return json.loads(buf.getvalue())


def test_both_families_carry_the_same_fields(home):
    models = payload(home)["models"]
    speech = {k for m in models if m["kind"] == "speech" for k in m}
    llm = {k for m in models if m["kind"] == "llm" for k in m}
    assert speech and llm
    assert speech == llm, f"speech-only {speech - llm}, llm-only {llm - speech}"


def test_every_model_answers_whether_it_fits(home):
    """Not just the LLMs. Every model costs VRAM, so every one can fail."""
    for m in payload(home)["models"]:
        assert isinstance(m["fits"], bool)
        assert isinstance(m["needed_mb"], int | float)
        assert m["needed_mb"] > 0, f"{m['key']} claims to need nothing"


def test_active_is_answerable_for_both_families(home):
    """`active` used to compare everything against the speech model, so an
    llm entry could never be active."""
    cfg = config.load()
    chosen = {str(e.get("model", "")) for e in cfg["llm"].values() if e.get("model")}
    models = payload(home)["models"]
    by_key = {m["key"]: m for m in models}
    for key in chosen:
        assert key in by_key, f"{key} is configured but not in the catalogue"
        assert by_key[key]["active"] is True, f"{key} is pointed at and not active"
    speech = cfg["speech"]["model"]
    if speech in by_key:
        assert by_key[speech]["active"] is True


def test_the_console_is_told_where_the_speech_key_lives(home):
    """It has to match what ApiWhisperBackend will read, or the key is
    saved into a name nothing looks up."""
    from omavoi.asr.api_whisper import ApiWhisperBackend

    p = payload(home)
    assert "speech_api" in p, "the remote speech endpoint has no payload"
    backend = ApiWhisperBackend(config.load())
    assert p["speech_api"]["key_name"] == backend.key_name
    for field in ("base_url", "model", "key_env", "has_key",
                  "default_base_url", "default_model"):
        assert field in p["speech_api"], f"speech_api has no {field}"


def test_llm_entries_say_where_their_key_lives_too(home):
    for entry in payload(home)["llm"]:
        assert "key_name" in entry, f"{entry['name']} does not say"
        assert entry["key_name"], f"{entry['name']} has an empty key name"
