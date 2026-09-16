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

from omavoi import config
from omavoi.commands import catalogue


def payload(home) -> dict:
    args = argparse.Namespace(action="list", rest=[], json=True, force=False)
    buf = io.StringIO()
    with redirect_stdout(buf):
        assert catalogue.cmd_model(args) == 0
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
    for field in ("base_url", "model", "key_env", "has_key", "ready",
                  "default_base_url", "default_model"):
        assert field in p["speech_api"], f"speech_api has no {field}"


def test_has_key_means_a_key_and_ready_means_nothing_is_missing(home):
    """One flag was answering two questions, and lying about one of them.

    `has_key` drives the console's "a key is stored" label; `ready` drives
    whether the endpoint panel opens itself because nothing is configured.
    They were the same field, defined as `bool(key) or not key_env` — so an
    endpoint that needs no key reported that a key was stored, and a custom
    base_url with no key and no key_env reported the same. Someone reading
    "a key is stored" then got a 401 from an endpoint the console had just
    called configured.

    No key resolves in a fresh home, so has_key must be false everywhere
    here; ready may be true, and that is exactly the difference.
    """
    p = payload(home)
    endpoints = [("speech_api", p["speech_api"])]
    endpoints += [(e["name"], e) for e in p["llm"]]
    for name, e in endpoints:
        if "has_key" not in e:
            continue
        assert e["has_key"] is False, (
            f"{name} claims a key with nothing in the environment or the file"
        )
        assert "ready" in e, f"{name} has has_key but no ready"


def test_no_payload_carries_a_key_value(home):
    """Not even a redacted one, and nothing ever read it.

    Three payloads carried `key` as `sk-…abcd`. The console never looked at
    it — it reads key_source and has_key — so it was a secret-shaped value
    travelling through a JSON blob that lands in logs and screenshots for no
    reader at all.
    """
    p = payload(home)
    blobs = [p["speech_api"], *p["llm"]]
    for blob in blobs:
        assert "key" not in blob, f"a key value is still in {blob.get('name', 'speech_api')}"


def test_llm_entries_say_where_their_key_lives_too(home):
    for entry in payload(home)["llm"]:
        assert "key_name" in entry, f"{entry['name']} does not say"
        assert entry["key_name"], f"{entry['name']} has an empty key name"
