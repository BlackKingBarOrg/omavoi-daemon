"""A long download has to show how far along it is.

Three gigabytes used to say "downloading" and nothing else until it finished,
so a slow mirror and a stalled one looked exactly alike.
"""
from __future__ import annotations

from omavoi import models


def _incomplete(home, key: str, size: int) -> None:
    entry = models.spec(key)
    assert entry is not None
    root = models.llm_dir() if entry.kind == models.LLM else models.model_root() / "ggml"
    d = root / ".cache" / "huggingface" / "download"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{entry.filename}.incomplete").write_bytes(b"\0" * size)


def test_nothing_in_flight_is_zero(home):
    assert models.bytes_in_flight("ggml:large-v3") == 0


def test_an_unknown_key_is_zero(home):
    assert models.bytes_in_flight("no-such-model") == 0


def test_a_partial_file_is_counted(home):
    _incomplete(home, "ggml:large-v3", 4096)
    assert models.bytes_in_flight("ggml:large-v3") == 4096


def test_the_llm_side_looks_in_its_own_directory(home):
    _incomplete(home, "llm:qwen3-4b", 2048)
    assert models.bytes_in_flight("llm:qwen3-4b") == 2048
    # And the two families do not read each other's.
    assert models.bytes_in_flight("ggml:large-v3") == 0


def test_the_percentage_can_be_computed_from_the_payload(home):
    """What the row actually does with it: bytes over the catalogue's size."""
    _incomplete(home, "ggml:large-v3", 1024 * 1024 * 100)
    spec = models.spec("ggml:large-v3")
    done = models.bytes_in_flight("ggml:large-v3")
    pct = min(99, int(100 * done / (spec.size_mb * 1048576)))
    assert 0 < pct < 10, f"100 MB of {spec.size_mb} MB should be a few percent, got {pct}"
