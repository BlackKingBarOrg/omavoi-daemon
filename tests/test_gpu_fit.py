"""The arithmetic that refuses a mode switch.

Switching to a mode whose local model will not fit in free VRAM is refused
rather than left to fail on the next take. That makes these numbers a gate in
front of a feature, and a gate that is wrong in the strict direction refuses
something that would have worked — with `--force` as the only way through and
no way to tell whether the refusal was right.

All of it is pure except the one call to nvidia-smi, which is patched here.
None of these tests needs a GPU.
"""
from __future__ import annotations

import pytest

from omavoi import gpu

# -- needed_mb --------------------------------------------------------------


def test_cpu_only_needs_no_vram():
    """gpu_layers <= 0 means the weights never reach the card."""
    assert gpu.needed_mb(8000, gpu_layers=0) == 0
    assert gpu.needed_mb(8000, gpu_layers=-1) == 0


def test_the_estimate_is_above_the_weights():
    """Weights, a KV cache, and room for the runtime.

    Generous on purpose: refusing something that would just have fit is a
    smaller sin than evicting the speech model mid-sentence.
    """
    assert gpu.needed_mb(4000) > 4000


def test_the_kv_cache_grows_with_the_context():
    small = gpu.needed_mb(4000, ctx_size=2048)
    large = gpu.needed_mb(4000, ctx_size=32768)
    assert large > small, "a longer context needs more KV cache"


def test_a_tiny_context_still_reserves_a_floor():
    """max(128, ...), so a 1-token context does not imply a free lunch."""
    assert gpu.needed_mb(1000, ctx_size=1) - int(1000 * 1.03) >= 128 + 256


# -- fits -------------------------------------------------------------------


@pytest.fixture
def card(monkeypatch):
    """A discrete card with a pool of its own, at whatever size a test wants."""
    def setup(free_mb, total_mb=16000):
        info = {"name": "Test GPU", "total_mb": total_mb,
                "used_mb": total_mb - free_mb, "free_mb": free_mb}
        monkeypatch.setattr(gpu, "vram", lambda **kw: info)
        monkeypatch.setattr(gpu, "holders", lambda limit=3: [])
        return info
    return setup


def test_a_model_that_fits_fits(card):
    card(free_mb=12000)
    out = gpu.fits(4000)
    assert out["known"] is True
    assert out["fits"] is True


def test_a_model_that_does_not_fit_says_so_and_names_the_holders(card, monkeypatch):
    card(free_mb=1000)
    monkeypatch.setattr(gpu, "holders",
                        lambda limit=3: [gpu.Holder(pid=42, name="something",
                                                    used_mb=9000)])
    out = gpu.fits(8000)
    assert out["fits"] is False
    assert out["free_mb"] == 1000
    assert out["holders"], "a refusal must say what is holding the memory"
    assert out["holders"][0]["used_mb"] == 9000


def test_nothing_to_interrogate_must_not_block(monkeypatch):
    """No NVIDIA card, or an integrated one whose budget is system RAM.

    Overcommitting shared memory costs swap, not the allocation failure this
    guard exists for — a different conversation, and not one to have by
    refusing to start.
    """
    for info in ({}, {"unified": True, "total_mb": 32000, "free_mb": 100}):
        # Bound, not captured: the lambda outlives the iteration it is made in.
        monkeypatch.setattr(gpu, "vram", lambda _i=info, **kw: _i)
        out = gpu.fits(20000)
        assert out["fits"] is True, info
        assert out["known"] is False, "it must not claim to know"


def test_a_cpu_only_model_fits_on_a_full_card(card):
    card(free_mb=0)
    assert gpu.fits(20000, gpu_layers=0)["fits"] is True


# -- segments ---------------------------------------------------------------


def test_the_bar_adds_up_to_the_number_printed_beside_it(monkeypatch):
    """`other` is computed, not summed.

    nvidia-smi's total includes memory no compute app claims — the display,
    mostly — and a stacked bar that does not add up to its own total is worse
    than one honest catch-all.
    """
    monkeypatch.setattr(gpu, "usage_by_pid", lambda: {10: 2000, 20: 3000, 30: 500})
    segs = gpu.segments(total_used_mb=7000, speech_pid=10, llm_pids={20})
    by = {s["kind"]: s["used_mb"] for s in segs}
    assert by["speech"] == 2000
    assert by["llm"] == 3000
    assert by["other"] == 2000, "7000 - 2000 - 3000"
    assert sum(by.values()) == 7000


def test_a_segment_never_goes_negative(monkeypatch):
    """Our own processes can be reported as holding more than the total."""
    monkeypatch.setattr(gpu, "usage_by_pid", lambda: {10: 9000})
    segs = gpu.segments(total_used_mb=1000, speech_pid=10, llm_pids=set())
    assert all(s["used_mb"] >= 0 for s in segs)


def test_no_speech_pid_is_not_pid_zero(monkeypatch):
    """0 is falsy and also a real key shape; it must not match a holder."""
    monkeypatch.setattr(gpu, "usage_by_pid", lambda: {0: 5000})
    segs = gpu.segments(total_used_mb=5000, speech_pid=0, llm_pids=set())
    by = {s["kind"]: s["used_mb"] for s in segs}
    assert by["speech"] == 0
    assert by["other"] == 5000


# -- explain_shortfall ------------------------------------------------------


def test_the_refusal_says_the_two_numbers_and_who_has_the_memory():
    msg = gpu.explain_shortfall(
        {"free_mb": 1024, "needed_mb": 8192,
         "holders": [{"pid": 1, "name": "firefox", "used_mb": 4096}]},
        "llm:qwen",
    )
    assert "llm:qwen" in msg
    assert "8.0 GB" in msg, "what it needs"
    assert "1.0 GB" in msg, "what is free"
    assert "firefox" in msg and "4.0 GB" in msg, "and who is holding it"


def test_it_says_something_useful_with_no_holder_list():
    msg = gpu.explain_shortfall({"free_mb": 0, "needed_mb": 4096}, "llm:x")
    assert "llm:x" in msg
    assert "Currently held by" not in msg


# -- the nvidia-smi parse ---------------------------------------------------


def _parse(stdout, monkeypatch, rc=0):
    import subprocess

    class R:
        returncode = rc
        def __init__(self): self.stdout = stdout.encode()
    monkeypatch.setattr(gpu.shutil, "which", lambda n: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: R())
    return gpu._nvidia_vram()


def test_a_normal_reading_parses(monkeypatch):
    got = _parse("NVIDIA GeForce RTX 5070 Ti, 12340, 16303\n", monkeypatch)
    assert got["name"] == "NVIDIA GeForce RTX 5070 Ti"
    assert got["used_mb"] == 12340
    assert got["total_mb"] == 16303
    assert got["free_mb"] == 3963


def test_a_comma_in_the_gpu_name_does_not_read_as_no_gpu(monkeypatch):
    """It split on every comma, so this unpack raised and the handler below
    turned it into {} — "no NVIDIA GPU", which is the same shape as evdev
    reporting no keyboard when it means no permission."""
    got = _parse("NVIDIA RTX A6000, Ada, 1234, 49140\n", monkeypatch)
    assert got["name"] == "NVIDIA RTX A6000, Ada"
    assert got["used_mb"] == 1234
    assert got["total_mb"] == 49140


def test_only_the_first_gpu_is_read(monkeypatch):
    """Which is the one both engines use: neither is told a device, so both
    take index 0."""
    got = _parse("GPU Zero, 100, 8000\nGPU One, 200, 24000\n", monkeypatch)
    assert got["name"] == "GPU Zero"
    assert got["total_mb"] == 8000


def test_junk_is_no_reading_rather_than_a_wrong_one(monkeypatch):
    assert _parse("", monkeypatch) == {}
    assert _parse("not,numbers,here\n", monkeypatch) == {}
    assert _parse("GPU, 1, 2\n", monkeypatch, rc=9) == {}


# -- the cache --------------------------------------------------------------


def test_the_cache_collapses_repeated_calls(monkeypatch):
    """`model list` asked nineteen times at 16 ms each."""
    calls = []

    def counted():
        calls.append(1)
        return {"name": "G", "used_mb": 1, "total_mb": 2, "free_mb": 1}

    monkeypatch.setattr(gpu, "_nvidia_vram", counted)
    monkeypatch.setattr(gpu, "_vram_cache", None)
    for _ in range(19):
        gpu.vram()
    assert len(calls) == 1, f"asked {len(calls)} times"


def test_fresh_bypasses_the_cache(monkeypatch):
    calls = []

    def counted():
        calls.append(1)
        return {"name": "G", "used_mb": 1, "total_mb": 2, "free_mb": 1}

    monkeypatch.setattr(gpu, "_nvidia_vram", counted)
    monkeypatch.setattr(gpu, "_vram_cache", None)
    gpu.vram()
    gpu.vram(fresh=True)
    assert len(calls) == 2
