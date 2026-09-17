"""The faster-whisper / CTranslate2 engine is gone, and configs move off it.

It was NVIDIA-only, needed about 2.2 GB of CUDA wheels, brought a second
model format, a second set of runtime-library problems and a module whose
whole job was making CTranslate2 find cuBLAS — and on the short push-to-talk
takes this program is for it measured within a tenth of a second of
whisper.cpp on Vulkan, because fixed overhead dominates at that length.

The part worth testing is not the removal, it is the landing: an install that
was using it must come back up. Without the migration `asr.build` raises on
an unknown backend and the daemon fails on every start, which is the loudest
possible way to deliver a removal, to someone who did nothing but update.
"""
from __future__ import annotations

import pytest

from omavoi import asr, config, i18n, models


def test_the_engine_is_not_offered_any_more():
    assert "local-whisper" not in asr.BACKENDS
    assert set(asr.BACKENDS) == {"local-whispercpp", "api"}
    # The names that meant that engine specifically resolve to nothing.
    assert asr.canonical("ct2") is None
    assert asr.canonical("faster-whisper") is None
    # The ones that just meant "the local one" land on the local one.
    assert asr.canonical("local") == "local-whispercpp"
    assert asr.canonical("whisper") == "local-whispercpp"


def test_no_ct2_models_are_left_in_the_catalogue():
    assert all(entry.fmt == "ggml" for entry in models.CATALOG
               if entry.kind == models.SPEECH)
    assert not hasattr(models, "CT2")
    # And every key carries its prefix, which the bare ct2 spelling did not.
    assert all(":" in entry.key for entry in models.CATALOG)


@pytest.mark.parametrize("backend", ["local-whisper", "faster-whisper", "ct2"])
def test_a_config_on_the_old_engine_comes_back_up(home, backend):
    cfg = config.defaults()
    cfg["speech"]["backend"] = backend
    notes = config.retire_cuda_engine(cfg)

    assert cfg["speech"]["backend"] == "local-whispercpp"
    assert notes, "a change to the config file must be reported"
    assert asr.canonical(cfg["speech"]["backend"]) is not None, "build would raise"


def test_the_engines_settings_block_is_dropped(home):
    cfg = config.defaults()
    cfg["speech"]["local_whisper"] = {"device": "cuda", "compute_type": "float16"}
    notes = config.retire_cuda_engine(cfg)
    assert "local_whisper" not in cfg["speech"]
    assert any("local_whisper" in note for note in notes)


@pytest.mark.parametrize("was,becomes", [
    # Six of the eight ct2 names exist verbatim in the ggml catalogue.
    ("base", "ggml:base"),
    ("small", "ggml:small"),
    ("medium", "ggml:medium"),
    ("large-v3", "ggml:large-v3"),
    ("large-v3-turbo", "ggml:large-v3-turbo"),
    # large-v2 has no ggml build; the shipped default is the nearest large.
    ("large-v2", "ggml:large-v3"),
    # These two have no equivalent, and falling through to the 3 GB default
    # would hand someone who chose 75 MB a download they did not ask for.
    ("tiny", "ggml:base"),
    ("distil-large-v3", "ggml:large-v3-turbo"),
])
def test_every_ct2_model_name_lands_on_something_comparable(home, was, becomes):
    cfg = config.defaults()
    cfg["speech"]["model"] = was
    config.retire_cuda_engine(cfg)
    assert cfg["speech"]["model"] == becomes
    assert models.spec(cfg["speech"]["model"]) is not None, "it must exist"


def test_a_model_that_names_nothing_falls_back_rather_than_stopping_the_daemon(home):
    cfg = config.defaults()
    cfg["speech"]["model"] = "ggml:invented"
    config.retire_cuda_engine(cfg)
    assert cfg["speech"]["model"] == config.DEFAULTS["speech"]["model"]


def test_a_current_config_is_left_alone(home):
    cfg = config.defaults()
    before = dict(cfg["speech"])
    assert config.retire_cuda_engine(cfg) == [], "nothing to say about a fresh config"
    assert cfg["speech"] == before


def test_the_migration_runs_on_load(home):
    """Not just when called: `config.load` is the only thing that runs it."""
    path = config.paths.config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('[speech]\nbackend = "local-whisper"\nmodel = "large-v3"\n',
                    encoding="utf-8")
    cfg = config.load()
    assert cfg["speech"]["backend"] == "local-whispercpp"
    assert cfg["speech"]["model"] == "ggml:large-v3"
    # And written back, or it happens again on every load and the file keeps
    # disagreeing with what is running.
    assert "local-whispercpp" in path.read_text(encoding="utf-8")


def test_no_translation_describes_a_model_that_is_gone():
    """The eight ct2 notes were translated into seven languages each."""
    live = {entry.note for entry in models.CATALOG} | {config.DEFAULT_STEP_PROMPT}
    orphans = [key for key in i18n._TABLE if key not in live]
    assert not orphans, orphans
