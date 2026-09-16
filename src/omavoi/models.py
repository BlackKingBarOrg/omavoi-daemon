"""Model catalog: what you can download, what you already have, in both formats.

Two on-disk formats, because the two local engines can't share weights:

  ct2   CTranslate2, for faster-whisper. CUDA only, fastest on NVIDIA.
  ggml  whisper.cpp. Runs on Vulkan (any GPU), CUDA, ROCm, or CPU.

Existing ggml files from other tools are discovered rather than re-downloaded —
a 3 GB model is not worth having twice.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from . import paths

CT2, GGML = "ct2", "ggml"
SPEECH, LLM = "speech", "llm"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    fmt: str              # CT2 | GGML
    backend: str          # which backend runs it
    repo: str             # HuggingFace repo
    filename: str = ""    # single-file models (ggml); empty = whole snapshot
    size_mb: int = 0
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    kind: str = SPEECH    # SPEECH | LLM
    languages: str = ""

    @property
    def key(self) -> str:
        """How you name it: `large-v3`, `ggml:large-v3`, `llm:qwen3-8b`."""
        if self.kind == LLM:
            return f"{LLM}:{self.id}"
        return self.id if self.fmt == CT2 else f"{GGML}:{self.id}"


_CT2 = "local-whisper"
_CPP = "local-whispercpp"
_LLM = "llama-local"
_HF_CPP = "ggerganov/whisper.cpp"

CATALOG: tuple[ModelSpec, ...] = (
    # -- CTranslate2 (CUDA) -------------------------------------------------
    ModelSpec("tiny", CT2, _CT2, "Systran/faster-whisper-tiny", "", 75,
              "Proves the pipeline runs. Not usable for real dictation.", ("test",)),
    ModelSpec("base", CT2, _CT2, "Systran/faster-whisper-base", "", 145,
              "Better than tiny, still not worth using daily.", ("test",)),
    ModelSpec("small", CT2, _CT2, "Systran/faster-whisper-small", "", 480,
              "Passable in English, struggles elsewhere."),
    ModelSpec("medium", CT2, _CT2, "Systran/faster-whisper-medium", "", 1530,
              "The floor of usable. A fallback when VRAM is tight."),
    ModelSpec("large-v2", CT2, _CT2, "Systran/faster-whisper-large-v2", "", 3090,
              "The previous large. Steadier on some accents."),
    ModelSpec("large-v3", CT2, _CT2, "Systran/faster-whisper-large-v3", "", 3090,
              "Best all-round. no_speech_prob is trustworthy, so silence is caught.", ("recommended",)),
    ModelSpec("large-v3-turbo", CT2, _CT2,
              "mobiuslabsgmbh/faster-whisper-large-v3-turbo", "", 1620,
              "2x faster, but no_speech_prob is always 0 — it cannot detect silence.", ("fast",)),
    ModelSpec("distil-large-v3", CT2, _CT2,
              "Systran/faster-distil-whisper-large-v3", "", 1510,
              "English-only distillation. Unusable for other languages.", ("fast", "en")),

    # -- whisper.cpp / ggml (Vulkan, any GPU) --------------------------------
    ModelSpec("base", GGML, _CPP, _HF_CPP, "ggml-base.bin", 148,
              "Proves the pipeline runs.", ("test",)),
    ModelSpec("small", GGML, _CPP, _HF_CPP, "ggml-small.bin", 488,
              "Passable in English."),
    ModelSpec("medium", GGML, _CPP, _HF_CPP, "ggml-medium.bin", 1530,
              "The floor of usable."),
    ModelSpec("large-v3", GGML, _CPP, _HF_CPP, "ggml-large-v3.bin", 3095,
              "The default. Runs on any GPU through Vulkan.", ("recommended",)),
    ModelSpec("large-v3-turbo", GGML, _CPP, _HF_CPP, "ggml-large-v3-turbo.bin", 1624,
              "Faster, but poor at telling silence apart.", ("fast",)),
    ModelSpec("large-v3-q5_0", GGML, _CPP, _HF_CPP, "ggml-large-v3-q5_0.bin", 1080,
              "Quantised large-v3: a third of the VRAM, slightly less accurate.", ("quant",)),
    ModelSpec("large-v3-turbo-q5_0", GGML, _CPP, _HF_CPP,
              "ggml-large-v3-turbo-q5_0.bin", 574,
              "The lightest thing still worth using.", ("quant", "fast")),

    # -- LLM, run by the bundled llama-server ---------------------------------
    #
    # Every repo and filename here was checked to exist before it went in: a
    # catalogue entry that 404s on click is worse than no entry.
    ModelSpec("qwen3-4b", GGML, _LLM, "Qwen/Qwen3-4B-GGUF",
              "Qwen3-4B-Q4_K_M.gguf", 2355,
              "Fast, and the best Chinese at this size. Other languages are "
              "along for the ride.", ("fast",), kind=LLM,
              languages="strong zh/en"),
    ModelSpec("qwen3-8b", GGML, _LLM, "Qwen/Qwen3-8B-GGUF",
              "Qwen3-8B-Q4_K_M.gguf", 4812,
              "The same strengths with more room. A good default when the "
              "source language is Chinese.", ("recommended",), kind=LLM,
              languages="strong zh/en"),
    ModelSpec("gemma-3-4b", GGML, _LLM, "ggml-org/gemma-3-4b-it-GGUF",
              "gemma-3-4b-it-Q4_K_M.gguf", 2355,
              "Broader language coverage than Qwen at this size, which shows "
              "on translation into anything but English.", ("fast",), kind=LLM,
              languages="broad multilingual"),
    ModelSpec("gemma-3-12b", GGML, _LLM, "ggml-org/gemma-3-12b-it-GGUF",
              "gemma-3-12b-it-Q4_K_M.gguf", 6963,
              "The best translation here, and the heaviest. Leaves little room "
              "beside a large speech model.", (), kind=LLM,
              languages="broad multilingual"),
)


def parse_key(key: str) -> tuple[str, str]:
    """`ggml:large-v3` -> ("ggml", "large-v3"); bare names default to ct2."""
    if ":" in key:
        fmt, _, name = key.partition(":")
        return fmt.strip().lower(), name.strip()
    return CT2, key.strip()


def spec(key: str) -> ModelSpec | None:
    prefix, name = parse_key(key)
    for entry in CATALOG:
        if prefix == LLM:
            if entry.kind == LLM and entry.id == name:
                return entry
        elif entry.kind == SPEECH and entry.fmt == prefix and entry.id == name:
            return entry
    return None


def model_root() -> Path:
    return paths.data_dir() / "models"


def llm_dir() -> Path:
    return model_root() / "llm"


def ggml_search_dirs() -> list[Path]:
    """Where a ggml model might already live, ours first."""
    return [
        model_root() / "ggml",
        paths.data_dir().parent / "voxtype" / "models",   # left behind by voxtype
        Path.home() / ".cache" / "whisper.cpp",
        Path("/usr/share/whisper.cpp/models"),
    ]


def _hf_dirname(repo: str) -> str:
    return "models--" + repo.replace("/", "--")


def bytes_in_flight(key: str) -> int:
    """How much of `key` is on disk so far, mid-download, or 0.

    A three-gigabyte download said "downloading" and nothing else until it
    finished, so there was no way to tell a slow mirror from a stalled one.
    huggingface_hub with local_dir= writes to
    `<dir>/.cache/huggingface/download/<filename>.incomplete`, and its length
    is the answer. Read rather than reported by the downloader itself because
    the download runs in another process — the console asks the catalogue,
    which is a file on disk either way.
    """
    entry = spec(key)
    if entry is None or not entry.filename:
        return 0
    root = model_root()
    for directory in (llm_dir() if entry.kind == LLM else root / "ggml",):
        part = directory / ".cache" / "huggingface" / "download" / f"{entry.filename}.incomplete"
        try:
            return part.stat().st_size
        except OSError:
            return 0
    return 0


def local_path(key: str) -> Path | None:
    """Where this model is on disk, or None."""
    entry = spec(key)
    if entry is None:
        return None

    if entry.kind == LLM:
        candidate = llm_dir() / entry.filename
        return candidate if candidate.is_file() else None

    if entry.fmt == GGML:
        for directory in ggml_search_dirs():
            candidate = directory / entry.filename
            if candidate.is_file():
                return candidate
        return None

    for root in (model_root(), paths.hf_cache_dir()):
        base = root / _hf_dirname(entry.repo) / "snapshots"
        if not base.is_dir():
            continue
        for snapshot in sorted(base.iterdir(), reverse=True):
            if (snapshot / "model.bin").exists():
                return snapshot
    return None


def is_downloaded(key: str) -> bool:
    return local_path(key) is not None


def owned_by_us(key: str) -> bool:
    """True only if it lives in our store — we never delete someone else's copy."""
    path = local_path(key)
    if path is None:
        return False
    try:
        path.relative_to(model_root())
        return True
    except ValueError:
        return False


def pull(key: str) -> Path:
    entry = spec(key)
    if entry is None:
        raise ValueError(f"unknown model {key!r}; see `omavoi model list`")

    root = model_root()
    root.mkdir(parents=True, exist_ok=True)

    if entry.kind == LLM:
        from huggingface_hub import hf_hub_download

        target = llm_dir()
        target.mkdir(parents=True, exist_ok=True)
        return Path(hf_hub_download(entry.repo, entry.filename, local_dir=str(target)))

    if entry.fmt == GGML:
        from huggingface_hub import hf_hub_download

        target = root / "ggml"
        target.mkdir(parents=True, exist_ok=True)
        downloaded = hf_hub_download(entry.repo, entry.filename, local_dir=str(target))
        return Path(downloaded)

    try:
        from faster_whisper.utils import download_model
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"{key} is a CTranslate2 model, which needs faster-whisper. "
            f"Run: uv tool install --reinstall omavoi[cuda]"
        ) from exc

    return Path(download_model(entry.repo, cache_dir=str(root), local_files_only=False))


def remove(key: str) -> bool:
    entry = spec(key)
    if entry is None:
        raise ValueError(f"unknown model {key!r}")
    if not owned_by_us(key):
        return False
    if entry.kind == LLM or entry.fmt == GGML:
        path = local_path(key)
        if path is None:
            return False
        path.unlink()
        return True
    target = model_root() / _hf_dirname(entry.repo)
    if not target.exists():
        return False
    shutil.rmtree(target)
    return True


def resolve_for_load(key: str) -> str:
    """What to hand the backend: a local path when we have one, else the repo id."""
    path = local_path(key)
    if path is not None:
        return str(path)
    entry = spec(key)
    return entry.repo if entry else key
