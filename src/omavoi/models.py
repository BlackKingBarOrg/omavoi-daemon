"""Model catalog: what you can download, what you already have, in both formats.

Two on-disk formats, because the two local engines can't share weights:

  ggml  whisper.cpp. Runs on Vulkan (any GPU), CUDA, ROCm, or CPU.
        Qwen3-ASR's GGUF files too: the same ggml family, run by llama.cpp,
        and keyed `ggml:` because a speech key has always been.

Existing ggml files from other tools are discovered rather than re-downloaded —
a 3 GB model is not worth having twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from . import paths

GGML = "ggml"
SPEECH, LLM = "speech", "llm"


@dataclass(frozen=True, slots=True)
class ModelSpec:
    id: str
    fmt: str              # GGML — the only speech format left
    backend: str          # which backend runs it
    repo: str             # HuggingFace repo
    filename: str = ""    # single-file models (ggml); empty = whole snapshot
    size_mb: int = 0
    note: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)
    kind: str = SPEECH    # SPEECH | LLM
    languages: str = ""
    # Files beside `filename` that the model cannot run without, fetched,
    # found and removed with it. Qwen3-ASR's audio encoder is a second GGUF,
    # and weights without it are not "downloaded" in any sense that matters.
    extra_files: tuple[str, ...] = ()
    # The recognition languages it takes, as speech_languages.json spells
    # them, when that is not every one whisper has a token for. The console's
    # language picker offers these and no others.
    language_codes: tuple[str, ...] = ()

    @property
    def files(self) -> tuple[str, ...]:
        return (self.filename, *self.extra_files)

    @property
    def key(self) -> str:
        """How you name it: `ggml:large-v3`, `llm:qwen3-8b`.

        Every key carries its prefix now. The bare spelling — `large-v3` —
        belonged to the ct2 models, which were the faster-whisper engine's,
        and both are gone.
        """
        return f"{LLM if self.kind == LLM else GGML}:{self.id}"


_CPP = "local-whispercpp"
_ASR = "local-llama-asr"
_LLM = "llama-local"
_HF_CPP = "ggerganov/whisper.cpp"

CATALOG: tuple[ModelSpec, ...] = (
    # -- Qwen3-ASR, run by llama-server (Vulkan, any GPU) ---------------------
    #
    # The default, and first so the console lists it first. Not whisper, so
    # not whisper.cpp: asr/local_llama_asr.py starts its own llama-server for
    # it. Measured on 2026-10-01 on an Intel Arc B390 over twenty recorded
    # takes: 0.80 s a take against large-v3-turbo's 3.38 s, with the two
    # transcripts 2.9% apart. The accuracy claim in the note is the published
    # one — AISHELL-1 CER 1.48 against whisper-large-v3's 5.14, Open ASR
    # Leaderboard English WER 4.31 against turbo's 6.36 — and not a
    # measurement of ours, which had no reference transcripts to score.
    #
    # What it costs against turbo: about 0.8 GB more download, llama-cpp,
    # which the first-run screen installs anyway, and 30 languages where
    # whisper has 99. Its card lists them; Filipino is `tl` here, the code
    # whisper's tokenizer gave it.
    ModelSpec("qwen3-asr-1.7b", GGML, _ASR, "ggml-org/Qwen3-ASR-1.7B-GGUF",
              "Qwen3-ASR-1.7B-Q8_0.gguf", 2404,
              "The default. Much more accurate than whisper in Chinese, a little "
              "better in English, and faster. Runs on llama.cpp.",
              ("recommended", "accurate", "fast"),
              languages="30 languages and 22 Chinese dialects",
              extra_files=("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",),
              language_codes=("zh", "en", "yue", "ar", "de", "fr", "es", "pt", "id",
                              "it", "ko", "ru", "th", "vi", "ja", "tr", "hi", "ms",
                              "nl", "sv", "da", "fi", "pl", "cs", "tl", "fa", "el",
                              "hu", "mk", "ro")),

    # -- whisper.cpp / ggml (Vulkan, any GPU) --------------------------------
    ModelSpec("base", GGML, _CPP, _HF_CPP, "ggml-base.bin", 141,
              "Only for checking that everything works.", ("test",),
              languages="English only, in practice"),
    ModelSpec("small", GGML, _CPP, _HF_CPP, "ggml-small.bin", 465,
              "Passable in English.",
              languages="English and other widely spoken languages"),
    ModelSpec("medium", GGML, _CPP, _HF_CPP, "ggml-medium.bin", 1463,
              "The smallest one that is really usable.",
              languages="all 99 languages, some better than others"),
    ModelSpec("large-v3", GGML, _CPP, _HF_CPP, "ggml-large-v3.bin", 2952,
              "Twice the download, and it reports when it heard nothing.",
              ("accurate",),
              languages="all 99 languages, evenly"),
    # The default until qwen3-asr-1.7b, and the whisper to pick for a
    # language Qwen3-ASR does not have. Half the size of large-v3 and faster,
    # at one measured cost: a turbo model's no_speech_prob is 0 on every
    # segment, so the silence filter, which needs that evidence, never drops
    # anything for it. Over 291 takes on one machine it typed two stock
    # hallucinations — "谢谢大家!" and a subtitle credit — both on
    # recordings below -58 dBFS, where the quietest take that carried real
    # speech was -51.
    ModelSpec("large-v3-turbo", GGML, _CPP, _HF_CPP, "ggml-large-v3-turbo.bin", 1549,
              "Half the size of large-v3 and faster. Cannot tell silence "
              "from speech, so a silent take may type a stock phrase.",
              ("fast",),
              languages="all 99 languages, some better than others"),
    ModelSpec("large-v3-q5_0", GGML, _CPP, _HF_CPP, "ggml-large-v3-q5_0.bin", 1031,
              "A compressed large-v3: a third of the graphics memory, slightly less accurate.", ("quant",),
              languages="all 99 languages, evenly"),
    ModelSpec("large-v3-turbo-q5_0", GGML, _CPP, _HF_CPP,
              "ggml-large-v3-turbo-q5_0.bin", 547,
              "The lightest thing still worth using.", ("quant", "fast"),
              languages="all 99 languages, some better than others"),

    # -- LLM, run by the bundled llama-server ---------------------------------
    #
    # Every repo and filename here was checked to exist before it went in: a
    # catalogue entry that 404s on click is worse than no entry.
    ModelSpec("qwen3-4b", GGML, _LLM, "Qwen/Qwen3-4B-GGUF",
              "Qwen3-4B-Q4_K_M.gguf", 2382,
              "The strongest Chinese at this size, and noticeably weaker "
              "outside Chinese and English.", ("fast",), kind=LLM,
              languages="strongest in Chinese and English"),
    ModelSpec("qwen3-8b", GGML, _LLM, "Qwen/Qwen3-8B-GGUF",
              "Qwen3-8B-Q4_K_M.gguf", 4795,
              "The same strengths with more room. Pick it if you dictate in "
              "Chinese.", (), kind=LLM,
              languages="strongest in Chinese and English"),
    # Recommended because it is the one that does not assume what language
    # you speak. The tag was on qwen3-8b, whose own note said it was "a good
    # default when the source language is Chinese" — a recommendation with a
    # condition on it, used as the unconditional one, and shipped as
    # llm.local.model so a fresh install ran it whatever you dictated.
    ModelSpec("gemma-3-4b", GGML, _LLM, "ggml-org/gemma-3-4b-it-GGUF",
              "gemma-3-4b-it-Q4_K_M.gguf", 2374,
              "Even coverage across languages, and half the size of qwen3-8b. "
              "The default.", ("recommended", "fast"), kind=LLM,
              languages="good in many languages"),
    ModelSpec("gemma-3-12b", GGML, _LLM, "ggml-org/gemma-3-12b-it-GGUF",
              "gemma-3-12b-it-Q4_K_M.gguf", 6962,
              "The best translation here, and the heaviest. Leaves little room "
              "beside a large speech model.", ("accurate",), kind=LLM,
              languages="good in many languages"),
    # Google's own QAT file, which unlike Gemma 3's is not gated, so it
    # downloads without a login. Tried on 2026-10-01 against qwen3-4b,
    # qwen3-8b and gemma-3-4b (not gemma-3-12b) on 24 Chinese, code-switched
    # and English dictations through the shipped prompt. With the transcript
    # fenced off it kept the meaning of every one and still removed most of
    # the filler — qwen3-8b removed a little more but typed the fence into
    # three, and qwen3-4b kept every meaning by editing almost nothing — and
    # it translated Chinese into Thai best.
    # About 6 GB resident, so beside large-v3-turbo it wants more than an
    # 8 GB card has to give — which is why it is not the default.
    ModelSpec("gemma-4-e4b", GGML, _LLM, "google/gemma-4-E4B-it-qat-q4_0-gguf",
              "gemma-4-E4B_q4_0-it.gguf", 4916,
              "Careful with wording in every language, and translates well. Uses "
              "about 6 GB while it runs.", ("accurate",), kind=LLM,
              languages="good in many languages"),
)


def parse_key(key: str) -> tuple[str, str]:
    """`ggml:large-v3` -> ("ggml", "large-v3").

    A bare name used to mean a ct2 model. There are none, so it means the
    ggml one of that name — which is what someone typing `large-v3` wants,
    and what an upgraded config that still holds a bare key resolves to.
    """
    if ":" in key:
        fmt, _, name = key.partition(":")
        return fmt.strip().lower(), name.strip()
    return GGML, key.strip()


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
    directory = llm_dir() if entry.kind == LLM else model_root() / "ggml"
    # A model of several files counts the ones already finished too, so the
    # bar does not fall back to zero between them; but only while one is
    # still incomplete, because 0 is how "not downloading" is said.
    partial = done = 0
    for name in entry.files:
        try:
            partial += (directory / ".cache" / "huggingface" / "download"
                        / f"{name}.incomplete").stat().st_size
            continue
        except OSError:
            pass
        try:
            done += (directory / name).stat().st_size
        except OSError:
            pass
    return partial + done if partial else 0


def local_path(key: str) -> Path | None:
    """Where this model is on disk, or None."""
    entry = spec(key)
    if entry is None:
        return None

    # The first file is the model; the others have to be in the same
    # directory, since that is where the backend looks for them.
    dirs = [llm_dir()] if entry.kind == LLM else ggml_search_dirs()
    for directory in dirs:
        candidate = directory / entry.filename
        if candidate.is_file() and all((directory / name).is_file()
                                       for name in entry.extra_files):
            return candidate
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

    from huggingface_hub import hf_hub_download

    target = llm_dir() if entry.kind == LLM else root / "ggml"
    target.mkdir(parents=True, exist_ok=True)
    # The small companions first: the model file is what local_path asks
    # about, so it arriving last means a half-fetched model never reads as
    # downloaded.
    for name in entry.extra_files:
        hf_hub_download(entry.repo, name, local_dir=str(target))
    return Path(hf_hub_download(entry.repo, entry.filename, local_dir=str(target)))


def remove(key: str) -> bool:
    entry = spec(key)
    if entry is None:
        raise ValueError(f"unknown model {key!r}")
    if not owned_by_us(key):
        return False
    # Files, ggml and llm alike, and a companion goes with its model. The
    # directory-shaped case was the ct2 snapshot layout, and there are no
    # ct2 models.
    path = local_path(key)
    if path is None:
        return False
    for name in entry.extra_files:
        (path.parent / name).unlink(missing_ok=True)
    path.unlink()
    return True


def resolve_for_load(key: str) -> str:
    """What to hand the backend: a local path when we have one, else the repo id."""
    path = local_path(key)
    if path is not None:
        return str(path)
    entry = spec(key)
    return entry.repo if entry else key
