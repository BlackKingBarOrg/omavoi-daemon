"""ASR backend interface and the result type.

The result carries more than text on purpose. Per-segment probabilities
are what turn "it dropped a word again" into "segment 3 came back at
avg_logprob -1.4, the model was guessing there".
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    # Only the `transcribe` signature names it, and annotations are
    # strings here. Importing it for real put 22 ms of numpy on the
    # path of every command, including the ones that never see audio:
    # this module is what `asr/__init__` imports eagerly, and the
    # backends below it are already loaded on demand.
    import numpy as np


@dataclass(slots=True)
class Segment:
    start: float
    end: float
    text: str
    # None means the backend did not report this measurement. Zero is a
    # real measurement and must not stand in for missing evidence.
    avg_logprob: float | None = None
    no_speech_prob: float | None = None
    compression_ratio: float = 0.0
    temperature: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class Transcript:
    text: str
    segments: list[Segment] = field(default_factory=list)
    language: str = ""
    language_probability: float = 0.0
    audio_seconds: float = 0.0
    decode_seconds: float = 0.0
    model: str = ""
    device: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def rtf(self) -> float:
        """Real-time factor. 0.03 means 30x faster than the speech itself."""
        return self.decode_seconds / self.audio_seconds if self.audio_seconds else 0.0

    @property
    def min_avg_logprob(self) -> float | None:
        """The least confident segment — the one most likely to have dropped words."""
        return min((s.avg_logprob for s in self.segments if s.avg_logprob is not None), default=None)

    @property
    def max_no_speech(self) -> float | None:
        """Diagnostic only: one silent segment says nothing about the rest."""
        return max((s.no_speech_prob for s in self.segments if s.no_speech_prob is not None), default=None)

    def warnings(self, *, logprob_floor: float = -1.0) -> list[str]:
        """Human-readable reasons to distrust this transcript."""
        out: list[str] = []
        minimum = self.min_avg_logprob
        maximum = self.max_no_speech
        if minimum is not None and minimum < logprob_floor:
            out.append(f"low confidence: avg_logprob={minimum:.2f} — words may be wrong or missing")
        if maximum is not None and maximum > 0.6:
            out.append(f"a segment may be silence: no_speech_prob={maximum:.2f} — inspect its text")
        return out

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": round(self.language_probability, 4),
            "audio_seconds": round(self.audio_seconds, 3),
            "decode_seconds": round(self.decode_seconds, 3),
            "rtf": round(self.rtf, 4),
            "model": self.model,
            "device": self.device,
            "segments": [s.as_dict() for s in self.segments],
            "extra": self.extra,
        }


class NotReady(RuntimeError):
    """The machine is not set up for this backend yet.

    Distinct from a transient failure because it will not fix itself: a
    missing package, absent weights, an API backend with no key. The daemon
    exits with a code systemd is told not to retry, so a fresh install shows
    one actionable line instead of five stack traces and a start-limit.
    """


class Backend(Protocol):
    name: str

    def load(self) -> None:
        """Prepare the backend. Called once at daemon start."""

    def describe(self) -> str:
        """One line for `omavoi status`."""

    def use(self, model_key: str) -> None:
        """Load a different model, if this backend can.

        A mode may name its own speech model — turbo for a terminal, large for
        prose — and switching modes should switch the weights rather than ask
        for a restart. Backends that cannot swap raise NotReady and say so.
        """

    def state(self) -> dict[str, Any]:
        """What is actually running, structured, for the UI.

        `live` is the field that matters and the one nothing else reports: a
        configured model and a loaded model are different things, and the gap
        between them is where a whole afternoon goes.
        """

    def transcribe(
        self,
        samples: np.ndarray,
        rate: int,
        *,
        language: str | None = None,
        prompt: str | None = None,
    ) -> Transcript: ...
