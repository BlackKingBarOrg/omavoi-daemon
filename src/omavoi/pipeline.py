"""Run a take through its mode's chain: speech -> rules -> LLM steps -> inject.

Both the daemon and `omavoi transcribe <file>` go through here, so an
offline re-run of a bad take exercises exactly the same path.

The one rule that governs the whole file: a later stage may improve the
text, never destroy it. Every LLM step falls through to what it was given
if it fails, times out, or comes back empty — a slow model degrades your
dictation, it must not swallow it.
"""

from __future__ import annotations

import logging
import math
import re
import shutil
import time
from typing import Any

import numpy as np

from . import asr, modes, notify, post
from . import names as names_mod
from .audio import Capture
from .history import History
from .inject import Injector
from .llm import Registry
from .modes import Mode
from .window import Window, active_window

log = logging.getLogger(__name__)



def _head_level(capture: Capture, *, head_ms: float = 120.0) -> float | None:
    """The first `head_ms` of the take, in dBFS, or None if unmeasurable.

    Recorded next to the take because it is what the onset warning is built
    on, and a warning whose evidence is not visible is one you cannot check.
    The property this replaces claimed to be that evidence and never appeared
    in a diagnostic at all.
    """
    rate = int(getattr(capture, "rate", 0) or 0)
    samples = getattr(capture, "samples", None)
    if not rate or samples is None or len(samples) < rate // 4:
        return None
    head = samples[: max(1, int(rate * head_ms / 1000.0))]
    if head.size == 0:
        return None
    rms = float(np.sqrt(np.mean(np.square(head.astype(np.float64)))))
    return round(20.0 * math.log10(max(rms, 1e-9)), 1)


# There was a warning here that compared the take's first 120 ms against the
# take as a whole and said the onset looked clipped when they matched. It
# replaced an older one built on whisper's segmentation, which fired on 38 of
# 40 takes, and it was a real improvement — until a take picked up a video
# playing in the background. Continuous sound makes the head match the whole
# whatever the pre-roll did, so it fired again, correctly and uselessly: the
# take did open at speech level, and nothing had been clipped.
#
# Whether the onset survived is not inferable from the waveform once the room
# is not quiet, which on a machine with a browser open is most of the time.
# So there is no warning. head_dbfs goes into the record next to rms_dbfs and
# anyone who suspects a clipped onset can compare them, which is the honest
# amount to claim.

class Pipeline:
    def __init__(
        self,
        cfg: dict[str, Any],
        backend: asr.Backend,
        injector: Injector | None = None,
        history: History | None = None,
        registry: Registry | None = None,
    ) -> None:
        self.cfg = cfg
        self.backend = backend
        self.injector = injector or Injector(cfg)
        self.history = history or History(cfg)
        self.llms = registry or Registry(cfg)

    # -- chain -------------------------------------------------------------

    def _run_steps(self, text: str, mode: Mode, entry: dict[str, Any]) -> str:
        """Apply each LLM pass in order, keeping the last good text."""
        records: list[dict[str, Any]] = entry.setdefault("steps", [])
        current = text

        for index, step in enumerate(mode.steps):
            backend = self.llms.get(step.llm, getattr(step, "model", "") or "")
            if backend is None:
                why = self.llms.why(step.llm)
                records.append({"llm": step.llm, "error": why, "kept": True})
                entry["warnings"].append(f"step {index + 1}: llm {step.llm!r} unavailable — {why}")
                continue

            result = backend.complete(step.prompt, current, timeout=step.timeout)
            record = result.as_dict() | {"llm": step.llm, "index": index}
            if result.ok:
                record["before"] = current
                current = result.text
            else:
                record["kept"] = True
                reason = result.error or "empty response"
                entry["warnings"].append(f"step {index + 1} ({step.llm}) fell through: {reason}")
                log.warning("llm step %s failed, keeping previous text: %s", step.llm, reason)
            records.append(record)

        return current

    # -- one take ----------------------------------------------------------

    def process(
        self,
        capture: Capture,
        *,
        inject: bool = True,
        window: Window | None = None,
        forced_mode: str = "",
    ) -> dict[str, Any]:
        started = time.monotonic()
        cfg = self.cfg
        entry: dict[str, Any] = {
            "ts": time.time(),
            "audio": {
                "seconds": round(capture.seconds, 3),
                "peak_dbfs": round(capture.peak_dbfs, 1),
                "rms_dbfs": round(capture.rms_dbfs, 1),
                # The first 120 ms, which is what the onset warning is built
                # on. Recorded because a warning whose evidence is not
                # visible is one you cannot check.
                "head_dbfs": _head_level(capture),
                "preroll": capture.preroll_seconds,
                "tail": capture.tail_seconds,
                "truncated": capture.truncated,
            },
            "raw_text": "",
            "text": "",
            "warnings": [],
            "rejected": "",
        }
        warnings: list[str] = entry["warnings"]

        min_seconds = float(cfg["audio"]["min_seconds"])
        if capture.seconds < min_seconds:
            entry["rejected"] = (
                f"only {capture.seconds:.2f}s, below audio.min_seconds={min_seconds}"
            )
            return self._finish(entry, capture, started, notify_empty=False)

        warn_rms = float(cfg["audio"]["warn_rms_dbfs"])
        quiet = capture.rms_dbfs < warn_rms
        if quiet:
            warnings.append(
                f"input is quiet: rms {capture.rms_dbfs:.1f} dBFS, below {warn_rms} "
                "— the usual cause of dropped words"
            )
        if capture.truncated:
            warnings.append("the ring buffer wrapped; the start of this take was lost")
        # Whether the take opens mid-word, asked of the waveform. The pre-roll
        # reaches back before the keypress, so a healthy take begins with room
        # tone; an opening already at speech level means the buffer window
        # started after the first syllable.

        # Resolve where the text is going now: that decides the mode.
        win = window if window is not None else active_window()
        mode = modes.resolve(cfg, win, forced_mode)
        entry["window"] = win.as_dict()
        entry["mode"] = mode.as_dict()

        # Names are seeded into the decoder prompt: getting the model to
        # produce a name is cheaper and cleaner than fixing it afterwards.
        index = names_mod.NameIndex(cfg, mode.name)
        seed = index.seed_text() if mode.rules.get("names", True) else ""
        prompt = "\n".join(p for p in (mode.prompt, seed) if p)

        try:
            transcript = self.backend.transcribe(
                capture.samples,
                capture.rate,
                language=mode.language or None,
                prompt=prompt or None,
            )
        except Exception as exc:
            log.exception("transcription failed")
            entry["rejected"] = f"transcription failed: {exc}"
            return self._finish(entry, capture, started, notify_empty=True)

        entry["asr"] = transcript.as_dict()
        entry["raw_text"] = transcript.text
        warnings.extend(transcript.warnings())

        result = post.run(
            transcript.text,
            cfg,
            post.Context(win.cls, win.title, mode.rules),
            max_no_speech=transcript.max_no_speech,
            quiet=quiet,
        )
        entry["post"] = result.as_dict()

        # Whatever seeding did not catch, sound matching recovers here.
        if result.text and mode.rules.get("names", True):
            fixed, hits = index.apply(result.text)
            if hits:
                entry["names"] = [
                    {"name": h.name, "found": h.found, "score": round(h.score, 3)}
                    for h in hits
                ]
                entry["post"]["changes"].append(
                    "names: " + ", ".join(f"{h.found}->{h.name}" for h in hits)
                )
                result.text = fixed
        entry["rules_text"] = result.text
        if result.rejected:
            entry["text"] = ""
            entry["rejected"] = result.rejected
            return self._finish(entry, capture, started, notify_empty=True)

        final = self._run_steps(result.text, mode, entry) if mode.steps else result.text

        # Fold newlines last, whoever produced them. The rule that turns a
        # segment break into punctuation runs before the LLM, so an LLM asked
        # for two lines used to hand them straight to injection — and a
        # newline typed into a chat window is the send key, which is how a
        # bilingual take ended up posting only its first line.
        if mode.joiner != "keep" and "\n" in final:
            if mode.joiner == " ":
                # Script-aware, because a space is a Latin convention. Full-
                # width punctuation carries its own spacing, so two CJK lines
                # butt up and two Latin lines take a space — which is the
                # rule normalise_boundaries already applies to the raw
                # transcript, while this line put a space into
                # 这是第一句。这是第二句。 on the way out of an LLM step.
                # add_punctuation off: the model wrote whole sentences.
                joined = post.normalise_boundaries(
                    final, newlines="space", add_punctuation=False)
            else:
                # An explicit joiner is the one that was asked for.
                joined = re.sub(r"\s*\n+\s*", mode.joiner, final).strip()
            if joined != final:
                entry.setdefault("post", {}).setdefault("changes", []).append(
                    f"newlines folded with {mode.joiner!r}"
                )
                final = joined
        entry["text"] = final

        if inject and final and win.xwayland and not shutil.which("xdotool"):
            warnings.append(
                "this is an X11 window and xdotool is not installed — the paste "
                "keystroke will not reach it (sudo pacman -S xdotool)"
            )

        if inject and final:
            profile: dict[str, Any] = {"inject": mode.inject}
            if mode.paste_key:
                profile["paste_key"] = mode.paste_key
            outcome = self.injector.inject(final, win, profile)
            entry["inject"] = outcome.as_dict()
            if not outcome.ok:
                warnings.append(f"injection failed: {outcome.error}")
            elif outcome.fell_back:
                warnings.append(f"injection fell back to {outcome.method}")

        return self._finish(entry, capture, started, notify_empty=False)

    # -- reporting ---------------------------------------------------------

    def _finish(self, entry: dict[str, Any], capture: Capture, started: float,
                *, notify_empty: bool) -> dict[str, Any]:
        entry["total_seconds"] = round(time.monotonic() - started, 3)
        self.history.record(entry, capture.samples, capture.rate)

        for warning in entry["warnings"]:
            log.warning("%s", warning)
        if self.cfg["ui"].get("notify", True):
            self._notify(entry, notify_empty=notify_empty)
        if entry["text"]:
            log.info("typed in %.2fs [%s]: %s",
                     entry["total_seconds"], entry.get("mode", {}).get("name", "?"), entry["text"])
        elif entry["rejected"]:
            log.info("dropped: %s", entry["rejected"])
        return entry

    def _notify(self, entry: dict[str, Any], *, notify_empty: bool) -> None:
        if entry["text"]:
            if entry["warnings"]:
                notify.send("Omavoi", entry["warnings"][0], urgency="normal")
            return
        if notify_empty and self.cfg["ui"].get("notify_on_empty", True):
            notify.send("Omavoi heard nothing", entry["rejected"], urgency="low")
