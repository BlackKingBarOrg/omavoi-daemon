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



def _opens_mid_speech(capture: Capture, *, head_ms: float = 120.0,
                      margin_db: float = 6.0) -> str:
    """"" if the take opens on room tone, a warning if it opens on speech.

    The old test asked the transcript where its first segment began, which is
    where whisper.cpp always begins one — 0.00 on 38 of 40 takes in a row,
    none of them actually clipped. This asks the only thing that can answer
    it: whether the first 120 ms is already as loud as the take as a whole. A
    quiet head is the pre-roll doing its job.
    """
    rate = int(getattr(capture, "rate", 0) or 0)
    samples = getattr(capture, "samples", None)
    if not rate or samples is None or len(samples) < rate // 4:
        return ""
    # A floor of its own, so a direct caller gets the same answer the pipeline
    # does without having to know to check for silence first.
    floor_db = -50.0
    head = samples[: max(1, int(rate * head_ms / 1000.0))]
    if head.size == 0:
        return ""

    def dbfs(block) -> float:
        rms = float(np.sqrt(np.mean(np.square(block.astype(np.float64)))))
        return 20.0 * math.log10(max(rms, 1e-9))

    head_db, whole_db = dbfs(head), dbfs(samples)
    if whole_db < floor_db:
        return ""
    if head_db < whole_db - margin_db:
        return ""
    return (f"the take opens at speech level ({head_db:.1f} dBFS against "
            f"{whole_db:.1f} overall) — the first syllable may be missing; "
            f"raise audio.preroll_seconds if words are going astray")

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
        # Only on a take that has speech in it. Uniform silence has a head as
        # loud as its whole and would trip this, but nothing was clipped — the
        # quiet-input warning above is the right voice for that, and one
        # problem should not get two.
        if not quiet:
            onset = _opens_mid_speech(capture)
            if onset:
                warnings.append(onset)

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
