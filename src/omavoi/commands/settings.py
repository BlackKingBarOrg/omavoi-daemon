"""Reading and writing the config, and refusing what it cannot use.

Lifted out of cli.py, which had grown to 1,949 lines and four
responsibilities. Nothing here changed on the way across.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from typing import Any

from .. import config, paths
from ..term import DIM, GREEN, RED, RESET, YELLOW
from .catalogue import _why_not_that_llm_model, _why_not_that_provider
from .keys import _why_not_that_hotkey


def _legal_values(key: str, cfg: dict[str, Any]) -> tuple[tuple[str, ...], str]:
    """The values a key accepts, when it accepts a fixed set of them.

    Returns (values, what) or ((), "").

    152 settable keys and four of them were checked. The rest took anything —
    `speech.backend = nonsens` was written happily and then stopped the daemon
    from starting at all, and `switching.mode = typo` fell back to default with
    one line in a journal nobody reads. Every set below is read from wherever
    the code already keeps it, so this table cannot drift away from the thing
    it is describing.
    """
    from .. import asr, i18n
    from ..llm import BACKENDS as LLM_BACKENDS

    parts = key.split(".")

    def at(*shape: str) -> bool:
        """`llm.*.backend` matches ("llm", "*", "backend")."""
        return (len(parts) == len(shape)
                and all(a == "*" or a == b for a, b in zip(shape, parts, strict=True)))

    if key == "speech.backend":
        return tuple(sorted(asr.BACKENDS)), "a speech engine"
    if at("llm", "*", "backend"):
        return tuple(sorted(LLM_BACKENDS)), "an LLM backend"
    if key == "hotkey.mode":
        return ("push_to_talk", "toggle"), "a hotkey behaviour"
    if key == "inject.method" or at("modes", "*", "inject"):
        return ("auto", "clipboard", "wtype", "xdotool"), "an injection route"
    if at("modes", "*", "rules", "punctuation"):
        return ("keep", "strip"), "a punctuation policy"
    if key == "ui.hud_dwell":
        return ("always", "changed", "never"), "an overlay dwell"
    if key == "ui.language":
        # "" follows the system locale, which is the shipped default.
        return ("", *i18n.LANGUAGES), "an interface language"
    if key == "speech.local_whisper.device":
        return ("auto", "cpu", "cuda"), "a device"
    if key == "switching.mode":
        return tuple(sorted(cfg.get("modes", {}))), "a mode that exists"
    if key == "ui.hud_size":
        return ("xs", "s", "m"), "an overlay size"
    if key == "ui.hud_position":
        # One value, because one is implemented. "cursor" and "window" were in
        # the defaults comment and nothing built them, so they were accepted
        # and ignored.
        return ("bottom",), "an overlay position"
    if key == "inject.paste_method" or at("modes", "*", "paste_method"):
        # What _send_paste dispatches on. Anything else fell through to the
        # compositor shortcut silently, so a typo looked like it worked.
        return ("", "shortcut", "wtype", "xdotool"), "a paste route"
    if key == "speech.api.response_format":
        # ApiWhisperBackend calls response.json() unconditionally, so a format
        # that is not JSON is not a preference, it is a crash on the next take.
        return ("json", "verbose_json"), "a response format the reader parses"
    if key == "audio.rate":
        # config.check already says "whisper needs 16000 Hz" on every load. It
        # said it as a warning and left the value in place, so the daemon went
        # on feeding whisper audio at the wrong rate.
        return ("16000",), "a rate whisper can use"
    return (), ""


# Keys whose value is a number with a range, and what happens outside it.
# `config set audio.preroll_seconds 99` was accepted — ninety-nine seconds of
# pre-roll, from a ring buffer that does not hold it — and so was
# `warn_rms_dbfs 500`, a positive number for a quantity that is negative by
# definition. Neither produced an error anywhere; they simply made the program
# behave oddly.
_RANGES: dict[str, tuple[float, float, str]] = {
    "audio.preroll_seconds": (0.0, 5.0, "seconds of audio kept before the key"),
    "audio.tail_seconds": (0.0, 2.0, "seconds kept after the key is released"),
    "audio.warn_rms_dbfs": (-90.0, 0.0, "dBFS, which is negative below full scale"),
    "audio.max_seconds": (5.0, 3600.0, "seconds one take may run"),
    "audio.min_seconds": (0.0, 10.0, "seconds below which a take is dropped"),
    "history.keep_audio": (0.0, 500.0, "recordings kept on disk"),
    "hotkey.rescan_seconds": (0.5, 60.0, "seconds between device rescans"),
}


def _why_not_that_number(key: str, value: str) -> str:
    """Why this number is outside what the key can use, or "" if it is not."""
    limits = _RANGES.get(str(key))
    if limits is None:
        return ""
    low, high, what = limits
    try:
        n = float(value)
    except (TypeError, ValueError):
        return f"{value!r} is not a number. {key} is {what}"
    if n < low or n > high:
        return (f"{value} is outside {low:g}–{high:g} — {key} is {what}")
    return ""


def _why_not_that_level(key: str, value: str) -> str:
    """Why this is not a logging level.

    `ui.log_level = SHOUT` was accepted and then silently became INFO, because
    setup_logging reads it with `getattr(logging, level.upper(), logging.INFO)`.
    Nothing was broken by it and nothing was true either: the config said one
    thing and the logger did another.

    Not in _legal_values because the reader uppercases, so both cases are
    genuinely legal and a list of ten values in two spellings is noise. The
    names come from logging itself, so this cannot drift.
    """
    if key != "ui.log_level":
        return ""
    names = {n for n in logging.getLevelNamesMapping() if n != "NOTSET"}
    if value.upper() in names:
        return ""
    shown = ", ".join(sorted(names, key=lambda n: logging.getLevelNamesMapping()[n]))
    return f"{value!r} is not a logging level. One of: {shown}"


def _why_not_that_language(key: str, value: str) -> str:
    """Why this is not shaped like a speech language.

    Whisper takes an ISO-639 code, and `speech.language = chinese` or `zh-CN`
    was written happily — then the endpoint either 400s or quietly ignores it,
    on a take, minutes later, with nothing pointing back here.

    A shape check and not a list: the 99 codes whisper knows are whisper's,
    not ours, and hardcoding them here would be a copy that goes stale. The
    shape catches every realistic mistake — a language name, a locale, a
    region suffix — and lets through only invented two-letter codes, which the
    engine itself will then name.
    """
    if key != "speech.language" and not (
        key.startswith("modes.") and key.endswith(".language")
    ):
        return ""
    if value in ("", "auto") or (2 <= len(value) <= 3 and value.isalpha() and value.islower()):
        return ""
    return (
        f"{value!r} is not a language code. Use a 2- or 3-letter ISO-639 code "
        f"(en, zh, ja, yue), 'auto' to detect it, or '' to follow the mode"
    )


def _why_not_that_choice(key: str, value: str) -> str:
    """Why this value is not one of the ones the key accepts."""
    cfg = config.load()
    legal, what = _legal_values(str(key), cfg)
    if not legal or value in legal:
        return ""
    shown = ", ".join(repr(v) if v == "" else v for v in legal)
    return f"{value!r} is not {what}. One of: {shown}"


def cmd_config(args: argparse.Namespace) -> int:
    path = paths.config_file()
    if args.action == "path":
        print(path)
        return 0
    if args.action == "init":
        if path.exists() and not args.force:
            print(f"{YELLOW}{path} exists; pass --force to overwrite{RESET}", file=sys.stderr)
            return 1
        config.write(config.defaults(), path)
        print(f"{GREEN}wrote{RESET} {path}")
        return 0
    if args.action == "show":
        if args.json:
            print(json.dumps(config.load(), ensure_ascii=False, indent=2))
        else:
            print(config.dumps(config.load()))
        return 0
    if args.action == "get":
        try:
            print(json.dumps(config.get_path(config.load(), args.key), ensure_ascii=False))
        except KeyError:
            print(f"{RED}no such setting: {args.key}{RESET}", file=sys.stderr)
            return 1
        return 0
    if args.action == "set":
        # Pointing an LLM at weights that are not there is accepted by the
        # config and then falls through on every take, which reads as the LLM
        # step doing nothing rather than as a missing download.
        why = _why_not_that_hotkey(args.key, args.value) \
            or _why_not_that_llm_model(args.key, args.value) \
            or _why_not_that_provider(args.key, args.value) \
            or _why_not_that_level(args.key, args.value) \
            or _why_not_that_language(args.key, args.value) \
            or _why_not_that_choice(args.key, args.value) \
            or _why_not_that_number(args.key, args.value)
        if why and not getattr(args, "force", False):
            print(f"{RED}{why}{RESET}", file=sys.stderr)
            print(f"{DIM}nothing was changed; repeat with --force to set it anyway"
                  f"{RESET}", file=sys.stderr)
            return 1
        try:
            value = config.set_path(args.key, args.value)
        except KeyError:
            print(f"{RED}no such setting: {args.key}{RESET}", file=sys.stderr)
            return 1
        except ValueError as exc:
            print(f"{RED}{exc}{RESET}", file=sys.stderr)
            return 1
        print(f"{GREEN}ok{RESET} {args.key} = {json.dumps(value, ensure_ascii=False)}")
        return 0
    if args.action == "edit":
        if not path.exists():
            config.write(config.defaults(), path)
        editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "nano"
        return subprocess.call([editor, str(path)])
    return 1
